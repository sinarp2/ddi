# -*- coding: utf-8 -*-
r"""
DDI 자동 산출 엔진
=================

한국어 AI 평가 문항의 설계난이도 지수(Design Difficulty Index, DDI)를
문항 단위로 자동 산출하는 단일 파일 구현이다.

산출 벡터:
D_i^auto = (R_i, C_i, K_i, P_i, E_i, O_i, X_i, D_i^dist, M_i, Q_i)

- R_i: Reasoning, 추론 단계 수와 추론 복잡도
- C_i: Context, 맥락 의존도
- K_i: Korean-specificity, 한국어 특성 부하량
- P_i: Pragmatics, 화용 및 사회언어학 부하량
- E_i: External knowledge, 외부 지식 의존도
- O_i: Openness, 응답 개방성
- X_i: Constraint complexity, 제약 조건 복잡도
- D_i^dist: Distractor plausibility, 오답 매력도
- M_i: Multi-construct integration, 다중 구인 결합도
- Q_i: Question clarity, 문항 명료도

산식:
DDI_global = 100 * Σ(w_p * z_ip)
DDI_domain = 100 * Σ(w_d,p * z_ip)
DDI_auto = λ * DDI_global + (1 - λ) * DDI_domain
DDI_clean = DDI_auto * (0.5 + 0.5 * Q_i)

Windows PowerShell:
python .\ddi_auto_engine.py --demo
python .\ddi_auto_engine.py --input .\sample_items.jsonl --output .\ddi_results.jsonl
python .\ddi_auto_engine.py --input .\sample_items.jsonl --output .\ddi_results.csv --format csv
python .\ddi_auto_engine.py --input .\sample_items.jsonl --anchors .\anchors.jsonl --output .\ddi_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


COMPONENTS = ("R", "C", "K", "P", "E", "O", "X", "Ddist", "M", "Q")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.fmean(values) if values else default


def safe_stdev(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else default


def count_nonspace(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[가-힣]+|[A-Za-z]+|\d+(?:\.\d+)?", (text or "").lower())


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalized_length_score(length: int, midpoint: int, scale: int) -> float:
    if scale <= 0:
        return 0.0
    x = (length - midpoint) / scale
    return 1.0 / (1.0 + math.exp(-x))


def keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    text = text or ""
    return [kw for kw in keywords if kw in text]


def regex_hits(text: str, patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            out.append(pattern)
    return out


@dataclass
class DDIItem:
    item_id: str
    passage: str = ""
    question: str = ""
    options: List[str] = field(default_factory=list)
    answer_key: Any = None
    rationale: str = ""
    rubric: str = ""
    response_type: str = "constructed_response"
    domain: str = "general"
    subdomains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DDIItem":
        item_id = str(data.get("item_id") or data.get("id") or "").strip()
        if not item_id:
            raise ValueError("item_id 또는 id가 필요합니다.")

        options = data.get("options") or []
        if isinstance(options, Mapping):
            options = [str(v) for _, v in sorted(options.items(), key=lambda kv: str(kv[0]))]
        elif not isinstance(options, list):
            options = [str(options)]

        subdomains = data.get("subdomains") or []
        if isinstance(subdomains, str):
            subdomains = [s.strip() for s in re.split(r"[,;/]", subdomains) if s.strip()]

        return cls(
            item_id=item_id,
            passage=str(data.get("passage") or data.get("context") or ""),
            question=str(data.get("question") or data.get("stem") or ""),
            options=[str(x) for x in options],
            answer_key=data.get("answer_key"),
            rationale=str(data.get("rationale") or data.get("explanation") or ""),
            rubric=str(data.get("rubric") or ""),
            response_type=str(data.get("response_type") or "constructed_response").lower(),
            domain=str(data.get("domain") or "general").lower(),
            subdomains=[str(x).lower() for x in subdomains],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ComponentScore:
    value: float
    confidence: float
    applicable: bool = True
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DDIConfig:
    global_weights: Dict[str, float] = field(default_factory=lambda: {
        "R": 0.18, "C": 0.12, "K": 0.13, "P": 0.13, "E": 0.08,
        "O": 0.08, "X": 0.11, "Ddist": 0.07, "M": 0.10,
    })
    domain_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "honorifics": {
            "R": 0.13, "C": 0.12, "K": 0.19, "P": 0.22, "E": 0.04,
            "O": 0.07, "X": 0.08, "Ddist": 0.05, "M": 0.10,
        },
        "pragmatics": {
            "R": 0.14, "C": 0.14, "K": 0.10, "P": 0.23, "E": 0.06,
            "O": 0.08, "X": 0.09, "Ddist": 0.05, "M": 0.11,
        },
        "discourse": {
            "R": 0.18, "C": 0.20, "K": 0.10, "P": 0.08, "E": 0.05,
            "O": 0.07, "X": 0.09, "Ddist": 0.06, "M": 0.17,
        },
        "orthography": {
            "R": 0.12, "C": 0.08, "K": 0.26, "P": 0.04, "E": 0.04,
            "O": 0.05, "X": 0.10, "Ddist": 0.16, "M": 0.15,
        },
        "dialect": {
            "R": 0.13, "C": 0.14, "K": 0.20, "P": 0.10, "E": 0.15,
            "O": 0.06, "X": 0.07, "Ddist": 0.06, "M": 0.09,
        },
        "archaic": {
            "R": 0.15, "C": 0.13, "K": 0.18, "P": 0.06, "E": 0.20,
            "O": 0.06, "X": 0.07, "Ddist": 0.06, "M": 0.09,
        },
        "inference": {
            "R": 0.25, "C": 0.18, "K": 0.08, "P": 0.08, "E": 0.08,
            "O": 0.07, "X": 0.09, "Ddist": 0.06, "M": 0.11,
        },
        "business_email": {
            "R": 0.15, "C": 0.12, "K": 0.12, "P": 0.21, "E": 0.04,
            "O": 0.10, "X": 0.12, "Ddist": 0.03, "M": 0.11,
        },
    })
    lambda_global: float = 0.50
    rule_weight: float = 0.65
    panel_weight: float = 0.35
    clarity_floor: float = 0.50
    l1_upper: float = 35.0
    l2_upper: float = 70.0
    min_anchor_count_for_domain_calibration: int = 5

    def validate(self) -> None:
        if not math.isclose(sum(self.global_weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("global_weights의 합은 1이어야 합니다.")
        for domain, weights in self.domain_weights.items():
            if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
                raise ValueError(f"{domain} domain_weights의 합은 1이어야 합니다.")
        if not math.isclose(self.rule_weight + self.panel_weight, 1.0, abs_tol=1e-6):
            raise ValueError("rule_weight + panel_weight의 합은 1이어야 합니다.")


@dataclass
class CalibrationModel:
    slope: float = 1.0
    intercept: float = 0.0
    n: int = 0

    def predict(self, x: float) -> float:
        return max(0.0, min(100.0, self.slope * x + self.intercept))


class AnchorCalibrator:
    """
    자동 DDI와 전문가 앵커 난이도 간 선형 보정기.

    앵커 JSONL 예:
    {"item_id":"A001","domain":"business_email","ddi_clean":61.2,"expert_ddi":67.0}
    """

    def __init__(self, min_domain_n: int = 5) -> None:
        self.min_domain_n = min_domain_n
        self.global_model = CalibrationModel()
        self.domain_models: Dict[str, CalibrationModel] = {}

    @staticmethod
    def _fit_xy(xs: Sequence[float], ys: Sequence[float]) -> CalibrationModel:
        if len(xs) != len(ys) or not xs:
            return CalibrationModel()
        if len(xs) == 1:
            return CalibrationModel(slope=1.0, intercept=ys[0] - xs[0], n=1)

        mx, my = safe_mean(xs), safe_mean(ys)
        var_x = sum((x - mx) ** 2 for x in xs)
        if var_x < 1e-12:
            return CalibrationModel(slope=1.0, intercept=my - mx, n=len(xs))
        cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = cov_xy / var_x
        intercept = my - slope * mx
        return CalibrationModel(slope=slope, intercept=intercept, n=len(xs))

    def fit(self, anchors: Sequence[Mapping[str, Any]]) -> None:
        global_x, global_y = [], []
        by_domain: Dict[str, Tuple[List[float], List[float]]] = {}

        for row in anchors:
            try:
                x = float(row["ddi_clean"])
                y = float(row["expert_ddi"])
            except (KeyError, TypeError, ValueError):
                continue
            domain = str(row.get("domain") or "general").lower()
            global_x.append(x)
            global_y.append(y)
            by_domain.setdefault(domain, ([], []))
            by_domain[domain][0].append(x)
            by_domain[domain][1].append(y)

        self.global_model = self._fit_xy(global_x, global_y)
        for domain, (xs, ys) in by_domain.items():
            if len(xs) >= self.min_domain_n:
                self.domain_models[domain] = self._fit_xy(xs, ys)

    def predict(self, ddi_clean: float, domain: str) -> Tuple[float, str]:
        domain = (domain or "general").lower()
        if domain in self.domain_models:
            return self.domain_models[domain].predict(ddi_clean), f"domain:{domain}"
        return self.global_model.predict(ddi_clean), "global"


class KoreanRuleAnalyzer:
    REASONING_TERMS = (
        "분석하시오", "설명하시오", "비교하시오", "평가하시오", "추론하시오",
        "근거", "이유", "종합", "판단", "도출", "비판", "재구성", "수정하시오",
        "분류하시오", "해석하시오", "예측하시오",
    )
    MULTISTEP_TERMS = (
        "각각", "모두", "먼저", "다음으로", "그리고", "그 후", "동시에",
        "관점에서", "반영하여", "고치고", "설명하고", "비교한 뒤",
    )
    CONTEXT_TERMS = (
        "위", "아래", "앞서", "후술", "그", "이", "저", "해당", "이러한",
        "그러한", "당시", "여기", "거기", "이때", "그때", "앞 문장",
        "뒷문장", "대화", "상황", "문맥",
    )
    KOREAN_SPECIFIC_TERMS = (
        "맞춤법", "띄어쓰기", "조사", "어미", "높임", "경어", "존댓말",
        "반말", "주체 높임", "객체 높임", "상대 높임", "문장성분 호응",
        "관용표현", "유의어", "고어", "방언", "신조어", "의성어", "의태어",
        "화시", "무형대용어", "생략", "레지스터", "문체",
    )
    PRAGMATIC_TERMS = (
        "공손", "완곡", "화행", "요청", "거절", "사과", "책임", "압박",
        "배려", "상급자", "하급자", "자문위원", "고객", "민원인", "상대방",
        "공식", "비공식", "업무 메일", "공문", "격식", "관계", "체면",
    )
    EXTERNAL_KNOWLEDGE_TERMS = (
        "역사", "법률", "제도", "정책", "문화", "관습", "지역", "전문지식",
        "배경지식", "외부 지식", "사전 지식", "기관 규정", "법령", "표준",
    )
    CONSTRAINT_PATTERNS = (
        r"\b\d+\s*문장", r"\b\d+\s*자", r"\b\d+\s*개", r"\b\d+\s*가지",
        r"\b\d+\s*단계", r"이내", r"이상", r"이하", r"반드시", r"모두",
        r"포함하", r"제외하", r"사용하지", r"형식", r"표로", r"목록으로",
        r"JSON", r"마크다운", r"과잉 사과", r"책임 회피",
    )
    AMBIGUOUS_TERMS = (
        "적절히", "알아서", "등등", "가능한 한", "보통", "대충",
        "괜찮게", "잘", "좋게", "자연스럽게만", "필요한 만큼",
    )

    def analyze(self, item: DDIItem) -> Dict[str, ComponentScore]:
        combined = "\n".join(
            part for part in (item.passage, item.question, item.rationale, item.rubric) if part
        )
        return {
            "R": self._reasoning(item),
            "C": self._context(item, combined),
            "K": self._korean_specificity(item, combined),
            "P": self._pragmatics(item, combined),
            "E": self._external_knowledge(item, combined),
            "O": self._openness(item),
            "X": self._constraints(item),
            "Ddist": self._distractor_plausibility(item),
            "M": self._multi_construct(item, combined),
            "Q": self._clarity(item),
        }

    def _reasoning(self, item: DDIItem) -> ComponentScore:
        hits = keyword_hits(item.question, self.REASONING_TERMS)
        multistep = keyword_hits(item.question, self.MULTISTEP_TERMS)
        sentences = len(split_sentences(item.passage))
        subtasks = len(re.findall(r"(?:^|\n|\s)(?:\d+[\.\)]|[가-힣][\.\)])\s*", item.question))
        score = (
            0.12
            + 0.10 * min(len(hits), 4)
            + 0.08 * min(len(multistep), 4)
            + 0.07 * min(subtasks, 4)
            + 0.16 * normalized_length_score(count_nonspace(item.passage), 350, 180)
            + 0.08 * min(sentences / 8.0, 1.0)
        )
        evidence = []
        if hits:
            evidence.append("추론 동사: " + ", ".join(hits[:6]))
        if multistep:
            evidence.append("복합 과업 표지: " + ", ".join(multistep[:6]))
        evidence.append(f"명시적 하위 과업 수: {subtasks}")
        evidence.append(f"지문 문장 수: {sentences}")
        return ComponentScore(clamp(score), 0.72 if item.question else 0.35, evidence=evidence)

    def _context(self, item: DDIItem, text: str) -> ComponentScore:
        hits = keyword_hits(text, self.CONTEXT_TERMS)
        chars = count_nonspace(item.passage)
        sentences = len(split_sentences(item.passage))
        turns = len(re.findall(r"(?:^|\n)\s*[^:\n]{1,20}:", item.passage))
        score = (
            0.05
            + 0.28 * normalized_length_score(chars, 250, 170)
            + 0.18 * min(sentences / 10.0, 1.0)
            + 0.08 * min(len(hits) / 5.0, 1.0)
            + 0.12 * min(turns / 4.0, 1.0)
        )
        evidence = [f"지문 길이={chars}", f"문장 수={sentences}"]
        if hits:
            evidence.append("맥락 의존 표지: " + ", ".join(hits[:8]))
        return ComponentScore(clamp(score), 0.78 if item.passage else 0.45, evidence=evidence)

    def _korean_specificity(self, item: DDIItem, text: str) -> ComponentScore:
        hits = keyword_hits(text, self.KOREAN_SPECIFIC_TERMS)
        endings = regex_hits(
            text,
            (r"(으)?시(?:겠|었|는|ㄴ|ㅂ)", r"(드리|여쭙|모시|뵙)",
             r"(습니다|십시오|세요|해요|한다|하였다)", r"(은|는|이|가|을|를|에게|께서|께)")
        )
        domain_bonus = 0.20 if item.domain in {
            "honorifics", "orthography", "dialect", "archaic", "korean",
            "business_email", "pragmatics"
        } else 0.0
        score = 0.04 + domain_bonus + 0.08 * min(len(hits), 5) + 0.05 * min(len(endings), 4)
        evidence = []
        if hits:
            evidence.append("한국어 특수 구인: " + ", ".join(hits[:8]))
        if endings:
            evidence.append(f"조사·어미·높임 패턴 수={len(endings)}")
        return ComponentScore(clamp(score), 0.76, evidence=evidence)

    def _pragmatics(self, item: DDIItem, text: str) -> ComponentScore:
        hits = keyword_hits(text, self.PRAGMATIC_TERMS)
        roles = len(set(keyword_hits(text, (
            "상급자", "하급자", "자문위원", "신입", "주임", "부장", "고객",
            "민원인", "교수", "학생", "직원", "기관", "외부", "내부"
        ))))
        acts = len(set(keyword_hits(text, (
            "요청", "거절", "사과", "감사", "보고", "문의", "검토", "승인",
            "부탁", "확인", "안내", "경고"
        ))))
        score = 0.04 + 0.07 * min(len(hits), 6) + 0.08 * min(roles / 3, 1) + 0.10 * min(acts / 3, 1)
        evidence = []
        if hits:
            evidence.append("화용 표지: " + ", ".join(hits[:10]))
        evidence.append(f"역할 관계 수={roles}")
        evidence.append(f"화행 유형 수={acts}")
        return ComponentScore(clamp(score), 0.80, evidence=evidence)

    def _external_knowledge(self, item: DDIItem, text: str) -> ComponentScore:
        hits = keyword_hits(text, self.EXTERNAL_KNOWLEDGE_TERMS)
        meta_flag = bool(item.metadata.get("requires_external_knowledge"))
        citations = len(re.findall(r"(법|시행령|조례|연도|제\d+조|표준|지침|규정)", text))
        score = 0.03 + (0.35 if meta_flag else 0.0) + 0.09 * min(len(hits), 5) + 0.05 * min(citations / 4, 1)
        evidence = []
        if meta_flag:
            evidence.append("requires_external_knowledge=true")
        if hits:
            evidence.append("외부 지식 표지: " + ", ".join(hits[:8]))
        return ComponentScore(clamp(score), 0.70, evidence=evidence)

    def _openness(self, item: DDIItem) -> ComponentScore:
        mapping = {
            "multiple_choice": 0.12, "mcq": 0.12, "true_false": 0.08,
            "short_answer": 0.35, "constructed_response": 0.72,
            "essay": 0.82, "generation": 0.88, "generative": 0.88, "code": 0.80,
        }
        base = mapping.get(item.response_type, 0.55)
        rubric_len = count_nonspace(item.rubric)
        if item.response_type in {"constructed_response", "essay", "generation", "generative"}:
            base -= 0.12 * min(rubric_len / 300, 1)
        return ComponentScore(clamp(base), 0.92, evidence=[
            f"response_type={item.response_type}", f"루브릭 길이={rubric_len}"
        ])

    def _constraints(self, item: DDIItem) -> ComponentScore:
        hits = regex_hits(item.question, self.CONSTRAINT_PATTERNS)
        numeric = re.findall(r"\d+\s*(?:문장|자|개|가지|단계|쪽|분|초)", item.question)
        negatives = keyword_hits(item.question, ("하지 말", "금지", "제외", "없이", "초과", "미달"))
        score = 0.05 + 0.09 * min(len(hits), 6) + 0.07 * min(len(numeric), 4) + 0.08 * min(len(negatives), 3)
        evidence = []
        if hits:
            evidence.append("제약 표지: " + ", ".join(hits[:10]))
        if numeric:
            evidence.append("수량 제약: " + ", ".join(numeric[:6]))
        if negatives:
            evidence.append("부정 제약: " + ", ".join(negatives[:6]))
        return ComponentScore(clamp(score), 0.86, evidence=evidence)

    def _distractor_plausibility(self, item: DDIItem) -> ComponentScore:
        if item.response_type not in {"multiple_choice", "mcq"}:
            return ComponentScore(0.0, 1.0, applicable=False,
                                  evidence=["비객관식 문항이므로 오답 매력도 제외"])

        options = [x.strip() for x in item.options if x.strip()]
        if len(options) < 3:
            return ComponentScore(0.15, 0.30, warnings=["선택지가 3개 미만"])

        answer_index: Optional[int] = None
        if isinstance(item.answer_key, int):
            answer_index = item.answer_key
        elif isinstance(item.answer_key, str):
            key = item.answer_key.strip()
            labels = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            answer_index = int(key) if key.isdigit() else labels.get(key.upper())
            if answer_index is None:
                for i, option in enumerate(options):
                    if option == key:
                        answer_index = i
                        break

        if answer_index is None or not (0 <= answer_index < len(options)):
            return ComponentScore(0.45, 0.35, warnings=["정답 선택지를 확인할 수 없음"])

        answer = options[answer_index]
        wrong = [x for i, x in enumerate(options) if i != answer_index]
        lexical_sim = safe_mean([jaccard_similarity(answer, x) for x in wrong])
        answer_len = max(1, count_nonspace(answer))
        length_sim = safe_mean([
            1.0 - min(abs(count_nonspace(x) - answer_len) / answer_len, 1.0)
            for x in wrong
        ])
        uniqueness = 1.0 - max(
            (jaccard_similarity(wrong[i], wrong[j])
             for i in range(len(wrong)) for j in range(i + 1, len(wrong))),
            default=0.0,
        )
        score = 0.45 * lexical_sim + 0.35 * length_sim + 0.20 * uniqueness
        return ComponentScore(clamp(score), 0.75, evidence=[
            f"정답-오답 어휘 유사도={lexical_sim:.3f}",
            f"선택지 길이 유사도={length_sim:.3f}",
            f"오답 간 비중복성={uniqueness:.3f}",
        ])

    def _multi_construct(self, item: DDIItem, text: str) -> ComponentScore:
        groups = {
            "grammar": ("문법", "맞춤법", "띄어쓰기", "조사", "어미", "호응"),
            "lexical": ("어휘", "유의어", "관용표현", "신조어", "고어", "방언"),
            "discourse": ("담화", "접속", "화시", "지시", "생략", "무형대용어"),
            "pragmatics": ("화행", "공손", "완곡", "경어", "레지스터", "문체", "책임"),
            "reasoning": ("추론", "근거", "분석", "비교", "평가", "종합"),
            "constraints": ("문장", "글자", "형식", "포함", "제외", "이내"),
        }
        activated = [name for name, terms in groups.items() if any(term in text for term in terms)]
        declared = set(item.subdomains)
        n = max(len(activated), len(declared), 1)
        interactions = keyword_hits(item.question, ("동시에", "함께", "종합하여", "반영하여", "관점에서"))
        score = 0.08 + 0.14 * min((n - 1) / 4, 1) + 0.09 * min(len(interactions), 3)
        return ComponentScore(clamp(score), 0.74, evidence=[
            "활성 구인군: " + (", ".join(activated) if activated else "없음"),
            f"선언된 subdomains 수={len(declared)}",
        ])

    def _clarity(self, item: DDIItem) -> ComponentScore:
        score = 0.78
        evidence, warnings = [], []

        if not item.question.strip():
            score -= 0.50
            warnings.append("발문 누락")
        else:
            evidence.append("발문 존재")

        ambiguous = keyword_hits(item.question, self.AMBIGUOUS_TERMS)
        if ambiguous:
            score -= 0.06 * min(len(ambiguous), 5)
            warnings.append("모호 표현: " + ", ".join(ambiguous[:8]))

        verbs = keyword_hits(item.question, (
            "선택하시오", "쓰시오", "작성하시오", "분석하시오", "설명하시오",
            "수정하시오", "비교하시오", "평가하시오", "제시하시오"
        ))
        if verbs:
            score += 0.08
            evidence.append("과업 동사: " + ", ".join(verbs[:6]))
        else:
            score -= 0.12
            warnings.append("과업 동사가 명확하지 않음")

        numeric = re.findall(r"\d+\s*(?:문장|자|개|가지|단계|쪽)", item.question)
        if numeric:
            score += 0.05
            evidence.append("명시적 수량 조건: " + ", ".join(numeric[:6]))

        if item.response_type in {"multiple_choice", "mcq"}:
            if len(item.options) < 3:
                score -= 0.20
                warnings.append("선택지 수 부족")
            if item.answer_key in (None, ""):
                score -= 0.25
                warnings.append("정답 키 누락")
        else:
            if not item.rubric.strip():
                score -= 0.18
                warnings.append("서술형·생성형 루브릭 누락")
            else:
                score += 0.05 * min(count_nonspace(item.rubric) / 200, 1)
                evidence.append(f"루브릭 길이={count_nonspace(item.rubric)}")

        if item.rationale.strip():
            score += 0.04
            evidence.append("정답 근거 또는 해설 존재")
        else:
            warnings.append("정답 근거 또는 해설 누락")

        contradiction_pairs = (
            ("반드시", "가능하면"), ("정확히", "내외"),
            ("모두", "일부"), ("포함", "제외"),
        )
        contradictions = [f"{a}/{b}" for a, b in contradiction_pairs
                          if a in item.question and b in item.question]
        if contradictions:
            score -= 0.12 * min(len(contradictions), 3)
            warnings.append("상충 가능 지시: " + ", ".join(contradictions))

        return ComponentScore(clamp(score), 0.82, evidence=evidence, warnings=warnings)


class PanelFusion:
    """
    item.metadata["panel_scores"]의 외부 패널 점수와 규칙 점수를 결합한다.

    예:
    "panel_scores": {
      "judge_a": {"R":0.7,"C":0.6,...,"Q":0.9},
      "judge_b": {"R":0.8,"C":0.5,...,"Q":0.8}
    }
    """

    def __init__(self, rule_weight: float, panel_weight: float) -> None:
        self.rule_weight = rule_weight
        self.panel_weight = panel_weight

    def fuse(self, item: DDIItem, rule_scores: Dict[str, ComponentScore]):
        panel_scores = item.metadata.get("panel_scores") or {}
        if not isinstance(panel_scores, Mapping) or not panel_scores:
            return rule_scores, {"panel_used": False, "panel_n": 0, "mean_disagreement": None}

        fused: Dict[str, ComponentScore] = {}
        disagreements: List[float] = []

        for component in COMPONENTS:
            rule = rule_scores[component]
            values = []
            for judge_scores in panel_scores.values():
                if isinstance(judge_scores, Mapping) and component in judge_scores:
                    try:
                        values.append(clamp(float(judge_scores[component])))
                    except (TypeError, ValueError):
                        pass

            if not values:
                fused[component] = rule
                continue

            panel_mean = safe_mean(values)
            disagreement = safe_stdev(values)
            disagreements.append(disagreement)
            value = self.rule_weight * rule.value + self.panel_weight * panel_mean
            confidence = (
                0.45 * rule.confidence
                + 0.35 * min(len(values) / 3, 1)
                + 0.20 * (1 - min(disagreement / 0.25, 1))
            )
            fused[component] = ComponentScore(
                value=clamp(value),
                confidence=clamp(confidence),
                applicable=rule.applicable,
                evidence=rule.evidence + [
                    f"외부 패널 평균={panel_mean:.3f}",
                    f"외부 패널 표준편차={disagreement:.3f}",
                ],
                warnings=rule.warnings,
            )

        return fused, {
            "panel_used": True,
            "panel_n": len(panel_scores),
            "mean_disagreement": safe_mean(disagreements) if disagreements else None,
        }


@dataclass
class DDIResult:
    item_id: str
    domain: str
    components: Dict[str, Dict[str, Any]]
    ddi_global: float
    ddi_domain: float
    ddi_auto: float
    question_clarity: float
    ddi_clean: float
    ddi_calibrated: float
    calibration_source: str
    preliminary_level: str
    confidence: float
    warnings: List[str]
    panel_info: Dict[str, Any]
    formula_version: str = "DDI-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DDIEngine:
    def __init__(self, config: Optional[DDIConfig] = None,
                 calibrator: Optional[AnchorCalibrator] = None) -> None:
        self.config = config or DDIConfig()
        self.config.validate()
        self.analyzer = KoreanRuleAnalyzer()
        self.fusion = PanelFusion(self.config.rule_weight, self.config.panel_weight)
        self.calibrator = calibrator

    @staticmethod
    def _weighted_score(components: Mapping[str, ComponentScore],
                        weights: Mapping[str, float]) -> float:
        active = [(w, components[name].value)
                  for name, w in weights.items() if components[name].applicable]
        if not active:
            return 0.0
        total_weight = sum(w for w, _ in active)
        return 100 * sum(w * value for w, value in active) / total_weight

    def _domain_weights(self, domain: str) -> Dict[str, float]:
        return self.config.domain_weights.get((domain or "general").lower(),
                                              self.config.global_weights)

    def _level(self, score: float) -> str:
        if score < self.config.l1_upper:
            return "L1"
        if score < self.config.l2_upper:
            return "L2"
        return "L3"

    def score(self, item: DDIItem) -> DDIResult:
        rule_scores = self.analyzer.analyze(item)
        components, panel_info = self.fusion.fuse(item, rule_scores)

        ddi_global = self._weighted_score(components, self.config.global_weights)
        ddi_domain = self._weighted_score(components, self._domain_weights(item.domain))
        ddi_auto = (self.config.lambda_global * ddi_global
                    + (1 - self.config.lambda_global) * ddi_domain)

        q = components["Q"].value
        clarity_factor = self.config.clarity_floor + (1 - self.config.clarity_floor) * q
        ddi_clean = ddi_auto * clarity_factor

        if self.calibrator:
            ddi_calibrated, source = self.calibrator.predict(ddi_clean, item.domain)
        else:
            ddi_calibrated, source = ddi_clean, "none"

        active_conf = [c.confidence for c in components.values() if c.applicable]
        coverage = len(active_conf) / len(COMPONENTS)
        confidence = clamp(0.65 * safe_mean(active_conf) + 0.35 * coverage)

        warnings = []
        for name, comp in components.items():
            warnings.extend([f"{name}: {w}" for w in comp.warnings])
        if q < 0.60:
            warnings.append("Q_i가 낮으므로 고난도 해석보다 문항 수정 검토가 우선됨")
        if confidence < 0.60:
            warnings.append("자동 산출 신뢰도가 낮아 인간 감사 또는 패널 평정 권장")

        payload = {
            name: {
                "value": round(comp.value, 4),
                "confidence": round(comp.confidence, 4),
                "applicable": comp.applicable,
                "evidence": comp.evidence,
                "warnings": comp.warnings,
            }
            for name, comp in components.items()
        }

        return DDIResult(
            item_id=item.item_id,
            domain=item.domain,
            components=payload,
            ddi_global=round(ddi_global, 4),
            ddi_domain=round(ddi_domain, 4),
            ddi_auto=round(ddi_auto, 4),
            question_clarity=round(q, 4),
            ddi_clean=round(ddi_clean, 4),
            ddi_calibrated=round(ddi_calibrated, 4),
            calibration_source=source,
            preliminary_level=self._level(ddi_calibrated),
            confidence=round(confidence, 4),
            warnings=warnings,
            panel_info=panel_info,
        )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} {line_no}행 JSON 오류: {exc}") from exc
    return rows


def save_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "item_id": result["item_id"],
        "domain": result["domain"],
        "ddi_global": result["ddi_global"],
        "ddi_domain": result["ddi_domain"],
        "ddi_auto": result["ddi_auto"],
        "question_clarity": result["question_clarity"],
        "ddi_clean": result["ddi_clean"],
        "ddi_calibrated": result["ddi_calibrated"],
        "calibration_source": result["calibration_source"],
        "preliminary_level": result["preliminary_level"],
        "confidence": result["confidence"],
        "warnings": " | ".join(result.get("warnings") or []),
    }
    for name in COMPONENTS:
        comp = result["components"][name]
        row[name] = comp["value"]
        row[f"{name}_confidence"] = comp["confidence"]
    return row


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = [flatten_result(r) for r in rows]
    if not flattened:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


def demo_item() -> DDIItem:
    return DDIItem(
        item_id="DEMO-L3-EMAIL-001",
        domain="business_email",
        subdomains=["honorifics", "pragmatics", "register", "responsibility"],
        response_type="constructed_response",
        passage=(
            "홍길동 자문위원님, 안녕하세요.\n"
            "지방연구원 연구기획팀 주임 연구원 김철수입니다.\n"
            "지난번에 말씀하신 부분은 거의 고쳤고요, 예산표는 아직 저희가 못 정해서 빼놨습니다.\n"
            "검토 의견 주신 내용 중 연구 목표와 추진 체계 부분은 자료에 반영했습니다.\n"
            "다만 예산 산출 근거는 내부 결재가 아직 끝나지 않아서 이번 파일에는 넣지 않았습니다.\n"
            "내일 오전에 중간보고 자료를 제출해야 해서 시간이 조금 촉박합니다.\n"
            "확인하시고 틀린 거 있으면 오늘 안으로 알려주세요.\n"
            "특히 3쪽 연구 범위와 7쪽 추진 일정 부분을 봐주시면 됩니다.\n"
            "바쁘신 건 아는데, 저희도 일정이 촉박해서 빨리 봐주셔야 합니다.\n"
            "확인 부탁드립니다.\n김철수 드림"
        ),
        question=(
            "다음 2개 문항에 모두 답하시오.\n"
            "1. 위 메일 초안을 과잉 사과나 책임 회피 없이 상황에 맞는 공문형 업무 메일 전문으로 수정하시오.\n"
            "2. 1번 문항의 수정안처럼 고친 이유를 5문장으로 설명하시오. "
            "단, 각 문장은 30자 내외로 작성하고, 2번 답안의 총 분량은 150자 이내로 작성하시오."
        ),
        rubric=(
            "RB1 언어 형식 정확성: 격식체의 일관성, 경어 사용의 적절성, 문법성.\n"
            "RB2 상황 분석 및 논증 타당성: 역할 관계 파악, 문제 진단, 수정 근거.\n"
            "RB3 업무 담화 수행 적절성: 검토 요청의 완곡성, 부담 완화, 책임 귀속.\n"
            "RB4 과업 충족도 및 제약 준수: 문항 1과 2 모두 응답, 5문장, 150자 이내."
        ),
        rationale=(
            "원문은 외부 자문위원에게 구어적이고 압박적인 표현을 사용하며, "
            "내부 일정 책임을 상대에게 전가하는 인상을 줄 수 있다."
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="한국어 평가 문항 DDI 자동 산출 엔진")
    parser.add_argument("--input", type=Path, help="입력 JSONL 파일")
    parser.add_argument("--output", type=Path, help="출력 JSONL 또는 CSV 파일")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--anchors", type=Path, help="앵커 보정 JSONL 파일")
    parser.add_argument("--demo", action="store_true", help="내장 데모 문항 실행")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    calibrator = None
    if args.anchors:
        calibrator = AnchorCalibrator()
        calibrator.fit(load_jsonl(args.anchors))

    engine = DDIEngine(calibrator=calibrator)

    if args.demo:
        print(json.dumps(engine.score(demo_item()).to_dict(), ensure_ascii=False, indent=2))
        return 0

    if not args.input or not args.output:
        print("--input과 --output이 필요합니다. 또는 --demo를 사용하세요.", file=sys.stderr)
        return 2

    results = []
    for row in load_jsonl(args.input):
        try:
            results.append(engine.score(DDIItem.from_dict(row)).to_dict())
        except Exception as exc:
            item_id = row.get("item_id") or row.get("id") or "UNKNOWN"
            results.append({"item_id": str(item_id), "error": f"{type(exc).__name__}: {exc}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        save_csv(args.output, [r for r in results if "error" not in r])
    else:
        save_jsonl(args.output, results)

    print(f"완료: {len(results)}개 문항 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
