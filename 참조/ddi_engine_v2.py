# -*- coding: utf-8 -*-
"""
DDI 통합 산출 엔진 v2.0
========================

규칙 기반 분석기(65%)와 Qwen2.5 LLM 패널(35%)을 결합한 DDI 산출 엔진.

산식:
  z_fused   = 0.65 * z_rule + 0.35 * z_qwen
  DDI_global = 100 * Σ(w_p * z_fused)
  DDI_domain = 100 * Σ(w_d,p * z_fused)
  DDI_auto   = 0.5 * DDI_global + 0.5 * DDI_domain
  DDI_clean  = DDI_auto * (0.5 + 0.5 * Q_i)

실행:
  pip install fastapi uvicorn openai instructor pydantic

  # FastAPI 서버 실행
  uvicorn ddi_engine_v2:app --reload

  # CLI 데모
  python ddi_engine_v2.py --demo

  # 배치 처리
  python ddi_engine_v2.py --input items.jsonl --output results.jsonl
  python ddi_engine_v2.py --input items.jsonl --output results.csv --format csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import instructor
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

MODEL_NAME = "qwen2.5:7b-instruct-q8_0"
COMPONENTS = ("R", "C", "K", "P", "E", "O", "X", "Ddist", "M", "Q")


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 응답 유형 열거형
# ─────────────────────────────────────────────

class ResponseType(str, Enum):
    multiple_choice = "multiple_choice"
    mcq = "mcq"
    true_false = "true_false"
    short_answer = "short_answer"
    constructed_response = "constructed_response"
    essay = "essay"
    generation = "generation"
    generative = "generative"

    @classmethod
    def is_mcq(cls, value: str) -> bool:
        return value.strip().lower() in {cls.multiple_choice, cls.mcq}


# ─────────────────────────────────────────────
# 루브릭 항목
# ─────────────────────────────────────────────

@dataclass
class RubricItem:
    """채점 루브릭 단일 항목."""
    id: str        # "RB1"
    name: str      # "언어 형식 정확성"
    weight: float  # 0.25
    criteria: str  # "격식체 일관성, 경어 사용의 적절성"

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"RubricItem '{self.id}' weight는 0.0~1.0이어야 합니다.")

    def to_text(self) -> str:
        return f"{self.id} {self.name}(가중치 {self.weight}): {self.criteria}"


def validate_rubric_weights(rubric: List[RubricItem]) -> None:
    """루브릭 가중치 합이 1.0인지 검증한다."""
    if not rubric:
        return
    total = sum(r.weight for r in rubric)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"루브릭 가중치 합이 1.0이 아닙니다: {total:.4f}")


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class DDIItem:
    item_id: str
    passage: str = ""
    question: str = ""
    options: List[str] = field(default_factory=list)
    answer_key: Any = None
    rationale: str = ""
    rubric: List[RubricItem] = field(default_factory=list)
    response_type: str = "constructed_response"
    domain: str = "general"
    subdomains: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id는 필수입니다.")
        if not self.question.strip():
            raise ValueError(f"[{self.item_id}] question(발문)은 필수입니다.")
        validate_rubric_weights(self.rubric)

    @property
    def rubric_text(self) -> str:
        """루브릭을 LLM 프롬프트용 문자열로 변환."""
        if not self.rubric:
            return "없음"
        return "\n".join(r.to_text() for r in self.rubric)

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

        # 루브릭 파싱
        raw_rubric = data.get("rubric") or []
        rubric: List[RubricItem] = []
        if isinstance(raw_rubric, str):
            # 하위 호환: 문자열로 들어오면 단일 항목으로 처리
            if raw_rubric.strip():
                rubric = [RubricItem(id="RB1", name="채점 기준", weight=1.0, criteria=raw_rubric)]
        elif isinstance(raw_rubric, list):
            for r in raw_rubric:
                if isinstance(r, Mapping):
                    rubric.append(RubricItem(
                        id=str(r.get("id", "")),
                        name=str(r.get("name", "")),
                        weight=float(r.get("weight", 0.0)),
                        criteria=str(r.get("criteria", "")),
                    ))

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
            rubric=rubric,
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
class CalibrationModel:
    slope: float = 1.0
    intercept: float = 0.0
    n: int = 0

    def predict(self, x: float) -> float:
        return max(0.0, min(100.0, self.slope * x + self.intercept))


# ─────────────────────────────────────────────
# DDI 설정 (가중치)
# ─────────────────────────────────────────────

@dataclass
class DDIConfig:
    # 글로벌 가중치 (Q 제외, 합=1.0)
    global_weights: Dict[str, float] = field(default_factory=lambda: {
        "R": 0.18, "C": 0.12, "K": 0.13, "P": 0.13, "E": 0.08,
        "O": 0.08, "X": 0.11, "Ddist": 0.07, "M": 0.10,
    })
    # 도메인별 가중치
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
    lambda_global: float = 0.50      # DDI_auto = λ*global + (1-λ)*domain
    rule_weight: float = 0.65        # 규칙 엔진 비중
    panel_weight: float = 0.35       # Qwen 패널 비중
    clarity_floor: float = 0.50      # Q 보정 하한
    l1_upper: float = 35.0
    l2_upper: float = 70.0
    min_anchor_count_for_domain_calibration: int = 5

    def validate(self) -> None:
        if not math.isclose(sum(self.global_weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("global_weights 합이 1.0이 아닙니다.")
        for domain, weights in self.domain_weights.items():
            if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
                raise ValueError(f"{domain} domain_weights 합이 1.0이 아닙니다.")
        if not math.isclose(self.rule_weight + self.panel_weight, 1.0, abs_tol=1e-6):
            raise ValueError("rule_weight + panel_weight 합이 1.0이 아닙니다.")


# ─────────────────────────────────────────────
# 앵커 보정기
# ─────────────────────────────────────────────

class AnchorCalibrator:
    """
    전문가 앵커 DDI로 자동 DDI를 선형 보정한다.

    앵커 JSONL 형식:
    {"item_id":"A001","domain":"honorifics","ddi_clean":61.2,"expert_ddi":67.0}
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
        return CalibrationModel(slope=slope, intercept=my - slope * mx, n=len(xs))

    def fit(self, anchors: Sequence[Mapping[str, Any]]) -> None:
        global_x, global_y = [], []
        by_domain: Dict[str, Tuple[List[float], List[float]]] = {}
        for row in anchors:
            try:
                x, y = float(row["ddi_clean"]), float(row["expert_ddi"])
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


# ─────────────────────────────────────────────
# 규칙 기반 분석기 (65% 비중)
# ─────────────────────────────────────────────

class KoreanRuleAnalyzer:
    """텍스트 패턴·키워드 기반으로 10개 요인을 1차 추정한다."""

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
            part for part in (item.passage, item.question, item.rationale, item.rubric_text) if part
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
            "honorifics", "orthography", "dialect", "archaic",
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
        evidence.append(f"역할 관계 수={roles}, 화행 유형 수={acts}")
        return ComponentScore(clamp(score), 0.80, evidence=evidence)

    def _external_knowledge(self, item: DDIItem, text: str) -> ComponentScore:
        hits = keyword_hits(text, self.EXTERNAL_KNOWLEDGE_TERMS)
        meta_flag = bool(item.metadata.get("requires_external_knowledge"))
        citations = len(re.findall(r"(법|시행령|조례|연도|제\d+조|표준|지침|규정)", text))
        score = (
            0.03
            + (0.35 if meta_flag else 0.0)
            + 0.09 * min(len(hits), 5)
            + 0.05 * min(citations / 4, 1)
        )
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
            "essay": 0.82, "generation": 0.88, "generative": 0.88,
        }
        base = mapping.get(item.response_type, 0.55)
        rubric_len = count_nonspace(item.rubric_text)
        if item.response_type in {"constructed_response", "essay", "generation", "generative"}:
            base -= 0.12 * min(rubric_len / 300, 1)
        return ComponentScore(clamp(base), 0.92, evidence=[
            f"response_type={item.response_type}", f"루브릭 항목 수={len(item.rubric)}"
        ])

    def _constraints(self, item: DDIItem) -> ComponentScore:
        hits = regex_hits(item.question, self.CONSTRAINT_PATTERNS)
        numeric = re.findall(r"\d+\s*(?:문장|자|개|가지|단계|쪽|분|초)", item.question)
        negatives = keyword_hits(item.question, ("하지 말", "금지", "제외", "없이", "초과", "미달"))
        score = (
            0.05
            + 0.09 * min(len(hits), 6)
            + 0.07 * min(len(numeric), 4)
            + 0.08 * min(len(negatives), 3)
        )
        evidence = []
        if hits:
            evidence.append("제약 표지: " + ", ".join(hits[:10]))
        if numeric:
            evidence.append("수량 제약: " + ", ".join(numeric[:6]))
        return ComponentScore(clamp(score), 0.86, evidence=evidence)

    def _distractor_plausibility(self, item: DDIItem) -> ComponentScore:
        if not ResponseType.is_mcq(item.response_type):
            return ComponentScore(0.0, 1.0, applicable=False,
                                  evidence=["비객관식 문항 → 오답 매력도 제외"])

        options = [x.strip() for x in item.options if x.strip()]
        if len(options) < 3:
            return ComponentScore(0.15, 0.30, warnings=["선택지가 3개 미만"])

        answer_index: Optional[int] = None
        if isinstance(item.answer_key, int):
            answer_index = item.answer_key
        elif isinstance(item.answer_key, str):
            key = item.answer_key.strip()
            labels = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            if key.isdigit():
                answer_index = int(key)
            elif key.upper() in labels:
                answer_index = labels[key.upper()]
            else:
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

        if ResponseType.is_mcq(item.response_type):
            if len(item.options) < 3:
                score -= 0.20
                warnings.append("선택지 수 부족")
            if item.answer_key in (None, ""):
                score -= 0.25
                warnings.append("정답 키 누락")
        else:
            if not item.rubric:
                score -= 0.18
                warnings.append("서술형·생성형 루브릭 누락")
            else:
                rubric_text = item.rubric_text
                score += 0.05 * min(count_nonspace(rubric_text) / 200, 1)
                evidence.append(f"루브릭 항목 수={len(item.rubric)}")
                # 가중치 합 검증
                total_w = sum(r.weight for r in item.rubric)
                if not math.isclose(total_w, 1.0, abs_tol=1e-6):
                    warnings.append(f"루브릭 가중치 합={total_w:.2f} (1.0이어야 함)")

        if item.rationale.strip():
            score += 0.04
            evidence.append("정답 근거 또는 해설 존재")
        else:
            warnings.append("정답 근거 또는 해설 누락")

        contradiction_pairs = (
            ("반드시", "가능하면"), ("정확히", "내외"),
            ("모두", "일부"), ("포함", "제외"),
        )
        contradictions = [
            f"{a}/{b}" for a, b in contradiction_pairs
            if a in item.question and b in item.question
        ]
        if contradictions:
            score -= 0.12 * min(len(contradictions), 3)
            warnings.append("상충 가능 지시: " + ", ".join(contradictions))

        return ComponentScore(clamp(score), 0.82, evidence=evidence, warnings=warnings)


# ─────────────────────────────────────────────
# Qwen LLM 패널 (35% 비중)
# ─────────────────────────────────────────────

class QwenComponentProfile(BaseModel):
    """Qwen이 반환하는 10개 요인 구조화 출력."""
    R_value: float = Field(..., ge=0.0, le=1.0, description="추론 복잡도 (0~1)")
    R_rationale: str = Field(..., description="판정 근거 (한국어 1~2문장)")
    C_value: float = Field(..., ge=0.0, le=1.0, description="맥락 의존도")
    C_rationale: str = Field(...)
    K_value: float = Field(..., ge=0.0, le=1.0, description="한국어 특성 부하량")
    K_rationale: str = Field(...)
    P_value: float = Field(..., ge=0.0, le=1.0, description="화용·공손 부하량")
    P_rationale: str = Field(...)
    E_value: float = Field(..., ge=0.0, le=1.0, description="외부 지식 의존도")
    E_rationale: str = Field(...)
    O_value: float = Field(..., ge=0.0, le=1.0, description="응답 개방성")
    O_rationale: str = Field(...)
    X_value: float = Field(..., ge=0.0, le=1.0, description="제약 조건 복잡도")
    X_rationale: str = Field(...)
    Ddist_value: float = Field(..., ge=0.0, le=1.0, description="오답 매력도")
    Ddist_rationale: str = Field(...)
    M_value: float = Field(..., ge=0.0, le=1.0, description="다중 구인 결합도")
    M_rationale: str = Field(...)
    Q_value: float = Field(..., ge=0.0, le=1.0, description="문항 명료도")
    Q_rationale: str = Field(...)


SYSTEM_PROMPT = """당신은 한국어 AI 평가 문항의 난이도를 측정하는 전문 평정자입니다.
인간 출제자가 제공한 문항을 정밀 심사하여 10개 요인별 점수(0.0~1.0)와 판정 근거를 도출하십시오.

[절대 준수 지침]
1. 모든 rationale은 반드시 한국어로만 작성하십시오.
2. 고난도 문항과 불량 문항을 반드시 구분하십시오.
   - 좋은 고난도 문항: 정답 근거가 명확하고 구인 복잡도가 높으며 전문가 합의가 가능함
   - 불량 문항: 정답이 복수 가능하거나 발문이 애매하거나 루브릭으로 채점하기 어려움
   - 불량 문항으로 판단되면 Q 점수를 0.3 이하로 부여하십시오.
3. 모델이 틀렸다는 사실만으로 고난도라고 해석하지 마십시오.
4. 각 요인의 rationale은 해당 요인 기준에 따라 2~3문장으로 작성하십시오.
5. 점수 근거를 구체적 문항 내용과 반드시 연결하여 설명하십시오.
6. "복잡함", "어려움" 같은 추상적 표현 대신 구체적 근거를 쓰십시오.

[요인별 평정 기준]

R (Reasoning · 추론 단계 수)
- 0.0~0.2: 단순 사실 확인, 1단계 이내 판단
- 0.3~0.5: 2~3단계 추론, 조건 간 비교 필요
- 0.6~0.8: 4~5단계 추론, 복수 정보 통합 필요
- 0.9~1.0: 6단계 이상, 고차원 추론·메타인지 필요
근거 작성 시 실제 추론 단계를 순서대로 나열하십시오.

C (Context · 맥락 의존도)
- 0.0~0.2: 지문 없이 독립적으로 풀 수 있음
- 0.3~0.5: 지문 일부 참조 필요
- 0.6~0.8: 담화 상황·역할 관계 전체 파악 필요
- 0.9~1.0: 지문 외 암묵적 맥락·화시 이해 필수
근거 작성 시 어떤 맥락 요소가 핵심인지 명시하십시오.

K (Korean-specificity · 한국어 특성 부하량)
- 0.0~0.2: 한국어 고유 지식 거의 불필요
- 0.3~0.5: 기본 경어법·맞춤법·형태소 지식 필요
- 0.6~0.8: 경어법 체계·문장성분 호응·문체 구분 필요
- 0.9~1.0: 고어·방언·신조어·복합 형태소 통합 필요
근거 작성 시 구체적인 한국어 언어 현상을 지목하십시오.

P (Pragmatics · 화용·사회언어학 부하량)
- 0.0~0.2: 화용적 판단 불필요
- 0.3~0.5: 기본 공손성·발화 의도 파악 필요
- 0.6~0.8: 완곡 표현·간접 화행·역할 관계 고려 필요
- 0.9~1.0: 복합적 사회언어학적 상황 판단 필수
근거 작성 시 어떤 화행·공손 전략이 요구되는지 명시하십시오.

E (External knowledge · 외부 지식·문화 맥락)
- 0.0~0.2: 외부 지식 불필요
- 0.3~0.5: 일반 상식 수준의 배경지식 필요
- 0.6~0.8: 특정 도메인 전문지식 필요 (공문서, 업무 문화 등)
- 0.9~1.0: 고도의 전문지식·문화적 맥락 필수
근거 작성 시 요구되는 외부 지식의 종류를 명시하십시오.

O (Openness · 응답 개방성)
- 0.0~0.2: 객관식·단답형, 정답이 하나로 수렴
- 0.3~0.5: 제한적 서술형, 허용 답안 범위가 좁음
- 0.6~0.8: 서술형, 다양한 표현 허용
- 0.9~1.0: 완전 자유 서술, 정답 수렴 기준 없음
근거 작성 시 허용 답안의 범위를 설명하십시오.

X (Constraint complexity · 제약 조건 복잡도)
- 0.0~0.2: 제약 조건 없음
- 0.3~0.5: 단순 제약 1~2개 (예: 글자 수 제한)
- 0.6~0.8: 복합 제약 3~4개 동시 준수 필요
- 0.9~1.0: 5개 이상 제약 조건 동시 충족 필요
근거 작성 시 제약 조건을 항목별로 나열하십시오.

Ddist (Distractor plausibility · 오답 매력도)
- 0.0~0.2: 오답이 명백하여 혼동 가능성 없음
- 0.3~0.5: 일부 오답이 그럴듯하여 혼동 가능
- 0.6~0.8: 오답 대부분이 매력적, 신중한 판단 필요
- 0.9~1.0: 정답과 오답 구분이 매우 어려움
서술형 문항의 경우 감점 유발 표현의 매력도를 평가하십시오.
근거 작성 시 정답과 오답의 의미적 거리, 오답 간 유사성을 명시하십시오.

M (Multi-construct integration · 다중 구인 결합도)
- 0.0~0.2: 단일 구인만 측정
- 0.3~0.5: 2개 구인 결합
- 0.6~0.8: 3~4개 구인 유기적 결합
- 0.9~1.0: 5개 이상 구인 동시 통합 필요
근거 작성 시 결합된 구인을 모두 나열하십시오.

Q (Question clarity · 문항 명료도) ← 안전장치
- 0.9~1.0: 정답 근거 명확, 루브릭 채점 용이, 전문가 합의 가능
- 0.7~0.8: 대체로 명확하나 일부 해석 여지 있음
- 0.5~0.6: 발문 또는 채점 기준이 다소 모호함
- 0.3~0.4: 정답이 복수 가능하거나 루브릭이 불충분함
- 0.0~0.2: 출제 오류 수준, 즉시 수정·보류 필요
반드시 정답 유일성과 루브릭 채점 가능성을 중심으로 판단하십시오."""


def _build_user_prompt(item: DDIItem) -> str:
    is_mcq = ResponseType.is_mcq(item.response_type)
    options_text = ""
    if is_mcq and item.options:
        options_text = "\n".join(
            f"  {i + 1}. {opt}" for i, opt in enumerate(item.options)
        )
        options_text = f"\n- 선택지:\n{options_text}"
        options_text += f"\n- 정답 키: {item.answer_key}"

    return f"""[문항 정보]
- 문항 ID: {item.item_id}
- 도메인: {item.domain} / 서브도메인: {', '.join(item.subdomains) or '없음'}
- 문항 유형: {item.response_type}
- 지문: {item.passage or '없음'}
- 발문: {item.question}{options_text}
- 채점 기준(루브릭): {item.rubric_text}
- 정답 해설: {item.rationale or '없음'}"""


class QwenPanel:
    """Qwen LLM을 패널 평정자로 사용한다."""

    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "local-token",
                 timeout: float = 120.0) -> None:
        self._client = instructor.from_openai(
            AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
            mode=instructor.Mode.JSON,
        )

    async def evaluate(self, item: DDIItem) -> Optional[Dict[str, float]]:
        """
        Qwen으로 요인별 점수를 추정한다.
        실패 시 None 반환 → 규칙 점수만 사용하는 폴백으로 처리.
        """
        try:
            profile: QwenComponentProfile = await self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(item)},
                ],
                response_model=QwenComponentProfile,
                temperature=0.0,
                seed=42,
                max_retries=1,
            )
            return {
                "R": profile.R_value, "R_rationale": profile.R_rationale,
                "C": profile.C_value, "C_rationale": profile.C_rationale,
                "K": profile.K_value, "K_rationale": profile.K_rationale,
                "P": profile.P_value, "P_rationale": profile.P_rationale,
                "E": profile.E_value, "E_rationale": profile.E_rationale,
                "O": profile.O_value, "O_rationale": profile.O_rationale,
                "X": profile.X_value, "X_rationale": profile.X_rationale,
                "Ddist": profile.Ddist_value, "Ddist_rationale": profile.Ddist_rationale,
                "M": profile.M_value, "M_rationale": profile.M_rationale,
                "Q": profile.Q_value, "Q_rationale": profile.Q_rationale,
            }
        except Exception as exc:
            logger.warning("[%s] Qwen 평정 실패 → 규칙 점수만 사용: %s", item.item_id, exc)
            return None


# ─────────────────────────────────────────────
# 패널 퓨전 (규칙 65% + Qwen 35%)
# ─────────────────────────────────────────────

class PanelFusion:
    def __init__(self, rule_weight: float, panel_weight: float) -> None:
        self.rule_weight = rule_weight
        self.panel_weight = panel_weight

    def fuse(
        self,
        rule_scores: Dict[str, ComponentScore],
        qwen_scores: Optional[Dict[str, float]],
    ) -> Tuple[Dict[str, ComponentScore], Dict[str, Any]]:
        """
        규칙 점수와 Qwen 점수를 결합한다.
        qwen_scores가 None이면 규칙 점수만 사용(폴백).
        """
        if not qwen_scores:
            return rule_scores, {"panel_used": False, "fallback": True}

        fused: Dict[str, ComponentScore] = {}
        for comp in COMPONENTS:
            rule = rule_scores[comp]
            qwen_val = qwen_scores.get(comp)

            if qwen_val is None:
                fused[comp] = rule
                continue

            qwen_val = clamp(float(qwen_val))
            value = self.rule_weight * rule.value + self.panel_weight * qwen_val
            confidence = clamp(
                0.55 * rule.confidence
                + 0.30 * 1.0   # Qwen 단일 판정자
                + 0.15 * (1 - abs(rule.value - qwen_val))  # 일치도 보너스
            )
            rationale_key = f"{comp}_rationale"
            evidence = rule.evidence + (
                [f"Qwen 평정={qwen_val:.3f} | {qwen_scores.get(rationale_key, '')}"]
                if rationale_key in qwen_scores else [f"Qwen 평정={qwen_val:.3f}"]
            )
            fused[comp] = ComponentScore(
                value=clamp(value),
                confidence=clamp(confidence),
                applicable=rule.applicable,
                evidence=evidence,
                warnings=rule.warnings,
            )

        return fused, {"panel_used": True, "fallback": False}


# ─────────────────────────────────────────────
# DDI 결과
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# DDI 엔진 (통합)
# ─────────────────────────────────────────────

class DDIEngine:
    def __init__(
        self,
        config: Optional[DDIConfig] = None,
        calibrator: Optional[AnchorCalibrator] = None,
        qwen_panel: Optional[QwenPanel] = None,
    ) -> None:
        self.config = config or DDIConfig()
        self.config.validate()
        self.analyzer = KoreanRuleAnalyzer()
        self.fusion = PanelFusion(self.config.rule_weight, self.config.panel_weight)
        self.calibrator = calibrator
        self.qwen = qwen_panel  # None이면 규칙 전용 모드

    @staticmethod
    def _weighted_score(
        components: Mapping[str, ComponentScore],
        weights: Mapping[str, float],
    ) -> float:
        active = [
            (w, components[name].value)
            for name, w in weights.items()
            if components[name].applicable
        ]
        if not active:
            return 0.0
        total_weight = sum(w for w, _ in active)
        return 100 * sum(w * v for w, v in active) / total_weight

    def _domain_weights(self, domain: str) -> Dict[str, float]:
        return self.config.domain_weights.get(
            (domain or "general").lower(),
            self.config.global_weights,
        )

    def _level(self, score: float) -> str:
        if score < self.config.l1_upper:
            return "L1"
        if score < self.config.l2_upper:
            return "L2"
        return "L3"

    async def score(self, item: DDIItem) -> DDIResult:
        # 1. 규칙 기반 분석
        rule_scores = self.analyzer.analyze(item)

        # 2. Qwen 패널 평정 (설정된 경우)
        qwen_scores = await self.qwen.evaluate(item) if self.qwen else None

        # 3. 규칙 65% + Qwen 35% 결합
        components, panel_info = self.fusion.fuse(rule_scores, qwen_scores)

        # 4. 글로벌 + 도메인 가중합
        ddi_global = self._weighted_score(components, self.config.global_weights)
        ddi_domain = self._weighted_score(components, self._domain_weights(item.domain))
        ddi_auto = (
            self.config.lambda_global * ddi_global
            + (1 - self.config.lambda_global) * ddi_domain
        )

        # 5. Q 보정
        q = components["Q"].value
        clarity_factor = self.config.clarity_floor + (1 - self.config.clarity_floor) * q
        ddi_clean = ddi_auto * clarity_factor

        # 6. 앵커 보정 (있는 경우)
        if self.calibrator:
            ddi_calibrated, cal_source = self.calibrator.predict(ddi_clean, item.domain)
        else:
            ddi_calibrated, cal_source = ddi_clean, "none"

        # 7. 신뢰도·경고
        active_conf = [c.confidence for c in components.values() if c.applicable]
        coverage = len(active_conf) / len(COMPONENTS)
        confidence = clamp(0.65 * safe_mean(active_conf) + 0.35 * coverage)

        warnings: List[str] = []
        for name, comp in components.items():
            warnings.extend([f"{name}: {w}" for w in comp.warnings])
        if q < 0.60:
            warnings.append("Q가 낮음 → 고난도 해석보다 문항 수정 검토 우선")
        if confidence < 0.60:
            warnings.append("자동 산출 신뢰도 낮음 → 인간 감사 권장")
        if panel_info.get("fallback"):
            warnings.append("Qwen 평정 실패 → 규칙 점수만 사용됨")

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
            calibration_source=cal_source,
            preliminary_level=self._level(ddi_calibrated),
            confidence=round(confidence, 4),
            warnings=warnings,
            panel_info=panel_info,
        )


# ─────────────────────────────────────────────
# FastAPI 스키마
# ─────────────────────────────────────────────

class RubricItemPayload(BaseModel):
    """채점 루브릭 단일 항목 (API 입력용)"""
    id: str = Field(..., description="루브릭 항목 식별자. 예: RB1", examples=["RB1"])
    name: str = Field(..., description="루브릭 항목 이름. 예: 언어 형식 정확성", examples=["언어 형식 정확성"])
    weight: float = Field(..., ge=0.0, le=1.0, description="항목 가중치. 전체 합은 1.0이어야 함.", examples=[0.25])
    criteria: str = Field(..., description="채점 기준 설명.", examples=["격식체 일관성, 경어 사용의 적절성"])


class ItemPayload(BaseModel):
    """DDI 산출 요청 문항 데이터"""
    model_config = ConfigDict(populate_by_name=True)

    item_id: str = Field(
        ...,
        alias="id",
        description="문항 고유 식별자. 예: KOR-L3-EMAIL-001",
        examples=["KOR-L3-EMAIL-001"],
    )
    domain: str = Field(
        default="general",
        description=(
            "문항 도메인. 도메인별 가중치 적용에 사용됨. "
            "지원값: honorifics(경어법), pragmatics(화용), discourse(담화), "
            "orthography(맞춤법), dialect(방언), archaic(고어), "
            "inference(추론), business_email(업무이메일), general(기타)"
        ),
        examples=["business_email"],
    )
    subdomains: List[str] = Field(
        default_factory=list,
        description="세부 도메인 태그 목록. 다중 구인(M) 산출에 활용됨.",
        examples=[["honorifics", "pragmatics", "register"]],
    )
    response_type: str = Field(
        default="constructed_response",
        description=(
            "문항 유형. "
            "multiple_choice(객관식), short_answer(단답형), "
            "constructed_response(서술형), essay(논술형), generation(생성형). "
            "객관식은 options와 answer_key 필수."
        ),
        examples=["constructed_response"],
    )
    passage: str = Field(
        default="",
        description="평가 지문. 지문이 없는 문항은 빈 문자열로 전달.",
        examples=["홍길동 자문위원님, 안녕하세요..."],
    )
    question: str = Field(
        ...,
        description="발문(필수). 수행 조건이 명확한 지시형 문체여야 함.",
        examples=["위 메일 초안을 공문형 업무 메일로 수정하시오."],
    )
    rubric: List["RubricItemPayload"] = Field(
        default_factory=list,
        description=(
            "채점 루브릭 항목 목록. 서술형·생성형 문항은 필수. "
            "누락 시 Q(문항 명료도) 점수가 낮아짐. "
            "모든 항목의 weight 합은 반드시 1.0이어야 함."
        ),
        examples=[[
            {"id": "RB1", "name": "언어 형식 정확성", "weight": 0.25, "criteria": "격식체 일관성, 경어 사용의 적절성"},
            {"id": "RB2", "name": "상황 분석 및 논증 타당성", "weight": 0.25, "criteria": "역할 관계 파악, 수정 근거"},
            {"id": "RB3", "name": "업무 담화 수행 적절성", "weight": 0.25, "criteria": "검토 요청 완곡성, 책임 귀속"},
            {"id": "RB4", "name": "과업 충족도 및 제약 준수", "weight": 0.25, "criteria": "5문장, 150자 이내"},
        ]],
    )
    rationale: str = Field(
        default="",
        description="정답 해설. 출제 의도와 정답 근거를 포함.",
        examples=["원문은 구어적 표현으로 격식 위반이 발생한다..."],
    )
    options: List[str] = Field(
        default_factory=list,
        description="객관식 선택지 목록. response_type이 multiple_choice일 때 필수.",
        examples=[["보기1", "보기2", "보기3", "보기4", "보기5"]],
    )
    answer_key: Optional[Any] = Field(
        default=None,
        description=(
            "정답 키. 객관식은 정답 번호(0-based int) 또는 'A'~'E' 문자. "
            "서술형은 모범 답안 문자열 또는 허용 답안 리스트."
        ),
        examples=["2"],
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "추가 메타데이터. "
            "requires_external_knowledge(bool): 외부 지식 필요 여부 명시. "
            "panel_scores(dict): 외부 패널 평정 점수 직접 주입 시 사용."
        ),
        examples=[{"requires_external_knowledge": True}],
    )


class ComponentDetail(BaseModel):
    """요인별 상세 점수"""
    value: float = Field(..., description="요인 점수 (0.0~1.0)")
    confidence: float = Field(..., description="산출 신뢰도 (0.0~1.0)")
    applicable: bool = Field(..., description="해당 요인 적용 여부. 비객관식의 Ddist는 False.")
    evidence: List[str] = Field(..., description="점수 산출 근거 목록")
    warnings: List[str] = Field(..., description="경고 메시지 목록")


class DDIResponse(BaseModel):
    """DDI 산출 응답"""
    item_id: str = Field(..., description="문항 고유 식별자")
    domain: str = Field(..., description="적용된 도메인")
    formula_version: str = Field(..., description="적용 산식 버전. 예: DDI-v1.0")

    # 요인별 점수
    components: Dict[str, ComponentDetail] = Field(
        ...,
        description=(
            "10개 요인별 상세 점수. "
            "키: R(추론), C(맥락), K(한국어특성), P(화용), E(외부지식), "
            "O(개방성), X(제약), Ddist(오답매력도), M(다중구인), Q(문항명료도)"
        ),
    )

    # DDI 점수 단계별
    ddi_global: float = Field(
        ...,
        description="글로벌 가중치 적용 DDI 점수 (0~100). DDI_global = 100 × Σ(w_p × z_fused)",
    )
    ddi_domain: float = Field(
        ...,
        description="도메인 가중치 적용 DDI 점수 (0~100). DDI_domain = 100 × Σ(w_d,p × z_fused)",
    )
    ddi_auto: float = Field(
        ...,
        description="글로벌·도메인 결합 점수 (0~100). DDI_auto = 0.5×DDI_global + 0.5×DDI_domain",
    )
    question_clarity: float = Field(
        ...,
        description="문항 명료도 Q 점수 (0.0~1.0). DDI_clean 보정에 사용됨.",
    )
    ddi_clean: float = Field(
        ...,
        description=(
            "Q 보정 후 최종 DDI 점수 (0~100). "
            "DDI_clean = DDI_auto × (0.5 + 0.5×Q). "
            "불량 문항일수록 낮아짐."
        ),
    )
    ddi_calibrated: float = Field(
        ...,
        description="앵커 보정 후 DDI 점수 (0~100). 앵커 데이터 없으면 ddi_clean과 동일.",
    )
    calibration_source: str = Field(
        ...,
        description="앵커 보정 방식. 'none'(보정 없음), 'global'(전체 보정), 'domain:xxx'(도메인 보정)",
    )

    # 등급 및 품질
    preliminary_level: str = Field(
        ...,
        description="임시 난이도 등급. L1(DDI<35), L2(35≤DDI<70), L3(DDI≥70)",
    )
    confidence: float = Field(
        ...,
        description="전체 산출 신뢰도 (0.0~1.0). 0.6 미만이면 인간 감사 권장.",
    )
    warnings: List[str] = Field(
        ...,
        description="산출 과정 경고 메시지. Q 낮음, 루브릭 누락, Qwen 실패 등.",
    )
    panel_info: Dict[str, Any] = Field(
        ...,
        description="패널 퓨전 정보. panel_used(bool), fallback(bool) 등.",
    )


# ─────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────

app = FastAPI(
    title="DDI 통합 산출 API v2.0",
    description="규칙 엔진(65%) + Qwen 패널(35%) 결합 DDI 산출 엔진",
    version="2.0.0",
)


# 필드명 한국어 매핑
_FIELD_NAMES = {
    "id":            "문항 ID (id)",
    "item_id":       "문항 ID (id)",
    "domain":        "도메인 (domain)",
    "subdomains":    "서브도메인 (subdomains)",
    "response_type": "문항 유형 (response_type)",
    "passage":       "지문 (passage)",
    "question":      "발문 (question)",
    "rubric":        "루브릭 (rubric)",
    "rationale":     "정답 해설 (rationale)",
    "options":       "선택지 (options)",
    "answer_key":    "정답 키 (answer_key)",
    "metadata":      "메타데이터 (metadata)",
    "name":          "루브릭 항목명 (name)",
    "weight":        "루브릭 가중치 (weight)",
    "criteria":      "채점 기준 (criteria)",
}

# 에러 타입별 한국어 메시지
_ERROR_MESSAGES = {
    "missing": "필수 항목입니다. 반드시 입력해주세요.",
    "list_type": (
        "리스트 형식이어야 합니다.\n"
        "올바른 예시:\n"
        '  "rubric": [\n'
        '    {"id": "RB1", "name": "언어 형식 정확성", "weight": 0.25, "criteria": "격식체 일관성"},\n'
        '    {"id": "RB2", "name": "상황 분석", "weight": 0.25, "criteria": "역할 관계 파악"},\n'
        '    {"id": "RB3", "name": "업무 담화", "weight": 0.25, "criteria": "검토 요청 완곡성"},\n'
        '    {"id": "RB4", "name": "과업 충족도", "weight": 0.25, "criteria": "5문장, 150자 이내"}\n'
        "  ]"
    ),
    "string_type":  "문자열이어야 합니다.",
    "float_type":   "숫자여야 합니다. 예: 0.25",
    "int_type":     "정수여야 합니다.",
    "bool_type":    "true 또는 false여야 합니다.",
    "value_error":  "올바르지 않은 값입니다.",
    "greater_than_equal": "0.0 이상이어야 합니다.",
    "less_than_equal":    "1.0 이하이어야 합니다.",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        # loc에서 필드명 추출 (body 제외)
        loc_parts = [str(l) for l in error["loc"] if l != "body"]
        field_raw = loc_parts[-1] if loc_parts else "알 수 없음"
        field_kr = _FIELD_NAMES.get(field_raw, field_raw)
        loc_path = " → ".join(loc_parts) if loc_parts else "알 수 없음"

        # 에러 메시지 한국어 변환
        error_type = error.get("type", "")
        msg = _ERROR_MESSAGES.get(error_type, error.get("msg", "입력값을 확인해주세요."))

        errors.append({
            "항목": f"{field_kr} (경로: {loc_path})",
            "오류": msg,
        })

    return JSONResponse(
        status_code=422,
        content={
            "오류": "입력 형식이 올바르지 않습니다. 아래 항목을 확인해주세요.",
            "상세": errors,
        },
    )

_engine: Optional[DDIEngine] = None


def get_engine() -> DDIEngine:
    global _engine
    if _engine is None:
        qwen = QwenPanel()
        _engine = DDIEngine(qwen_panel=qwen)
    return _engine


@app.post(
    "/api/v1/measure-ddi",
    summary="DDI 자동 산출",
    description=(
        "문항 데이터를 입력받아 DDI(설계난이도지수)를 산출한다.\n\n"
        "**산출 방식**\n"
        "1. 규칙 기반 분석기(65%)로 10개 요인 1차 추정\n"
        "2. LLM 패널(35%)로 의미적 판단 보완\n"
        "3. 글로벌 가중치(50%) + 도메인 가중치(50%) 결합\n"
        "4. 문항 명료도 Q로 보정 → DDI_clean 산출\n\n"
        "**등급 기준**\n"
        "- L1: DDI < 35 (단일 규칙, 모든 모델이 정답)\n"
        "- L2: 35 ≤ DDI < 70 (복합 규칙, 일부 모델만 정답)\n"
        "- L3: DDI ≥ 70 (고차원 추론, 대부분 모델이 오답)\n\n"
        "**문항 유형별 필수 입력**\n"
        "- 객관식: options + answer_key 필수\n"
        "- 서술형/생성형: rubric 필수 (누락 시 Q 점수 하락)"
    ),
    response_model=DDIResponse,
    response_description="DDI 산출 결과. ddi_calibrated와 preliminary_level이 핵심 출력값.",
    responses={
        400: {"description": "입력 오류 (item_id 누락, question 누락 등)"},
        500: {"description": "서버 내부 오류"},
    },
)
async def measure_ddi(payload: ItemPayload) -> DDIResponse:
    try:
        rubric = [
            RubricItem(id=r.id, name=r.name, weight=r.weight, criteria=r.criteria)
            for r in payload.rubric
        ]
        item = DDIItem(
            item_id=payload.item_id,
            passage=payload.passage,
            question=payload.question,
            options=payload.options,
            answer_key=payload.answer_key,
            rationale=payload.rationale,
            rubric=rubric,
            response_type=payload.response_type,
            domain=payload.domain,
            subdomains=payload.subdomains,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = await get_engine().score(item)
        return DDIResponse(**result.to_dict())
    except Exception as exc:
        logger.exception("[%s] DDI 산출 오류", payload.item_id)
        raise HTTPException(status_code=500, detail="내부 처리 오류가 발생했습니다.")


# ─────────────────────────────────────────────
# CLI (배치 처리 및 데모)
# ─────────────────────────────────────────────

def _demo_item() -> DDIItem:
    return DDIItem(
        item_id="DEMO-L3-EMAIL-001",
        domain="business_email",
        subdomains=["honorifics", "pragmatics"],
        response_type="constructed_response",
        passage=(
            "홍길동 자문위원님, 안녕하세요.\n"
            "지방연구원 연구기획팀 주임 연구원 김철수입니다.\n"
            "지난번 말씀하신 부분은 거의 고쳤고요, 예산표는 아직 못 정해서 빼놨습니다.\n"
            "내일 오전에 중간보고 자료를 제출해야 해서 시간이 촉박합니다.\n"
            "확인하시고 틀린 거 있으면 오늘 안으로 알려주세요."
        ),
        question=(
            "다음 2개 문항에 모두 답하시오.\n"
            "1. 위 메일 초안을 과잉 사과나 책임 회피 없이 공문형 업무 메일 전문으로 수정하시오.\n"
            "2. 수정 이유를 5문장, 각 30자 내외, 총 150자 이내로 설명하시오."
        ),
        rubric=[
            RubricItem(id="RB1", name="언어 형식 정확성",       weight=0.25, criteria="격식체 일관성, 경어 적절성"),
            RubricItem(id="RB2", name="상황 분석 및 논증 타당성", weight=0.25, criteria="역할 관계 파악, 수정 근거"),
            RubricItem(id="RB3", name="업무 담화 수행 적절성",   weight=0.25, criteria="검토 요청 완곡성, 책임 귀속"),
            RubricItem(id="RB4", name="과업 충족도 및 제약 준수", weight=0.25, criteria="5문장, 150자 이내"),
        ],
        rationale="원문은 외부 자문위원에게 구어적이고 압박적인 표현을 사용하며 내부 일정 책임을 상대에게 전가하는 인상을 줄 수 있다.",
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _save_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten(result: Mapping[str, Any]) -> Dict[str, Any]:
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
        row[f"{name}_conf"] = comp["confidence"]
    return row


def _save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = [_flatten(r) for r in rows]
    if not flattened:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


async def _run_cli(args: argparse.Namespace) -> int:
    calibrator = None
    if args.anchors:
        calibrator = AnchorCalibrator()
        calibrator.fit(_load_jsonl(args.anchors))

    # CLI 모드: Qwen 연결 시도, 실패해도 규칙 모드로 동작
    try:
        qwen = QwenPanel()
    except Exception:
        qwen = None
        logger.warning("Qwen 초기화 실패 → 규칙 전용 모드로 실행")

    engine = DDIEngine(calibrator=calibrator, qwen_panel=qwen)

    if args.demo:
        result = await engine.score(_demo_item())
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if not args.input or not args.output:
        print("--input과 --output이 필요합니다. 또는 --demo를 사용하세요.", file=sys.stderr)
        return 2

    results = []
    for row in _load_jsonl(args.input):
        try:
            item = DDIItem.from_dict(row)
            results.append((await engine.score(item)).to_dict())
        except Exception as exc:
            item_id = row.get("item_id") or row.get("id") or "UNKNOWN"
            results.append({"item_id": str(item_id), "error": f"{type(exc).__name__}: {exc}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        _save_csv(args.output, [r for r in results if "error" not in r])
    else:
        _save_jsonl(args.output, results)

    print(f"완료: {len(results)}개 문항 → {args.output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DDI 통합 산출 엔진 v2.0")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())