# -*- coding: utf-8 -*-
"""
원데이터 → DDI 엔진 입력 변환기
=================================

원데이터 JSONL을 DDI 엔진 입력 형식으로 변환하고,
quality_status가 'approved'인 문항에서 앵커 데이터를 자동 생성한다.

실행:
  # 변환만
  python converter.py --input raw_items.jsonl --output items.jsonl

  # 변환 + 앵커 생성
  python converter.py --input raw_items.jsonl --output items.jsonl --anchors anchors.jsonl

  # DDI 엔진까지 연결해서 한 번에
  python converter.py --input raw_items.jsonl --output items.jsonl --anchors anchors.jsonl --run-ddi
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# 도메인 코드 → DDI 엔진 도메인명 매핑
# ─────────────────────────────────────────────

DOMAIN_CODE_MAP: Dict[str, str] = {
    "KOR-D01": "orthography",    # 맞춤법·띄어쓰기
    "KOR-D02": "honorifics",     # 경어법
    "KOR-D03": "discourse",      # 문장성분 호응
    "KOR-D04": "discourse",      # 담화 연결·접속
    "KOR-D05": "pragmatics",     # 문체/레지스터 전환
    "KOR-D06": "discourse",      # 문맥 이해도(화시)
    "KOR-D07": "pragmatics",     # 대화 화행·공손 전략
    "KOR-D08": "general",        # 관용표현
    "KOR-D09": "general",        # 유의어
    "KOR-D10": "archaic",        # 고어
    "KOR-D11": "dialect",        # 방언
    "KOR-D12": "general",        # 신조어
    "KOR-D13": "inference",      # 추론
    "KOR-D14": "discourse",      # 무형대용어 복원
}

# 도메인명 → DDI 서브도메인 태그
DOMAIN_NAME_SUBDOMAINS: Dict[str, List[str]] = {
    "맞춤법·띄어쓰기":    ["orthography"],
    "경어법":            ["honorifics", "pragmatics"],
    "문장성분 호응":      ["grammar", "discourse"],
    "담화 연결·접속":     ["discourse", "cohesion"],
    "문체/레지스터 전환": ["register", "pragmatics"],
    "문맥 이해도":        ["discourse", "deixis"],
    "화시":              ["discourse", "deixis"],
    "대화 화행":          ["pragmatics", "speech_act"],
    "공손 전략":          ["pragmatics", "politeness"],
    "관용표현":           ["lexical", "idiom"],
    "유의어":             ["lexical", "synonym"],
    "고어":              ["archaic", "korean"],
    "방언":              ["dialect", "korean"],
    "신조어":             ["lexical", "neologism"],
    "추론":              ["inference", "reasoning"],
    "무형대용어":         ["discourse", "anaphora"],
}

# difficulty_basis 키워드 → 추가 서브도메인
BASIS_KEYWORD_MAP: Dict[str, List[str]] = {
    "맞춤법":   ["orthography"],
    "띄어쓰기": ["orthography"],
    "경어법":   ["honorifics"],
    "높임법":   ["honorifics"],
    "담화":     ["discourse"],
    "접속":     ["cohesion"],
    "문체":     ["register"],
    "레지스터": ["register"],
    "화시":     ["deixis"],
    "화행":     ["speech_act"],
    "공손":     ["politeness"],
    "관용":     ["idiom"],
    "유의어":   ["synonym"],
    "고어":     ["archaic"],
    "방언":     ["dialect"],
    "신조어":   ["neologism"],
    "추론":     ["reasoning"],
    "대용어":   ["anaphora"],
    "문법":     ["grammar"],
    "어휘":     ["lexical"],
    "규정":     ["orthography"],
}

# answer_type → response_type
RESPONSE_TYPE_MAP: Dict[str, str] = {
    "single_choice":    "multiple_choice",
    "multiple_choice":  "multiple_choice",
    "short_answer":     "short_answer",
    "constructed_response": "constructed_response",
    "essay":            "essay",
    "generation":       "generation",
}

# 원형 선택지 번호 → 0-based 인덱스
CHOICE_ID_MAP: Dict[str, int] = {
    "①": 0, "②": 1, "③": 2, "④": 3, "⑤": 4,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
}

# difficulty → expert_ddi 중간값
DIFFICULTY_DDI_MAP: Dict[str, float] = {
    "L1": 22.5,   # 0~35 중간
    "L2": 52.5,   # 35~70 중간
    "L3": 82.5,   # 70~100 중간
}


# ─────────────────────────────────────────────
# 변환 함수
# ─────────────────────────────────────────────

def extract_subdomains(raw: Dict[str, Any]) -> List[str]:
    """domain_name + difficulty_basis에서 서브도메인 태그 추출."""
    subdomains: List[str] = []

    # domain_name 기반
    domain_name = raw.get("domain_name") or ""
    for key, tags in DOMAIN_NAME_SUBDOMAINS.items():
        if key in domain_name:
            subdomains.extend(tags)

    # difficulty_basis 기반
    basis = raw.get("difficulty_basis") or ""
    for keyword, tags in BASIS_KEYWORD_MAP.items():
        if keyword in basis:
            subdomains.extend(tags)

    return list(dict.fromkeys(subdomains))  # 중복 제거, 순서 유지


def extract_options(raw: Dict[str, Any]) -> List[str]:
    """choices 리스트에서 선택지 텍스트 추출."""
    choices = raw.get("choices") or []
    return [c.get("text", "") for c in choices if isinstance(c, dict)]


def extract_answer_key(raw: Dict[str, Any]) -> Optional[int]:
    """answer.choice_id를 0-based 인덱스로 변환."""
    answer = raw.get("answer") or {}
    choice_id = str(answer.get("choice_id") or "").strip()
    return CHOICE_ID_MAP.get(choice_id)


def extract_rubric(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """rubric 필드 변환. 원데이터가 null이면 빈 리스트."""
    rubric = raw.get("rubric")
    if not rubric:
        return []
    if isinstance(rubric, list):
        result = []
        for i, r in enumerate(rubric):
            if isinstance(r, dict):
                result.append({
                    "id": r.get("id") or f"RB{i+1}",
                    "name": r.get("name") or r.get("criteria") or f"채점기준{i+1}",
                    "weight": float(r.get("weight") or (1.0 / len(rubric))),
                    "criteria": r.get("criteria") or r.get("description") or "",
                })
        return result
    if isinstance(rubric, str) and rubric.strip():
        return [{"id": "RB1", "name": "채점 기준", "weight": 1.0, "criteria": rubric}]
    return []


def convert_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """원데이터 단일 문항을 DDI 엔진 입력 형식으로 변환."""
    domain_code = raw.get("domain_code") or ""
    domain = DOMAIN_CODE_MAP.get(domain_code, "general")
    response_type = RESPONSE_TYPE_MAP.get(
        raw.get("answer_type") or "", "constructed_response"
    )
    is_mcq = response_type == "multiple_choice"

    item: Dict[str, Any] = {
        "item_id": raw.get("item_id") or "",
        "domain": domain,
        "subdomains": extract_subdomains(raw),
        "response_type": response_type,
        "passage": raw.get("context") or "",
        "question": raw.get("user_prompt") or raw.get("task_instruction") or "",
        "rationale": raw.get("explanation") or "",
        "rubric": extract_rubric(raw),
        "metadata": {
            "domain_name": raw.get("domain_name") or "",
            "difficulty": raw.get("difficulty") or "",
            "difficulty_basis": raw.get("difficulty_basis") or "",
            "quality_status": raw.get("quality_status") or "",
            "source_type": raw.get("source_type") or "",
            "creator_id": raw.get("creator_id") or "",
        },
    }

    if is_mcq:
        item["options"] = extract_options(raw)
        item["answer_key"] = extract_answer_key(raw)

    return item


def convert_to_anchor(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    quality_status='approved'인 문항에서 앵커 레코드 생성.
    difficulty → expert_ddi 중간값으로 변환.
    """
    if raw.get("quality_status") != "approved":
        return None
    difficulty = raw.get("difficulty") or ""
    expert_ddi = DIFFICULTY_DDI_MAP.get(difficulty)
    if expert_ddi is None:
        return None

    domain_code = raw.get("domain_code") or ""
    return {
        "item_id": raw.get("item_id") or "",
        "domain": DOMAIN_CODE_MAP.get(domain_code, "general"),
        "difficulty": difficulty,
        "difficulty_basis": raw.get("difficulty_basis") or "",
        "expert_ddi": expert_ddi,
        # ddi_clean은 DDI 엔진 실행 후 채워짐 (지금은 placeholder)
        "ddi_clean": None,
    }


# ─────────────────────────────────────────────
# JSONL 입출력
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  경고: {path} {line_no}행 파싱 오류 — {exc}", file=sys.stderr)
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="원데이터 → DDI 엔진 입력 변환기")
    parser.add_argument("--input",   type=Path, required=True, help="원데이터 JSONL 파일")
    parser.add_argument("--output",  type=Path, required=True, help="DDI 엔진 입력 JSONL 출력 경로")
    parser.add_argument("--anchors", type=Path, help="앵커 데이터 출력 경로 (approved 문항만)")
    parser.add_argument("--run-ddi", action="store_true", help="변환 후 DDI 엔진(main.py) 자동 실행")
    parser.add_argument("--ddi-output", type=Path, help="DDI 결과 출력 경로 (--run-ddi 시 사용)")
    args = parser.parse_args(argv)

    print(f"읽는 중: {args.input}")
    raw_items = load_jsonl(args.input)
    print(f"  총 {len(raw_items)}개 문항 로드")

    # 변환
    converted, skipped = [], []
    for raw in raw_items:
        try:
            converted.append(convert_item(raw))
        except Exception as exc:
            item_id = raw.get("item_id") or "UNKNOWN"
            skipped.append(item_id)
            print(f"  경고: [{item_id}] 변환 실패 — {exc}", file=sys.stderr)

    save_jsonl(args.output, converted)
    print(f"변환 완료: {len(converted)}개 → {args.output}")
    if skipped:
        print(f"  건너뜀: {len(skipped)}개 ({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})")

    # 앵커 생성
    if args.anchors:
        anchors = [a for raw in raw_items if (a := convert_to_anchor(raw)) is not None]
        save_jsonl(args.anchors, anchors)
        print(f"앵커 생성: {len(anchors)}개 (approved 문항) → {args.anchors}")

        # difficulty별 분포 출력
        dist: Dict[str, int] = {}
        for a in anchors:
            dist[a["difficulty"]] = dist.get(a["difficulty"], 0) + 1
        for level, count in sorted(dist.items()):
            print(f"  {level}: {count}개 (expert_ddi={DIFFICULTY_DDI_MAP.get(level, '?')})")

    # DDI 엔진 자동 실행
    if args.run_ddi:
        ddi_out = args.ddi_output or args.output.parent / "ddi_results.jsonl"
        cmd = [sys.executable, "main.py", "--input", str(args.output), "--output", str(ddi_out)]
        if args.anchors:
            cmd += ["--anchors", str(args.anchors)]
        print(f"\nDDI 엔진 실행 중: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("DDI 엔진 실행 실패", file=sys.stderr)
            return result.returncode
        print(f"DDI 결과 → {ddi_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
