# -*- coding: utf-8 -*-
"""
원데이터 -> DDI 엔진 입력 변환기
=================================

원데이터 JSONL을 DDI 엔진 입력 형식으로 변환하고,
quality_status가 approved인 문항에서 앵커 데이터를 자동 생성한다.

실행:
  python converter.py --input raw_items.jsonl --output items.jsonl

앵커 파일은 <output stem>_anchors.jsonl, 에러 리포트는 <output stem>_errors.jsonl 로 자동 생성된다.
앵커 생성을 끄려면 --no-anchors, 에러 리포트를 끄려면 --no-error-report 옵션을 사용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ErrorType:
    PARSE_ERROR   = "PARSE_ERROR"
    MISSING_FIELD = "MISSING_FIELD"
    FORMAT_ERROR  = "FORMAT_ERROR"
    VALUE_ERROR   = "VALUE_ERROR"
    CONVERT_ERROR = "CONVERT_ERROR"


class ConvertError:
    def __init__(self, line_no: int, item_id: str, error_type: str, field: str, message: str):
        self.line_no = line_no
        self.item_id = item_id
        self.error_type = error_type
        self.field = field
        self.message = message

    def display(self) -> str:
        return (
            f"  [{self.error_type}] {self.line_no}행 | item_id={self.item_id} | "
            f"필드={self.field} | {self.message}"
        )


class ErrorCollector:
    def __init__(self):
        self.errors: List[ConvertError] = []

    def add(self, line_no: int, item_id: str, error_type: str, field: str, message: str):
        err = ConvertError(line_no, item_id, error_type, field, message)
        self.errors.append(err)
        print(err.display(), file=sys.stderr)

    def print_summary(self):
        if not self.errors:
            print("\n에러 없음 -- 모든 문항 변환 성공")
            return

        groups: Dict[str, List[ConvertError]] = {}
        for e in self.errors:
            groups.setdefault(e.error_type, []).append(e)

        print("\n" + "=" * 60, file=sys.stderr)
        print(f"에러 요약 -- 총 {len(self.errors)}건", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for etype, errs in sorted(groups.items()):
            print(f"\n  [{etype}] {len(errs)}건", file=sys.stderr)
            for e in errs:
                print(f"    . {e.line_no}행 | {e.item_id} | 필드={e.field} | {e.message}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)


DOMAIN_CODE_MAP: Dict[str, str] = {
    "KOR-D01": "orthography",
    "KOR-D02": "honorifics",
    "KOR-D03": "discourse",
    "KOR-D04": "discourse",
    "KOR-D05": "pragmatics",
    "KOR-D06": "discourse",
    "KOR-D07": "pragmatics",
    "KOR-D08": "general",
    "KOR-D09": "general",
    "KOR-D10": "archaic",
    "KOR-D11": "dialect",
    "KOR-D12": "general",
    "KOR-D13": "inference",
    "KOR-D14": "discourse",
}

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

RESPONSE_TYPE_MAP: Dict[str, str] = {
    "single_choice":        "multiple_choice",
    "multiple_choice":      "multiple_choice",
    "multi_choice":         "multiple_choice",
    "short_answer":         "short_answer",
    "closed_constructed":   "short_answer",
    "open_constructed":     "constructed_response",
    "constructed_response": "constructed_response",
    "essay":                "essay",
    "generation":           "generation",
}

VALID_RESPONSE_TYPES = set(RESPONSE_TYPE_MAP.keys())

CHOICE_ID_MAP: Dict[str, int] = {
    "①": 0, "②": 1, "③": 2, "④": 3, "⑤": 4,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
}

DIFFICULTY_DDI_MAP: Dict[str, float] = {
    "L1": 22.5,
    "L2": 52.5,
    "L3": 82.5,
}

VALID_DIFFICULTIES = set(DIFFICULTY_DDI_MAP.keys())


def validate_item(raw: Dict[str, Any], line_no: int, collector: ErrorCollector) -> bool:
    item_id = raw.get("item_id") or f"(line {line_no})"
    ok = True

    for f in ["item_id", "domain_code", "answer_type", "difficulty"]:
        if not raw.get(f):
            collector.add(line_no, item_id, ErrorType.MISSING_FIELD, f,
                          f"'{f}' 필드가 없거나 비어 있음")
            ok = False

    domain_code = raw.get("domain_code") or ""
    if domain_code and domain_code not in DOMAIN_CODE_MAP:
        collector.add(line_no, item_id, ErrorType.VALUE_ERROR, "domain_code",
                      f"알 수 없는 도메인 코드: '{domain_code}' (허용값: KOR-D01 ~ KOR-D14)")
        ok = False

    difficulty = raw.get("difficulty") or ""
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        collector.add(line_no, item_id, ErrorType.VALUE_ERROR, "difficulty",
                      f"알 수 없는 난이도: '{difficulty}' (허용값: L1, L2, L3)")
        ok = False

    answer_type = raw.get("answer_type") or ""
    if answer_type and answer_type not in VALID_RESPONSE_TYPES:
        collector.add(line_no, item_id, ErrorType.VALUE_ERROR, "answer_type",
                      f"알 수 없는 answer_type: '{answer_type}' "
                      f"(허용값: {', '.join(sorted(VALID_RESPONSE_TYPES))})")
        ok = False

    if answer_type in ("single_choice", "multiple_choice", "multi_choice"):
        choices = raw.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            collector.add(line_no, item_id, ErrorType.MISSING_FIELD, "choices",
                          "객관식인데 'choices' 필드가 없거나 비어 있음")
            ok = False
        else:
            for i, c in enumerate(choices):
                if not isinstance(c, dict):
                    collector.add(line_no, item_id, ErrorType.FORMAT_ERROR, f"choices[{i}]",
                                  f"선택지가 dict가 아님: {type(c).__name__}")
                    ok = False
                elif not c.get("text"):
                    collector.add(line_no, item_id, ErrorType.MISSING_FIELD, f"choices[{i}].text",
                                  f"선택지 {i+1}번의 'text' 필드가 없거나 비어 있음")

        answer = raw.get("answer") or {}
        choice_id = str(answer.get("choice_id") or "").strip()
        if not choice_id:
            collector.add(line_no, item_id, ErrorType.MISSING_FIELD, "answer.choice_id",
                          "객관식인데 'answer.choice_id'가 없음")
            ok = False
        elif choice_id not in CHOICE_ID_MAP:
            collector.add(line_no, item_id, ErrorType.VALUE_ERROR, "answer.choice_id",
                          f"인식할 수 없는 choice_id: '{choice_id}' "
                          f"(허용값: 원문자 1-5 또는 숫자 1-5 또는 A-E)")
            ok = False

    if not raw.get("user_prompt") and not raw.get("task_instruction"):
        collector.add(line_no, item_id, ErrorType.MISSING_FIELD, "user_prompt",
                      "'user_prompt' 또는 'task_instruction' 중 하나는 있어야 함")

    return ok


def extract_subdomains(raw: Dict[str, Any]) -> List[str]:
    subdomains: List[str] = []
    domain_name = raw.get("domain_name") or ""
    for key, tags in DOMAIN_NAME_SUBDOMAINS.items():
        if key in domain_name:
            subdomains.extend(tags)
    basis = raw.get("difficulty_basis") or ""
    for keyword, tags in BASIS_KEYWORD_MAP.items():
        if keyword in basis:
            subdomains.extend(tags)
    return list(dict.fromkeys(subdomains))


def extract_options(raw: Dict[str, Any]) -> List[str]:
    choices = raw.get("choices") or []
    return [c.get("text", "") for c in choices if isinstance(c, dict)]


def extract_answer_key(raw: Dict[str, Any]) -> Optional[int]:
    answer = raw.get("answer") or {}
    choice_id = str(answer.get("choice_id") or "").strip()
    return CHOICE_ID_MAP.get(choice_id)


def extract_rubric(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    rubric = raw.get("rubric")
    if not rubric:
        return []
    if isinstance(rubric, list):
        n = len(rubric)
        result = []
        for i, r in enumerate(rubric):
            if not isinstance(r, dict):
                continue
            # 워크밴치: criterion/score 필드 / 구형: name/criteria/weight 필드
            name = r.get("name") or r.get("criterion") or f"채점기준{i+1}"
            criteria = r.get("criteria") or r.get("description") or r.get("criterion") or ""
            raw_weight = r.get("weight") or r.get("score")
            weight = float(raw_weight) if raw_weight is not None else (1.0 / n)
            result.append({
                "id": r.get("id") or f"RB{i+1}",
                "name": name,
                "weight": weight,
                "criteria": criteria,
            })
        # score 기반이면 합계로 정규화
        total = sum(r["weight"] for r in result)
        if total > 0 and not (0.99 < total < 1.01):
            for r in result:
                r["weight"] = round(r["weight"] / total, 6)
        return result
    if isinstance(rubric, str) and rubric.strip():
        return [{"id": "RB1", "name": "채점 기준", "weight": 1.0, "criteria": rubric}]
    return []


def convert_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    domain_code = raw.get("domain_code") or ""
    domain = DOMAIN_CODE_MAP.get(domain_code, "general")
    response_type = RESPONSE_TYPE_MAP.get(raw.get("answer_type") or "", "constructed_response")
    is_mcq = response_type == "multiple_choice"

    item: Dict[str, Any] = {
        "item_id": raw.get("item_id") or "",
        "domain": domain,
        "subdomains": extract_subdomains(raw),
        "response_type": response_type,
        "passage": (raw.get("context") or {}).get("passage") if isinstance(raw.get("context"), dict) else (raw.get("context") or ""),
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
        if raw.get("distractor_rationale"):
            item["distractor_rationale"] = raw["distractor_rationale"]

    if response_type == "short_answer":
        if raw.get("acceptable_answers"):
            item["acceptable_answers"] = raw["acceptable_answers"]
        if raw.get("unacceptable_answers"):
            item["unacceptable_answers"] = raw["unacceptable_answers"]

    if raw.get("checkpoints"):
        item["checkpoints"] = raw["checkpoints"]

    return item


def convert_to_anchor(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        "ddi_clean": None,
    }


def load_jsonl(path: Path, collector: ErrorCollector) -> List[tuple]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                collector.add(line_no, f"(line {line_no})", ErrorType.PARSE_ERROR,
                              "(raw)", f"JSON 파싱 실패: {exc}")
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(
    input_path: Path,
    output_dir: Path,
    no_anchors: bool,
    no_error_report: bool,
    strict: bool,
    collector: ErrorCollector,
) -> None:
    stem = input_path.stem
    out_path          = output_dir / f"{stem}_params.jsonl"
    anchors_path      = None if no_anchors      else output_dir / f"{stem}_params_anchors.jsonl"
    error_report_path = None if no_error_report else output_dir / f"{stem}_params_errors.jsonl"

    print(f"\n읽는 중: {input_path}")
    raw_pairs = load_jsonl(input_path, collector)
    print(f"  총 {len(raw_pairs)}개 문항 로드")

    converted: List[Dict[str, Any]] = []
    error_raws: List[Dict[str, Any]] = []

    for line_no, raw in raw_pairs:
        is_valid = validate_item(raw, line_no, collector)
        if not is_valid and strict:
            error_raws.append(raw)
            continue
        try:
            converted.append(convert_item(raw))
            if not is_valid:
                error_raws.append(raw)
        except Exception as exc:
            item_id = raw.get("item_id") or f"(line {line_no})"
            collector.add(line_no, item_id, ErrorType.CONVERT_ERROR, "(변환 전체)",
                          f"예상치 못한 예외: {type(exc).__name__}: {exc}")
            error_raws.append(raw)

    save_jsonl(out_path, converted)
    print(f"  변환 완료: {len(converted)}개 -> {out_path}")

    skipped = len(raw_pairs) - len(converted)
    if skipped:
        print(f"  건너뜀: {skipped}개 (--strict 모드 검증 실패)")

    if error_report_path and error_raws:
        save_jsonl(error_report_path, error_raws)
        print(f"  에러/경고 원본: {len(error_raws)}개 -> {error_report_path}")

    if anchors_path:
        anchors = [a for _, raw in raw_pairs if (a := convert_to_anchor(raw)) is not None]
        save_jsonl(anchors_path, anchors)
        print(f"  앵커: {len(anchors)}개 -> {anchors_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="원데이터 -> DDI 엔진 입력 변환기")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  type=Path, help="원데이터 JSONL 파일 (단일)")
    group.add_argument("--dir",    type=Path, help="원데이터 JSONL 폴더 (일괄 처리)")
    parser.add_argument("--output-dir",      type=Path, default=None,
                        help="출력 폴더 (기본: --input 파일과 같은 폴더 / --dir 지정 시 같은 폴더)")
    parser.add_argument("--no-anchors",      action="store_true", help="앵커 자동 생성 비활성화")
    parser.add_argument("--no-error-report", action="store_true", help="에러 리포트 자동 생성 비활성화")
    parser.add_argument("--strict",          action="store_true",
                        help="검증 실패 문항은 변환 결과에서 제외")
    args = parser.parse_args(argv)

    if args.dir:
        input_files = sorted(args.dir.glob("*.jsonl"))
        # _params / _anchors / _errors 파일은 재처리 제외
        input_files = [f for f in input_files
                       if not any(f.stem.endswith(s) for s in ("_params", "_params_anchors", "_params_errors"))]
        if not input_files:
            print(f"처리할 JSONL 파일 없음: {args.dir}", file=sys.stderr)
            return 1
        output_dir = args.output_dir or args.dir
    else:
        input_files = [args.input]
        output_dir = args.output_dir or args.input.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    collector = ErrorCollector()

    print(f"출력 폴더: {output_dir}")
    print(f"처리 대상: {len(input_files)}개 파일")

    for input_path in input_files:
        process_file(input_path, output_dir, args.no_anchors, args.no_error_report, args.strict, collector)

    collector.print_summary()
    return 1 if collector.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())