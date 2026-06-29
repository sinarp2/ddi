# -*- coding: utf-8 -*-
"""
DDI 배치 호출기

실행:
  pip install httpx
  python ddi_batch_caller.py --input items.jsonl --output results.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

API_URL  = "http://localhost:8000/api/v1/measure-ddi"
TIMEOUT  = 120.0

CHOICE_ID_MAP = {
    "①": 0, "②": 1, "③": 2, "④": 3, "⑤": 4,
    "1":  0, "2":  1, "3":  2, "4":  3, "5":  4,
    "A":  0, "B":  1, "C":  2, "D":  3, "E":  4,
}

DOMAIN_MAP = {
    "맞춤법·띄어쓰기": "orthography", "맞춤법": "orthography",
    "경어법": "honorifics", "높임말": "honorifics",
    "담화": "discourse", "문체": "discourse", "레지스터": "discourse",
    "화용": "pragmatics", "화행": "pragmatics", "공손": "pragmatics",
    "고어": "archaic", "방언": "dialect",
    "추론": "inference", "문맥": "inference", "무형대용어": "inference",
    "업무 이메일": "business_email",
}


def to_ddi(item: dict) -> dict:
    choices = item.get("choices") or []
    options = [c.get("text", "") for c in choices]

    answer   = item.get("answer") or {}
    cid      = answer.get("choice_id", "") if isinstance(answer, dict) else str(answer)
    ans_key  = CHOICE_ID_MAP.get(str(cid).strip())

    domain_name = item.get("domain_name", "")
    domain = next((v for k, v in DOMAIN_MAP.items() if k in domain_name), "general")

    return {
        "id":            item.get("item_id", ""),
        "domain":        domain,
        "subdomains":    [domain_name] if domain_name else [],
        "response_type": "multiple_choice",
        "passage":       (lambda c: c.get("passage", "") if isinstance(c, dict) else (c or ""))(item.get("context")),
        "question":      item.get("user_prompt") or item.get("task_instruction") or "",
        "options":       options,
        "answer_key":    ans_key,
        "rationale":     item.get("explanation") or "",
        "rubric":        [],
    }


async def call(client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict) -> dict:
    async with sem:
        try:
            r = await client.post(API_URL, json=payload, timeout=TIMEOUT)
            result = r.json()
            result["_status"] = "success" if r.status_code == 200 else "error"
            result["id"] = payload["id"]  # API 응답에 id 없어도 항상 보장
            if r.status_code != 200:
                logger.warning("[%s] HTTP %d → %s", payload["id"], r.status_code, result)
            else:
                logger.info("[%s] %s → %s (%.1f점)",
                            payload["id"],
                            result.get("_status"),
                            result.get("preliminary_level", "-"),
                            result.get("ddi_calibrated", 0))
            return result
        except Exception as e:
            logger.error("[%s] 실패: %s", payload["id"], e)
            return {"_status": "exception", "_error": str(e), "id": payload["id"]}


async def main(args):
    # 읽기 (상대 경로는 스크립트 위치 기준으로 해석)
    _script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _script_dir / input_path
    lines = [json.loads(l) for l in input_path.read_text("utf-8").splitlines() if l.strip()]

    # 필터 (choices 있는 것만)
    mcq = [item for item in lines if item.get("choices")]
    logger.info("전체 %d개 중 객관식 %d개", len(lines), len(mcq))

    # 변환
    payloads = [to_ddi(item) for item in mcq]

    # 호출
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[call(client, sem, p) for p in payloads])

    # 저장
    out = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_result")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success = sum(1 for r in results if r.get("_status") == "success")
    logger.info("완료: 성공 %d / 전체 %d → %s", success, len(results), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      default=None)  # 생략 시 입력파일명_result.jsonl
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api-url",     default=API_URL)
    args = parser.parse_args()
    API_URL = args.api_url
    asyncio.run(main(args))