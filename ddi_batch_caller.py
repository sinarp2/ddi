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

async def call(client: httpx.AsyncClient, sem: asyncio.Semaphore, payload: dict) -> dict:
    item_id = payload.get("item_id", "")
    async with sem:
        try:
            r = await client.post(API_URL, json=payload, timeout=TIMEOUT)
            result = r.json()
            result["_status"] = "success" if r.status_code == 200 else "error"
            result["item_id"] = item_id
            if r.status_code != 200:
                logger.warning("[%s] HTTP %d → %s", item_id, r.status_code, result)
            else:
                logger.info("[%s] → %s (%.1f점)",
                            item_id,
                            result.get("preliminary_level", "-"),
                            result.get("ddi_calibrated", 0))
            return result
        except Exception as e:
            logger.error("[%s] 실패: %s", item_id, e)
            return {"_status": "exception", "_error": str(e), "item_id": item_id}


async def process_file(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    input_path: Path,
    output_dir: Path,
) -> None:
    items = [json.loads(l) for l in input_path.read_text("utf-8").splitlines() if l.strip()]
    logger.info("[%s] %d개 문항 로드", input_path.name, len(items))

    results = await asyncio.gather(*[call(client, sem, item) for item in items])

    stem = input_path.stem
    # _params 접미사 제거 후 _results 붙이기
    base = stem[:-len("_params")] if stem.endswith("_params") else stem
    out_path = output_dir / f"{base}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success = sum(1 for r in results if r.get("_status") == "success")
    logger.info("[%s] 완료: 성공 %d / 전체 %d → %s", input_path.name, success, len(results), out_path)


async def main(args):
    if args.dir:
        input_files = sorted(Path(args.dir).glob("*_params.jsonl"))
        if not input_files:
            logger.error("처리할 *_params.jsonl 파일 없음: %s", args.dir)
            return
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.dir)
    else:
        input_files = [Path(args.input)]
        output_dir = Path(args.output_dir) if args.output_dir else input_files[0].parent

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("처리 대상: %d개 파일 → 출력: %s", len(input_files), output_dir)

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        for input_path in input_files:
            await process_file(client, sem, input_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",       help="단일 _params.jsonl 파일")
    group.add_argument("--dir",         help="*_params.jsonl 일괄 처리할 폴더")
    parser.add_argument("--output-dir", default=None, help="결과 출력 폴더 (기본: 입력과 같은 폴더)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api-url",     default=API_URL)
    args = parser.parse_args()
    API_URL = args.api_url
    asyncio.run(main(args))