# -*- coding: utf-8 -*-
"""
폴더 내 모든 JSONL의 필드 구조(스키마)를 추출한다.
값은 출력하지 않고 필드명과 값 타입만 표시.

실행:
  python scan_schema.py --dir ./data
"""

import argparse
import json
from pathlib import Path


def infer_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        if not value:
            return "list[]"
        inner = infer_type(value[0])
        return f"list[{inner}]"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def merge_schema(schema: dict, obj: dict, prefix: str = "") -> None:
    for key, val in obj.items():
        full_key = f"{prefix}{key}"
        t = infer_type(val)
        if full_key not in schema:
            schema[full_key] = set()
        schema[full_key].add(t)
        if isinstance(val, dict):
            merge_schema(schema, val, prefix=f"{full_key}.")
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            merge_schema(schema, val[0], prefix=f"{full_key}[].")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="스캔할 폴더 경로")
    args = parser.parse_args()

    schema: dict = {}
    total_files = 0
    total_items = 0

    for jsonl_path in sorted(args.dir.rglob("*.jsonl")):
        total_files += 1
        with jsonl_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    merge_schema(schema, obj)
                    total_items += 1
                except json.JSONDecodeError:
                    pass

    print(f"스캔 완료: {total_files}개 파일, {total_items}개 문항\n")
    print(f"{'필드':<45} {'타입'}")
    print("-" * 60)
    for field, types in sorted(schema.items()):
        print(f"{field:<45} {' | '.join(sorted(types))}")


if __name__ == "__main__":
    main()
