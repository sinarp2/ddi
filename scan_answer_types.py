# -*- coding: utf-8 -*-
"""
폴더 내 모든 JSONL의 answer_type 값을 추출한다.
문항 내용은 출력하지 않음.

실행:
  python scan_answer_types.py --dir ./data
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="스캔할 폴더 경로")
    args = parser.parse_args()

    counter: Counter = Counter()
    total_files = 0

    for jsonl_path in sorted(args.dir.rglob("*.jsonl")):
        total_files += 1
        with jsonl_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    val = obj.get("answer_type") or "(없음)"
                    counter[val] += 1
                except json.JSONDecodeError:
                    counter["(파싱오류)"] += 1

    print(f"스캔 완료: {total_files}개 파일\n")
    print(f"{'answer_type':<30} {'건수':>6}")
    print("-" * 38)
    for val, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{val:<30} {count:>6}")


if __name__ == "__main__":
    main()
