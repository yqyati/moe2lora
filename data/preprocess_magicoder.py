"""Download Magicoder-OSS-Instruct-75K and convert it to ShareGPT JSON."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert Magicoder-OSS-Instruct-75K to LlamaFactory sharegpt format")
    parser.add_argument("--limit", type=int, default=None, help="Keep the first N examples")
    parser.add_argument("--output_path", type=str, default="data/magicoder_oss_75k.json",
                        help="Output JSON path")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading ise-uiuc/Magicoder-OSS-Instruct-75K ...")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    print(f"Loaded {len(ds)} samples")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limited to {len(ds)} samples")

    converted = []
    for sample in ds:
        # Magicoder 字段: problem, solution
        user_msg = sample.get("problem") or sample.get("instruction") or ""
        assistant_msg = sample.get("solution") or sample.get("output") or ""

        if not user_msg.strip() or not assistant_msg.strip():
            continue

        converted.append({
            "messages": [
                {"role": "user", "content": user_msg.strip()},
                {"role": "assistant", "content": assistant_msg.strip()},
            ]
        })

    # 保存
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None)

    print(f"Saved {len(converted)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
