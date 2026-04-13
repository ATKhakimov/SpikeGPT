"""
Tokenize cointegrated/taiga_stripped_rest (parquet files) with rugpt3large tokenizer.
Outputs a numpy uint16 .npy file ready for SpikeGPT training.

Usage:
    python prepare_data.py [--max-tokens 2000000000] [--output data/ru_train.npy]
"""
import os
import argparse
import glob
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TOKENIZER_PATH = "tokenizer/rugpt3"
TAIGA_DIR = "data/taiga/data"
DEFAULT_OUTPUT = "data/ru_train.npy"
DEFAULT_MAX_TOKENS = 2_000_000_000  # 2B tokens = ~4 GB on disk
TEXTS_PER_BATCH = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--taiga-dir", type=str, default=TAIGA_DIR)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of parquet files to process")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id  # 50256
    assert len(tokenizer) == 50257, f"Expected vocab 50257, got {len(tokenizer)}"
    print(f"Vocab size: {len(tokenizer)}  EOS id: {eos_id}")

    parquet_files = sorted(glob.glob(os.path.join(args.taiga_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {args.taiga_dir}")
    if args.max_files:
        parquet_files = parquet_files[:args.max_files]
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  {os.path.basename(f)}")

    raw_path = args.output + ".bin"
    total_tokens = 0
    done = False

    with open(raw_path, "wb") as fout:
        for pq_file in parquet_files:
            if done:
                break
            source = os.path.basename(pq_file).split("-")[0]
            table = pq.read_table(pq_file, columns=["text"])
            texts = table.column("text").to_pylist()
            print(f"\n[{source}] {len(texts):,} texts", flush=True)

            for i in range(0, len(texts), TEXTS_PER_BATCH):
                batch = [t for t in texts[i:i + TEXTS_PER_BATCH] if t and t.strip()]
                if not batch:
                    continue

                encoded = tokenizer(batch, add_special_tokens=False,
                                    truncation=False)["input_ids"]
                for ids in encoded:
                    ids.append(eos_id)
                    np.array(ids, dtype=np.uint16).tofile(fout)
                    total_tokens += len(ids)

                if total_tokens >= args.max_tokens:
                    done = True
                    break

            print(f"  → {total_tokens:>13,} tokens total  ({total_tokens*2/1e9:.2f} GB)",
                  flush=True)

    print(f"\nConverting binary → {args.output} ...")
    raw = np.fromfile(raw_path, dtype=np.uint16)[:args.max_tokens]
    np.save(args.output, raw)
    os.remove(raw_path)

    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Done! {len(raw):,} tokens saved to {args.output} ({size_gb:.2f} GB)")
    print(f"\nNext: python train.py")


if __name__ == "__main__":
    main()
