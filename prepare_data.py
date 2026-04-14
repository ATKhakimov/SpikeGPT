"""
Tokenize parquet datasets with rugpt3large tokenizer.
Outputs a numpy uint16 .npy file ready for SpikeGPT training.

Usage:
    python prepare_data.py --taiga-dir data/taiga/data --output data/ru_train.npy
    python prepare_data.py --taiga-dir data/taiga_proza/data --output data/ru_proza.npy --max-files 6
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
DEFAULT_MAX_TOKENS = 2_000_000_000
TEXTS_PER_BATCH = 1024
WRITE_BUFFER = 4_000_000  # flush to disk every ~8 MB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--taiga-dir", type=str, default=TAIGA_DIR)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading tokenizer from: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    eos_id = tokenizer.eos_token_id
    print(f"Tokenizer: {type(tokenizer).__name__}  vocab={len(tokenizer)}  eos={eos_id}", flush=True)

    parquet_files = sorted(glob.glob(os.path.join(args.taiga_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.taiga_dir}")
    if args.max_files:
        parquet_files = parquet_files[:args.max_files]
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  {os.path.basename(f)}")

    raw_path = args.output + ".bin"
    total_tokens = 0
    done = False
    write_buffer = []

    with open(raw_path, "wb") as fout:

        def flush_buffer():
            if write_buffer:
                np.array(write_buffer, dtype=np.uint16).tofile(fout)
                write_buffer.clear()

        for pq_file in parquet_files:
            if done:
                break
            source = os.path.basename(pq_file).split("-")[0]
            print(f"\n[{source}] reading {os.path.basename(pq_file)}...", flush=True)
            table = pq.read_table(pq_file, columns=["text"])
            texts = table.column("text").to_pylist()
            n_batches = (len(texts) + TEXTS_PER_BATCH - 1) // TEXTS_PER_BATCH
            print(f"[{source}] {len(texts):,} texts  ({n_batches} batches)", flush=True)

            for i in range(0, len(texts), TEXTS_PER_BATCH):
                batch = [t for t in texts[i:i + TEXTS_PER_BATCH] if t and t.strip()]
                if not batch:
                    continue

                encoded = tokenizer(batch, add_special_tokens=False,
                                    truncation=False)["input_ids"]
                for ids in encoded:
                    ids.append(eos_id)
                    write_buffer.extend(ids)
                    total_tokens += len(ids)

                if len(write_buffer) >= WRITE_BUFFER:
                    flush_buffer()

                batch_num = i // TEXTS_PER_BATCH + 1
                if batch_num % 5 == 0 or batch_num == n_batches:
                    print(f"  {batch_num}/{n_batches} batches  {total_tokens:>13,} tokens  ({total_tokens*2/1e9:.2f} GB)",
                          flush=True)

                if total_tokens >= args.max_tokens:
                    done = True
                    break

            flush_buffer()
            print(f"  → {total_tokens:>13,} tokens total  ({total_tokens*2/1e9:.2f} GB)", flush=True)

    print(f"\nConverting binary → {args.output} ...", flush=True)
    raw = np.fromfile(raw_path, dtype=np.uint16)[:args.max_tokens]
    np.save(args.output, raw)
    os.remove(raw_path)

    print(f"Done! {len(raw):,} tokens saved to {args.output} ({os.path.getsize(args.output)/1e9:.2f} GB)")
    print(f"\nNext: python train.py")


if __name__ == "__main__":
    main()
