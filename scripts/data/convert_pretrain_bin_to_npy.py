"""Convert a tokenized binary shard to the legacy .npy format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/tokenized/pretrain_smoke/spikerugpt-pretrain.manifest.json",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out", default="data/pretrain_smoke.npy")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shard = manifest["shards"][args.shard_index]
    path = Path(shard["path"])
    dtype = np.dtype(shard["dtype"])
    arr = np.fromfile(path, dtype=dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"Read {arr.size:,} tokens from {path} ({dtype})", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
