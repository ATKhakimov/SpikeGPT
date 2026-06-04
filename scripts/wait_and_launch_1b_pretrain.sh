#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="data/tokenized/pretrain_1b/spikerugpt-pretrain.manifest.json"
TOKENIZER="tokenizer/spikerugpt-bpe-32k.model"
VALIDATION_DIR="data/validation_text"
BASE_CKPT="checkpoints/autonomous/autonomous-ctx1024-12h/final.pt"
WATCH_LOG="reports/wait_and_launch_1b_pretrain.log"

EARLY_RUN="autonomous-ctx1024-1b-bf16-early4h"
EARLY_DIR="checkpoints/autonomous/${EARLY_RUN}"
EARLY_REPORT="reports/${EARLY_RUN}.json"
EARLY_METRICS="reports/${EARLY_RUN}.metrics.jsonl"

LONG_RUN="autonomous-ctx1024-1b-bf16-5d"
LONG_DIR="checkpoints/autonomous/${LONG_RUN}"
LONG_REPORT="reports/${LONG_RUN}.json"
LONG_METRICS="reports/${LONG_RUN}.metrics.jsonl"

mkdir -p reports checkpoints/autonomous

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$WATCH_LOG"
}

manifest_tokens() {
  python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit
j = json.loads(path.read_text())
print(int(j.get("written_tokens") or sum(s.get("tokens", 0) for s in j.get("shards", []))))
PY
}

report_ok() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
j = json.loads(path.read_text())
if j.get("status") != "ok":
    raise SystemExit(2)
if not j.get("gate", {}).get("ok"):
    raise SystemExit(3)
train = j.get("train", {})
if train.get("final_loss") is None:
    raise SystemExit(4)
if train.get("run_tokens_seen", 0) <= 0:
    raise SystemExit(5)
PY
}

log "watcher started; waiting for ${MANIFEST}"

while true; do
  tokens="$(manifest_tokens)"
  if [[ "$tokens" -ge 970000000 ]]; then
    log "manifest ready tokens=${tokens}"
    break
  fi
  log "waiting manifest tokens=${tokens}; sleep=300s"
  sleep 300
done

log "starting early gate run ${EARLY_RUN}"
PYTHONUNBUFFERED=1 python scripts/run_autonomous_training.py \
  --run-id "$EARLY_RUN" \
  --manifest "$MANIFEST" \
  --tokenizer "$TOKENIZER" \
  --validation-dir "$VALIDATION_DIR" \
  --precision bf16 \
  --batch-size 22 \
  --max-wall-time-sec $((4 * 60 * 60)) \
  --min-steps 1 \
  --max-steps 1000000 \
  --lr 3e-4 \
  --log-every 10 \
  --save-every-sec 1800 \
  --eval-batches 8 \
  --eval-batch-size 2 \
  --resume-from "$BASE_CKPT" \
  --checkpoint-dir "$EARLY_DIR" \
  --report "$EARLY_REPORT" \
  --metrics-jsonl "$EARLY_METRICS" \
  --hf-repo-id "Koras1k/spikerugpt-autonomous-runs" \
  --progress-bar 2>&1 | tee "reports/${EARLY_RUN}.log"

if ! report_ok "$EARLY_REPORT"; then
  log "early gate failed; not starting long run; report=${EARLY_REPORT}"
  exit 10
fi

log "early gate ok; starting long run ${LONG_RUN}"
PYTHONUNBUFFERED=1 python scripts/run_autonomous_training.py \
  --run-id "$LONG_RUN" \
  --manifest "$MANIFEST" \
  --tokenizer "$TOKENIZER" \
  --validation-dir "$VALIDATION_DIR" \
  --precision bf16 \
  --batch-size 22 \
  --max-wall-time-sec $((5 * 24 * 60 * 60)) \
  --min-steps 1 \
  --max-steps 1000000 \
  --lr 3e-4 \
  --log-every 10 \
  --save-every-sec 1800 \
  --eval-batches 8 \
  --eval-batch-size 2 \
  --resume-from "${EARLY_DIR}/final.pt" \
  --checkpoint-dir "$LONG_DIR" \
  --report "$LONG_REPORT" \
  --metrics-jsonl "$LONG_METRICS" \
  --hf-repo-id "Koras1k/spikerugpt-autonomous-runs" \
  --progress-bar 2>&1 | tee "reports/${LONG_RUN}.log"

log "long run finished report=${LONG_REPORT}"
