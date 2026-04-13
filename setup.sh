#!/bin/bash
# SpikeGPT Russian — full setup script
# Idempotent: safe to re-run, skips already completed steps
#
# Usage (from any directory on a fresh machine):
#   HF_TOKEN=hf_xxx bash setup.sh
#   bash setup.sh --token hf_xxx
#   bash setup.sh --token hf_xxx --proza-files 20

set -e

# ── Parse args ────────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:-}"
PROZA_FILES=5   # how many proza parquet files to download (each ~245 MB, ~350M tokens)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --token)       HF_TOKEN="$2";    shift 2 ;;
        --proza-files) PROZA_FILES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is required. Pass via env or --token flag."
    echo "  HF_TOKEN=hf_xxx bash setup.sh"
    exit 1
fi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "${YELLOW}[--]${NC} $1"; }

# ── Step 0: Clone SpikeGPT repo ───────────────────────────────────────────────
REPO_URL="https://github.com/ridgerchu/SpikeGPT.git"
if [[ -d "SpikeGPT/.git" ]]; then
    ok "SpikeGPT repo already cloned"
else
    info "Cloning SpikeGPT..."
    git clone "$REPO_URL" SpikeGPT
    ok "SpikeGPT cloned"
fi

cd SpikeGPT
SCRIPT_DIR="$(pwd)"

# ── Step 1: Python dependencies ───────────────────────────────────────────────
info "Checking Python dependencies..."
python3 -c "
import importlib, sys
needed = {
    'transformers': '>=4.45',
    'datasets':     '>=2.20',
    'huggingface_hub': '>=0.20',
    'pyarrow':      '>=14',
    'numpy':        '>=1.26',
    'accelerate':   '>=0.34',
    'tqdm':         '>=4.66',
}
missing = []
for pkg in needed:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print('MISSING: ' + ' '.join(missing))
    sys.exit(1)
print('all present')
" && ok "Dependencies OK" || {
    info "Installing missing packages..."
    pip install -r requirements_cuda129_notorch.txt --quiet
    ok "Dependencies installed"
}

# ── Step 2: Tokenizer ─────────────────────────────────────────────────────────
TOKENIZER_DIR="tokenizer/rugpt3"
if [[ -f "$TOKENIZER_DIR/vocab.json" && -f "$TOKENIZER_DIR/merges.txt" ]]; then
    ok "Tokenizer already downloaded"
else
    info "Downloading rugpt3large tokenizer..."
    mkdir -p "$TOKENIZER_DIR"
    python3 - <<EOF
from huggingface_hub import hf_hub_download
for f in ['vocab.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer_config.json']:
    hf_hub_download(
        repo_id='ai-forever/rugpt3large_based_on_gpt2',
        filename=f,
        local_dir='$TOKENIZER_DIR',
        token='$HF_TOKEN'
    )
    print(f'  {f}')
EOF
    ok "Tokenizer downloaded"
fi

# ── Step 3: Download taiga_stripped_rest ──────────────────────────────────────
TAIGA_REST_DIR="data/taiga/data"
TAIGA_REST_MARKER="data/taiga/.done"
if [[ -f "$TAIGA_REST_MARKER" ]]; then
    ok "taiga_stripped_rest already downloaded"
else
    info "Downloading cointegrated/taiga_stripped_rest (~2.1 GB)..."
    mkdir -p data/taiga
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='cointegrated/taiga_stripped_rest',
    repo_type='dataset',
    local_dir='data/taiga',
    token='$HF_TOKEN',
    ignore_patterns=['*.md', '.gitattributes']
)
print('done')
EOF
    touch "$TAIGA_REST_MARKER"
    ok "taiga_stripped_rest downloaded"
fi

# ── Step 4: Download taiga_proza (first N files) ──────────────────────────────
PROZA_MARKER="data/taiga_proza/.done_${PROZA_FILES}files"
if [[ -f "$PROZA_MARKER" ]]; then
    ok "taiga_proza (${PROZA_FILES} files) already downloaded"
else
    info "Downloading cointegrated/taiga_proza — first ${PROZA_FILES} files (~$((PROZA_FILES * 245)) MB)..."
    mkdir -p data/taiga_proza/data
    python3 - <<EOF
from huggingface_hub import HfApi, hf_hub_download
import os

api = HfApi()
all_files = sorted([
    f for f in api.list_repo_files(
        'cointegrated/taiga_proza', repo_type='dataset', token='$HF_TOKEN'
    )
    if f.startswith('data/') and f.endswith('.parquet')
])

files_to_get = all_files[:$PROZA_FILES]
for i, f in enumerate(files_to_get, 1):
    print(f'  [{i}/{len(files_to_get)}] {os.path.basename(f)}')
    hf_hub_download(
        repo_id='cointegrated/taiga_proza',
        repo_type='dataset',
        filename=f,
        local_dir='data/taiga_proza',
        token='$HF_TOKEN'
    )
print('done')
EOF
    touch "$PROZA_MARKER"
    ok "taiga_proza downloaded"
fi

# ── Step 5: Tokenize taiga_stripped_rest ──────────────────────────────────────
if [[ -f "data/ru_train.npy" ]]; then
    ok "data/ru_train.npy already exists"
else
    info "Tokenizing taiga_stripped_rest..."
    python3 prepare_data.py \
        --taiga-dir data/taiga/data \
        --output data/ru_train.npy
    ok "data/ru_train.npy ready"
fi

# ── Step 6: Tokenize taiga_proza ──────────────────────────────────────────────
if [[ -f "data/ru_proza.npy" ]]; then
    ok "data/ru_proza.npy already exists"
else
    info "Tokenizing taiga_proza (${PROZA_FILES} files)..."
    python3 prepare_data.py \
        --taiga-dir data/taiga_proza/data \
        --output data/ru_proza.npy \
        --max-files "$PROZA_FILES"
    ok "data/ru_proza.npy ready"
fi

# ── Step 7: Merge datasets ────────────────────────────────────────────────────
if [[ -f "data/ru_train_full.npy" ]]; then
    ok "data/ru_train_full.npy already exists"
else
    info "Merging datasets..."
    python3 - <<'EOF'
import numpy as np, os
a = np.load('data/ru_train.npy', mmap_mode='r')
b = np.load('data/ru_proza.npy', mmap_mode='r')
out = np.concatenate([a, b])
np.save('data/ru_train_full.npy', out)
gb = os.path.getsize('data/ru_train_full.npy') / 1e9
print(f'  Total: {len(out):,} tokens  ({gb:.2f} GB)')
EOF
    ok "data/ru_train_full.npy ready"
fi

# ── Step 8: Create checkpoints dir ───────────────────────────────────────────
mkdir -p checkpoints

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - <<'EOF'
import numpy as np, os

files = {
    'Tokenizer':  'tokenizer/rugpt3/vocab.json',
    'Train+Val':  'data/ru_train_full.npy',
}
for name, path in files.items():
    if os.path.exists(path):
        if path.endswith('.npy'):
            d = np.load(path, mmap_mode='r')
            print(f"  {name}: {len(d):,} tokens  ({os.path.getsize(path)/1e9:.2f} GB)")
        else:
            print(f"  {name}: OK")
    else:
        print(f"  {name}: MISSING")
EOF
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
ok "Setup complete. Run: python train.py"
