#!/bin/bash
# Drive the K-Search loop against the FlashRT referee.
#
# LLM access is any OpenAI-compatible chat-completions endpoint: set LLM_API_KEY
# and, if you are not using api.openai.com, BASE_URL.
#
#   LLM_API_KEY=sk-... MODEL_NAME=gpt-5.2 bash scripts/run_flashrt_search.sh
#
# Resumable: re-running picks the world model back up via
# --continue-from-world-model auto, so a killed session loses at most one cycle.
# Seed a known-good kernel with CONTINUE_SOLUTION=/abs/path/solution.json.
#
set -uo pipefail

KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FLASHRT_ROOT="${FLASHRT_ROOT:?set FLASHRT_ROOT to your FlashRT checkout}"
# (10,4096,1024) is the FFN gate/up shape: 5.70 ms per inference, the largest
# single decoder-GEMM bucket, and at 34% of roofline the one where a win is most
# likely to be real rather than a measurement artifact. See docs §5.7.
TUNABLE="${TUNABLE:-pi05.decoder.gemm_bf16.dtype_bf16-k_1024-m_10-n_4096}"
MODEL_NAME="${MODEL_NAME:-claude-opus-4-8}"
TARGET="${TARGET:-local-a100}"
# generate_kernels_and_eval.py defaults --target-gpu to H100. The measured profile
# overrides it and the mismatch is warned about, but leaving a wrong name in the
# prompt is a free way to confuse the model about which chip it is writing for.
TARGET_GPU="${TARGET_GPU:-A100}"
# Optional: resume codegen from a known-good kernel instead of the baseline.
# Accepts a path to a persisted solution JSON, or a bare solution name.
CONTINUE_SOLUTION="${CONTINUE_SOLUTION:-}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
# Ceiling on action difficulty (1-5). Empty keeps the selection-policy default.
WM_MAX_DIFFICULTY="${WM_MAX_DIFFICULTY:-}"
ARTIFACTS="${ARTIFACTS:-$KSEARCH_ROOT/.ksearch-flashrt}"
PROFILE="${PROFILE:-$FLASHRT_ROOT/targets/local-a100.json}"

mkdir -p "$ARTIFACTS"


: "${LLM_API_KEY:?set LLM_API_KEY for your chat-completions endpoint}"

LOG="$ARTIFACTS/search-$(date +%Y%m%d-%H%M%S).log"
echo "tunable : $TUNABLE"
echo "model   : $MODEL_NAME @ $BASE_URL"
echo "log     : $LOG"

# K-Search names the artifact subdirectory after the *definition*, not the bare
# tunable id: "flashrt_<tunable>__<target>". Testing the bare id never matched,
# so every restart silently re-initialised the world model from scratch.
DEFINITION="flashrt_${TUNABLE}__${TARGET}"

RESUME=()
[ -f "$ARTIFACTS/$DEFINITION/world_model/world_model.json" ] && \
  RESUME=(--continue-from-world-model auto)
[ -n "$CONTINUE_SOLUTION" ] && \
  RESUME+=(--continue-from-solution "$CONTINUE_SOLUTION")

DIFFICULTY=()
[ -n "$WM_MAX_DIFFICULTY" ] && \
  DIFFICULTY=(--wm-max-difficulty "$WM_MAX_DIFFICULTY")

cd "$KSEARCH_ROOT"
exec > >(tee -a "$LOG") 2>&1
python3 -u generate_kernels_and_eval.py \
  --task-source flashrt \
  --flashrt-tunable "$TUNABLE" \
  --flashrt-referee "$FLASHRT_ROOT/scripts/kopt-referee" \
  --flashrt-target "$TARGET" \
  --target-gpu "$TARGET_GPU" \
  --transport local \
  --language cuda \
  --model-name "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --target-profile "$PROFILE" \
  --world-model \
  --max-opt-rounds "$MAX_ROUNDS" \
  --save-solutions \
  --artifacts-dir "$ARTIFACTS" \
  "${DIFFICULTY[@]}" \
  "${RESUME[@]}"
