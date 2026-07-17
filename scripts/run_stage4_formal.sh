#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/hdd/mwz/????2
: "${HF_TOKEN:?HF_TOKEN must be provided in the process environment}"
export PYTHONPATH="$PWD"
export HF_HOME="/home/ubuntu/hdd/mwz/stage4_outputs/hf_cache"
export HF_HUB_DOWNLOAD_TIMEOUT="300"
export HF_HUB_ETAG_TIMEOUT="60"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
PY=/home/ubuntu/hdd/mwz/stage4_venv/bin/python
PARQUET="$PWD/data/iu_xray_hf/data"
MANIFEST="$PWD/evaluation_outputs/manifest.jsonl"
VISION_ADAPTER="/home/ubuntu/hdd/mwz/stage4_outputs/lora/stage4_qwenvl_vision_lora_1024"
TEXT_ADAPTER="/home/ubuntu/hdd/mwz/stage4_outputs/lora/stage4_text_lora_medfindings_ref_512"
OUTDIR="/home/ubuntu/hdd/mwz/stage4_outputs/stage4_qwen3vl32b_medgemma27b_test10"
LOGDIR="/home/ubuntu/hdd/mwz/stage4_outputs/stage4_logs"
mkdir -p "$LOGDIR" "$OUTDIR"

echo "[$(date -Is)] check ollama models"
ollama list | grep -q '^qwen3-vl:32b' || ollama pull qwen3-vl:32b
ollama list | grep -q '^medgemma:27b' || ollama pull medgemma:27b

echo "[$(date -Is)] check/train vision LoRA"
if [ ! -f "$VISION_ADAPTER/adapter_model.safetensors" ]; then
  "$PY" scripts/stage4_real_pipeline.py train-vision \
    --parquet-dir "$PARQUET" --manifest "$MANIFEST" --split development \
    --max-samples 1024 --output-dir "$VISION_ADAPTER" \
    --base-model Qwen/Qwen3-VL-32B-Instruct --epochs 1 \
    --learning-rate 1e-4 --max-length 1024 --grad-accum 8 \
    --lora-r 8 --lora-alpha 16 2>&1 | tee "$LOGDIR/train_vision.log"
else
  echo "vision adapter exists: $VISION_ADAPTER"
fi

echo "[$(date -Is)] check/train text LoRA"
if [ ! -f "$TEXT_ADAPTER/adapter_model.safetensors" ]; then
  "$PY" scripts/stage4_real_pipeline.py train-text \
    --parquet-dir "$PARQUET" --manifest "$MANIFEST" --split development \
    --max-samples 512 --output-dir "$TEXT_ADAPTER" \
    --base-model google/medgemma-27b-text-it --epochs 2 \
    --learning-rate 2e-4 --max-length 1536 --grad-accum 8 \
    --top-k 3 --lora-r 16 --lora-alpha 32 2>&1 | tee "$LOGDIR/train_text.log"
else
  echo "text adapter exists: $TEXT_ADAPTER"
fi

echo "[$(date -Is)] run 10-sample stage4 test"
"$PY" scripts/stage4_real_pipeline.py run \
  --records "$PWD/evaluation_outputs/formal_test100_full/generated_full.jsonl" \
  --source-method full_english_template \
  --output "$OUTDIR/generated_stage4.jsonl" \
  --parquet-dir "$PARQUET" --manifest "$MANIFEST" --train-split development \
  --limit 10 --top-k 3 \
  --vision-base-model Qwen/Qwen3-VL-32B-Instruct --vision-lora "$VISION_ADAPTER" \
  --text-base-model google/medgemma-27b-text-it --text-lora "$TEXT_ADAPTER" \
  --embedding-model openai/clip-vit-base-patch32 2>&1 | tee "$LOGDIR/run_test10.log"

echo "[$(date -Is)] score 10-sample output"
if [ -f scripts/evaluate_reports.py ]; then
  "$PY" scripts/evaluate_reports.py --input "$OUTDIR/generated_stage4.jsonl" --output-dir "$OUTDIR/automatic_basic" 2>&1 | tee "$LOGDIR/score_test10.log"
elif [ -f scripts/run_automatic_metrics.py ]; then
  "$PY" scripts/run_automatic_metrics.py --input "$OUTDIR/generated_stage4.jsonl" --output-dir "$OUTDIR/automatic_basic" 2>&1 | tee "$LOGDIR/score_test10.log"
else
  echo "No scoring script found" | tee "$LOGDIR/score_test10.log"
fi

echo "[$(date -Is)] stage4 formal pipeline complete"
