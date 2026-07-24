#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$ROOT/training_cub"

DATA="${DATA:-$ROOT/data/CUB_200_2011}"
RUN="${RUN:-$ROOT/models/cub200_reproduced}"
CACHE="${CACHE:-$ROOT/.cache/salf_cub}"
CONCEPTS="${CONCEPTS:-$RUN/concepts.txt}"
DEVICE="${DEVICE:-auto}"

mkdir -p "$RUN" "$CACHE"

if [[ ! -d "$DATA/images" ]]; then
  bash "$TRAIN_DIR/download_cub.sh" "$ROOT/data"
fi

python "$TRAIN_DIR/salf_cub200.py" make-concepts   --data "$DATA" --output "$CONCEPTS" --limit 370

python "$TRAIN_DIR/salf_cub200.py" train-backbone   --data "$DATA" --output "$RUN/backbone" --device "$DEVICE"

python "$TRAIN_DIR/salf_cub200.py" clip-targets   --data "$DATA" --concept-file "$CONCEPTS"   --output "$CACHE/clip_targets_7x7_r32"   --grid 7 --radius 32 --clip-model ViT-B-16   --clip-pretrained openai --device "$DEVICE"

python "$TRAIN_DIR/salf_cub200.py" cache-features   --data "$DATA"   --backbone "$RUN/backbone/backbone_best.pt"   --output "$CACHE/resnet18_features" --device "$DEVICE"

python "$TRAIN_DIR/salf_cub200.py" train-cbl   --data "$DATA"   --features "$CACHE/resnet18_features"   --targets "$CACHE/clip_targets_7x7_r32"   --concept-file "$CONCEPTS" --output "$RUN"   --grid 7 --batch-size 128 --device "$DEVICE"

python "$TRAIN_DIR/salf_cub200.py" train-classifier   --data "$DATA"   --features "$CACHE/resnet18_features"   --cbl "$RUN/cbl_best.pt" --output "$RUN" --device "$DEVICE"

python "$TRAIN_DIR/evaluate_accuracy.py"   --data "$DATA"   --backbone "$RUN/backbone/backbone_best.pt"   --cbl "$RUN/cbl_best.pt"   --model-dir "$RUN"   --output-json "$RUN/test_accuracy.json"   --device "$DEVICE"

echo "Finished. Results: $RUN/test_accuracy.json"
