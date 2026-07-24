#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
mkdir -p "$ROOT"
cd "$ROOT"

ARCHIVE="CUB_200_2011.tgz"
DATASET_DIR="CUB_200_2011"

if [[ -d "$DATASET_DIR/images" ]]; then
  echo "CUB-200-2011 already exists at $(pwd)/$DATASET_DIR"
  exit 0
fi

# Official Caltech URL used by the CUB project page.
URLS=(
  "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
  "https://www.vision.caltech.edu/datasets/cub_200_2011/CUB_200_2011.tgz"
  "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
)

downloaded=0
for url in "${URLS[@]}"; do
  echo "Trying: $url"
  if command -v curl >/dev/null 2>&1; then
    if curl -fL --retry 3 --retry-delay 3 "$url" -o "$ARCHIVE"; then
      downloaded=1
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -O "$ARCHIVE" "$url"; then
      downloaded=1
      break
    fi
  else
    echo "Install curl or wget first." >&2
    exit 1
  fi
done

if [[ "$downloaded" -ne 1 ]]; then
  echo "Automatic download failed." >&2
  echo "Download CUB_200_2011.tgz manually from:" >&2
  echo "https://www.vision.caltech.edu/datasets/cub_200_2011/" >&2
  echo "and place it at: $(pwd)/$ARCHIVE" >&2
  exit 1
fi

echo "Extracting..."
tar -xzf "$ARCHIVE"

test -d "$DATASET_DIR/images" || {
  echo "Extraction did not produce $DATASET_DIR/images" >&2
  exit 1
}

echo "Dataset ready at: $(pwd)/$DATASET_DIR"
