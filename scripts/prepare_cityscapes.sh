#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${CITYSCAPES_DATASET:-/home/lin/datasets/cityscapes}"
mkdir -p "$DESTINATION"

echo "Official Cityscapes download; credentials are requested interactively by csDownload."
python -m cityscapesscripts.download.downloader \
  --destination_path "$DESTINATION" \
  --resume \
  leftImg8bit_trainvaltest.zip gtFine_trainvaltest.zip

unzip -q -o "$DESTINATION/leftImg8bit_trainvaltest.zip" -d "$DESTINATION"
unzip -q -o "$DESTINATION/gtFine_trainvaltest.zip" -d "$DESTINATION"
echo "Extracted official Cityscapes files under $DESTINATION"
