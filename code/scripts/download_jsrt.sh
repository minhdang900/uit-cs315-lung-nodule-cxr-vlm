#!/bin/bash
# Download JSRT: official DICOM release (free, imgcom.jsrt.or.jp) + clinical info.
# The official clinical files are for registered users only (the registration page db.jsrt.or.jp is gone);
# this uses the copy in the HybridGNet repository (github.com/ngaggion/HybridGNet). Research use only.
# Usage: bash scripts/download_jsrt.sh <raw_dir>
set -euo pipefail
RAW=${1:?usage: download_jsrt.sh <raw_dir>}; mkdir -p "$RAW"; cd "$RAW"
for f in 2017/11/Nodule154images.zip 2016/01/NonNodule93images.zip; do
  curl -sSfL --retry 3 -O "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/$f"; unzip -q -o "$(basename $f)"; rm "$(basename $f)"
done
GH=https://raw.githubusercontent.com/ngaggion/HybridGNet/main/Datasets/JSRT/Clinical_Information
curl -sSfL -o CLNDAT_EN.txt "$GH/CLNDAT_EN.txt"
curl -sSfL -o CNNDAT_EN.TXT "$GH/CNNDAT_EN.TXT"
echo "$(find . -iname '*.dcm' | wc -l) DICOM files (expect 247)"
