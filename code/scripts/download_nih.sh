#!/bin/bash
# Download what prepare_nih.py needs from the official NIH Clinical Center Box share.
# Only the images in the manifest (positives with a Nodule/Mass box + 3 "No Finding" negatives each,
# seed 0) are kept; each archive is streamed and filtered, so ~45 GB is read but little is stored.
# Usage: bash scripts/download_nih.sh <raw_dir>        (run from code/, package installed)
set -euo pipefail
RAW=${1:?usage: download_nih.sh <raw_dir>}; mkdir -p "$RAW"; RAW=$(cd "$RAW" && pwd)
BOX="https://nihcc.app.box.com/index.php?rm=box_download_shared_file&vanity_name=ChestXray-NIHCC&file_id="
curl -sSfL -o "$RAW/Data_Entry_2017_v2020.csv" "${BOX}f_219760887468"
curl -sSfL -o "$RAW/BBox_List_2017.csv" "${BOX}f_219760940956"
ln -sf Data_Entry_2017_v2020.csv "$RAW/Data_Entry_2017.csv"
python - "$RAW" <<'PY'
import sys, pandas as pd
from pathlib import Path
from nodulecxr.data.nih import build_nih_manifest
raw = Path(sys.argv[1])
man = build_nih_manifest(pd.read_csv(raw / "Data_Entry_2017.csv"), pd.read_csv(raw / "BBox_List_2017.csv"), 3, seed=0)
(raw / "needed_members.txt").write_text("".join(f"images/{i}\n" for i in man["image_id"]))
print(len(man), "images needed")
PY
cd "$RAW"
for pair in 001:f_219764235225 002:f_219767703471 003:f_219770039352 004:f_221185642661 005:f_219776556743 \
            006:f_219777758783 007:f_220610700915 008:f_219776273384 009:f_219782291318 010:f_219781375034 \
            011:f_219777519815 012:f_219778785923; do
  n=${pair%%:*}; [ -f ".done_$n" ] && continue
  echo "$(date +%T) images_$n.tar.gz"
  set +e; curl -sSL --retry 5 "${BOX}${pair#*:}" | tar -xzf - -T needed_members.txt 2>>tar_warnings.log
  st=${PIPESTATUS[0]}; set -e
  [ "$st" = 0 ] && touch ".done_$n" || echo "curl failed ($st) on $n; re-run to resume"
done
echo "kept $(ls images | wc -l) of $(wc -l < needed_members.txt) images"
