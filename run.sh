#!/usr/bin/env bash
# usage: ./run.sh <audio_folder> [out_folder]
set -euo pipefail
python run.py --audio-dir "$1" --out "${2:-out}"
