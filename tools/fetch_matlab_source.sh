#!/usr/bin/env bash
# Fetches the original NLvib MATLAB source from GitHub for reference comparison.
# The MATLAB source is NOT bundled in this repo (GPL-3.0 requires we link, not copy).
# Running this script downloads it into tools/NLvib_matlab/ for local use only.
#
# Usage:
#   bash tools/fetch_matlab_source.sh             # clones NLvib-Basic branch
#   bash tools/fetch_matlab_source.sh peace       # clones NLvib-PEACE branch
#
# Original authors: Malte Krack & Johann Gross (University of Stuttgart)
# Source: https://github.com/maltekrack/NLvib

set -euo pipefail

BRANCH="${1:-NLvib-Basic}"
TARGET_DIR="$(dirname "$0")/NLvib_matlab"

if [ -d "$TARGET_DIR" ]; then
  echo "NLvib MATLAB source already present at $TARGET_DIR"
  echo "To re-fetch, delete the directory first: rm -rf $TARGET_DIR"
  exit 0
fi

echo "Cloning NLvib MATLAB source (branch: $BRANCH) ..."
git clone \
  --branch "$BRANCH" \
  --depth 1 \
  https://github.com/maltekrack/NLvib.git \
  "$TARGET_DIR"

echo ""
echo "Done. MATLAB source at: $TARGET_DIR"
echo ""
echo "To generate reference fixtures (requires MATLAB or Octave):"
echo "  matlab -batch \"run('$TARGET_DIR/EXAMPLES/01_Duffing/Duffing.m')\""
echo "  or"
echo "  octave $TARGET_DIR/EXAMPLES/01_Duffing/Duffing.m"
echo ""
echo "To generate all fixtures via the Python helper:"
echo "  python tools/generate_fixtures.py"
