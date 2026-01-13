#!/bin/bash
# Verify tensor logic + homology formalization
set -e

cd "$(dirname "$0")/.."

echo "=== Tensor Logic + Homology Verification ==="
echo ""

echo "[1/4] Checking for sorry/admit..."
if grep -r "sorry\|admit" HeytingLean --include="*.lean" | grep -v "-- sorry" | grep -v "-- admit"; then
    echo "ERROR: Found sorry/admit in source files"
    exit 1
fi
echo "OK: No sorry/admit found"
echo ""

echo "[2/4] Building library (strict mode)..."
lake build --wfail
echo "OK: Library build passed"
echo ""

echo "[3/4] Building executables..."
lake build tensor_logic_cli tensor_homology_cli homology_cli
echo "OK: Executables built"
echo ""

echo "[4/4] Running demos..."
echo "  tensor_homology_cli (S² demo):"
lake exe tensor_homology_cli
echo ""
echo "  tensor_logic_cli (sphere2_as_logic):"
lake exe tensor_logic_cli --rules data/homology/sphere2_as_logic.rules.json --facts data/homology/sphere2_as_logic.facts.tsv --mode boolean --pred connected | head -5
echo "  ..."
echo ""

echo "=== VERIFICATION PASSED ==="
