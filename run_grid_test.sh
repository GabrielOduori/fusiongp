#!/bin/bash
# Quick script to run grid structure analysis

echo "========================================================================"
echo "FusionGP Grid Structure Test"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Analyze your data's grid structure"
echo "  2. Benchmark kernel evaluation strategies"
echo "  3. Test inducing point initialization"
echo "  4. Generate diagnostic visualizations"
echo ""
echo "Results will be saved to: outputs/"
echo ""
echo "========================================================================"
echo ""

# Run the test
python tests/test_grid_structure.py "$@"

echo ""
echo "========================================================================"
echo "Test complete!"
echo ""
echo "Check outputs/ for:"
echo "  - grid_structure_analysis.png"
echo "  - inducing_points_comparison.png"
echo "========================================================================"
