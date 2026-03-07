"""
Run FusionGP on synthetic presentation data.

This creates results suitable for presentations/talks without using unpublished research data.
"""

import sys
from pathlib import Path

# Update the data file path to use presentation data
ORIGINAL_DATA_FILE = Path(__file__).parent.parent / "data" / "dublin_realistic_no2.csv"
PRESENTATION_DATA_FILE = Path(__file__).parent.parent / "data" / "presentation" / "demo_air_quality.csv"

print("="*70)
print("FUSONGP PRESENTATION DEMO")
print("="*70)
print(f"\nUsing synthetic presentation data:")
print(f"  {PRESENTATION_DATA_FILE}")
print(f"\nThis is NOT the actual research data - safe for presentations!")
print("="*70)

# Check if presentation data exists
if not PRESENTATION_DATA_FILE.exists():
    print(f"\n⚠️  Presentation data not found!")
    print(f"   Run: python experiments/generate_presentation_data.py")
    sys.exit(1)

# Temporarily modify the reproduce_paper.py to use presentation data
import importlib.util
spec = importlib.util.spec_from_file_location("reproduce_paper",
                                               Path(__file__).parent / "reproduce_paper.py")
reproduce_paper = importlib.util.module_from_spec(spec)

# Override the DATA_FILE constant
reproduce_paper.DATA_FILE = str(PRESENTATION_DATA_FILE)

print(f"\n✓ Loading presentation data...")

# Run the main experiment
spec.loader.exec_module(reproduce_paper)
