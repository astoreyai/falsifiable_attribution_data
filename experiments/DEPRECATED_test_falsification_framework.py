#!/usr/bin/env python3
"""
⚠️ DEPRECATED - DO NOT USE ⚠️

This test validates the OLD (FLAWED) distance-based falsification framework.

DATE DEPRECATED: October 18, 2025
REASON: Circular logic - sorts by distance then tests if sorted values differ
REPLACEMENT: test_regional_falsification.py

The OLD approach:
1. Generated counterfactuals on hypersphere (no spatial info)
2. Sorted by distance
3. Tested if high-distance > low-distance (trivially true!)

The NEW approach (test_regional_falsification.py):
1. Masks high-attribution spatial regions and recomputes embeddings
2. Masks low-attribution spatial regions and recomputes embeddings
3. Tests if masking high-attr causes larger changes (causal test!)

See:
- /home/aaron/projects/xai/experiments/WEEK_1_DAY_5_COMPLETION_REPORT.md
- /home/aaron/projects/xai/experiments/test_regional_falsification.py

DO NOT RUN THIS FILE - Use test_regional_falsification.py instead.
"""

import sys
print("=" * 60)
print("⚠️  DEPRECATED TEST FILE ⚠️")
print("=" * 60)
print()
print("This test validates the OLD (FLAWED) falsification framework.")
print()
print("Reason: Circular logic - sorts by distance then tests if sorted")
print("        values differ. This doesn't test the attribution!")
print()
print("Replacement: test_regional_falsification.py")
print()
print("The NEW implementation:")
print("  - Masks spatial regions identified by attribution map")
print("  - Recomputes embeddings after masking")
print("  - Tests CAUSAL relationship (does masking cause changes?)")
print()
print("See WEEK_1_DAY_5_COMPLETION_REPORT.md for details.")
print("=" * 60)
sys.exit(1)
