# Testing Plan for Experiment 6.1 Bug Fix

**Date:** 2025-10-18
**Bug:** Key mismatch in results aggregation causing empty results.json
**Fix Applied:** Changed `t['falsified']` → `t['falsification_rate']` + validation

---

## Quick Test (5 minutes)

### Test 1: Verify Fix with n=5 pairs

```bash
cd /home/aaron/projects/xai

# Run minimal test
python experiments/run_final_experiment_6_1.py \
  --n_pairs 5 \
  --device cuda \
  --output_dir experiments/test_fix_minimal \
  --seed 42

# Check results
cat experiments/test_fix_minimal/exp6_1_n5_*/results.json | jq '.methods | keys'
```

**Expected Output:**
```json
[
  "Biometric Grad-CAM",
  "Geodesic IG",
  "Grad-CAM",
  "LIME",
  "SHAP"
]
```

**If this works, proceed to Test 2.**

---

## Medium Test (15 minutes)

### Test 2: Verify Statistics with n=20 pairs

```bash
cd /home/aaron/projects/xai

python experiments/run_final_experiment_6_1.py \
  --n_pairs 20 \
  --device cuda \
  --output_dir experiments/test_fix_medium \
  --seed 42

# Verify results structure
cat experiments/test_fix_medium/exp6_1_n20_*/results.json | jq '.methods["Grad-CAM"]'
```

**Expected Output:**
```json
{
  "falsification_rate_mean": <number between 0-100>,
  "falsification_rate_std": <number between 0-100>,
  "confidence_interval": {
    "lower": <number>,
    "upper": <number>,
    "level": 0.95
  },
  "n_samples": 20,
  "raw_falsification_rates": [<array of 20 numbers>]
}
```

**Validate:**
- All 5 methods present
- Each method has 20 samples
- Falsification rates in [0, 100] range
- Statistical tests populated

**If this works, proceed to Production Test.**

---

## Production Test (90 minutes)

### Test 3: Full n=500 experiment

```bash
cd /home/aaron/projects/xai

# Run full experiment
python experiments/run_final_experiment_6_1.py \
  --n_pairs 500 \
  --device cuda \
  --output_dir experiments/production_facenet_n500_FIXED \
  --seed 42 \
  2>&1 | tee experiments/production_facenet_n500_FIXED/stdout.log
```

**Monitor Progress:**
```bash
# In another terminal
watch -n 10 'ls experiments/production_facenet_n500_FIXED/exp6_1_n500_*/visualizations/ | wc -l'
```

**After Completion:**

```bash
# Verify results
RESULTS_DIR=$(ls -d experiments/production_facenet_n500_FIXED/exp6_1_n500_* | tail -1)

# 1. Check all methods present
jq '.methods | keys' $RESULTS_DIR/results.json

# 2. Check sample counts
jq '.methods[] | .n_samples' $RESULTS_DIR/results.json

# 3. Check falsification rates
jq '.methods[] | {method: .falsification_rate_mean, std: .falsification_rate_std}' $RESULTS_DIR/results.json

# 4. Check statistical tests
jq '.statistical_tests | keys | length' $RESULTS_DIR/results.json

# 5. Verify visualizations
ls $RESULTS_DIR/visualizations/*.png | wc -l
```

**Expected:**
- 5 methods with results
- 500 samples per method
- Falsification rates in reasonable range (0-100%)
- 10 statistical test comparisons (5 choose 2)
- 500 visualization files

---

## Validation Checklist

### Phase 1: Code Validation
- [x] Bug identified (key mismatch: 'falsified' vs 'falsification_rate')
- [x] Fix applied to line 434 (changed key name)
- [x] Fix applied to lines 445-446 (removed * 100)
- [x] Validation added after line 395 (schema check)
- [x] Validation added after line 461 (results count check)
- [ ] Unit test created
- [ ] Integration test created

### Phase 2: Small-Scale Testing
- [ ] Test 1 passed (n=5)
- [ ] Test 2 passed (n=20)
- [ ] No errors in logs
- [ ] All 5 methods produce results
- [ ] Falsification rates in valid range [0, 100]

### Phase 3: Production Testing
- [ ] Test 3 running (n=500)
- [ ] Progress monitoring shows files being created
- [ ] Experiment completes without errors
- [ ] All validation checks pass
- [ ] Results.json fully populated

### Phase 4: Results Analysis
- [ ] Compare with previous experiments (if any)
- [ ] Falsification rates reasonable
- [ ] Statistical tests show expected patterns
- [ ] Confidence intervals computed correctly
- [ ] Ready for dissertation inclusion

---

## Rollback Plan

If the fix fails:

```bash
cd /home/aaron/projects/xai
git checkout experiments/run_final_experiment_6_1.py

# Restore original (buggy) version
# Then debug further
```

---

## Success Criteria

**The fix is successful if:**

1. ✅ No errors during execution
2. ✅ All 5 attribution methods produce results
3. ✅ Each method has 500 samples (or n_pairs samples)
4. ✅ Falsification rates in [0, 100] range
5. ✅ Statistical tests computed (10 comparisons for 5 methods)
6. ✅ results.json contains complete data
7. ✅ Visualizations match expectations (500 files)

**If all criteria met:** Bug fix is validated, proceed with dissertation analysis.

**If any criteria fail:** Review logs, identify new issues, update fix.

---

## Debugging Commands

If tests fail, use these to diagnose:

```bash
# Check for errors in logs
grep -i "error\|critical" experiments/test_fix_*/exp6_1_*/stdout.log

# Check what keys are actually in results
python3 << 'EOF'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

if 'methods' in data and data['methods']:
    print("Methods found:", list(data['methods'].keys()))
    for method, stats in data['methods'].items():
        print(f"\n{method}:")
        print(f"  n_samples: {stats.get('n_samples', 'N/A')}")
        print(f"  FR mean: {stats.get('falsification_rate_mean', 'N/A'):.2f}%")
else:
    print("ERROR: No methods in results!")
    print("Available keys:", list(data.keys()))
EOF
```

---

## Timeline

- **Test 1 (n=5):** 5 minutes
- **Test 2 (n=20):** 15 minutes
- **Test 3 (n=500):** 90 minutes
- **Validation:** 10 minutes
- **Total:** ~2 hours

---

## Next Steps After Success

1. ✅ Commit fix to git (if using version control)
2. ✅ Document in CHANGELOG.md
3. ✅ Update dissertation with real results
4. ✅ Generate publication-quality plots
5. ✅ Archive old (empty) results
6. ✅ Run additional experiments if needed (n=1000)

---

**Created:** 2025-10-18
**Status:** Ready for testing
