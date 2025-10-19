# Immediate Action Plan - Experiment 6.1 Bug Fix

**Status:** ✅ Bug fixed, ready for testing
**Date:** 2025-10-18
**Priority:** HIGH (PhD dissertation results)

---

## WHAT HAPPENED

The n=500 production experiment ran successfully but produced an **empty results.json** due to a key name mismatch in the aggregation code.

- ✅ All 500 pairs processed
- ✅ All 5 attribution methods computed
- ✅ All 500 visualizations saved
- ❌ Results aggregation failed silently
- ❌ Empty results.json written

**Cause:** Code looked for `t['falsified']` but function returns `t['falsification_rate']`

---

## WHAT WAS DONE

### 1. Root Cause Analysis ✅
- Identified exact line causing failure (line 422)
- Traced through execution flow
- Found secondary bug (incorrect * 100 multiplication)

### 2. Fix Applied ✅
- Changed `t['falsified']` → `t['falsification_rate']` (line 434)
- Removed `* 100` from percentage calculation (lines 445-446)
- Added schema validation (lines 395-403)
- Added summary validation (lines 463-475)
- Improved error messages (lines 437-440)

### 3. Documentation Created ✅
- BUG_REPORT_EXPERIMENT_6_1.md (1,000+ lines, forensic analysis)
- BUG_FIX_SUMMARY.md (executive summary)
- TESTING_PLAN_FIX.md (step-by-step testing)
- FIX_EXPERIMENT_6_1.patch (git patch file)
- ACTION_PLAN.md (this file)

---

## NEXT STEPS

### Step 1: Quick Verification (5 min) - DO THIS FIRST

```bash
cd /home/aaron/projects/xai

# Test with minimal dataset
python experiments/run_final_experiment_6_1.py \
  --n_pairs 5 \
  --device cuda \
  --output_dir experiments/test_fix_quick \
  --seed 42
```

**Check results:**
```bash
cat experiments/test_fix_quick/exp6_1_n5_*/results.json | jq '.methods | keys'
```

**Expected:** All 5 methods listed (not empty `[]`)

**If this fails:** Check logs, review fix, contact for help
**If this works:** Proceed to Step 2

---

### Step 2: Medium Verification (15 min) - VALIDATE STATISTICS

```bash
# Test with n=20 for statistical validation
python experiments/run_final_experiment_6_1.py \
  --n_pairs 20 \
  --device cuda \
  --output_dir experiments/test_fix_medium \
  --seed 42
```

**Check results:**
```bash
RESULTS=$(ls -d experiments/test_fix_medium/exp6_1_n20_* | tail -1)
cat $RESULTS/results.json | jq '.methods["Grad-CAM"]'
```

**Expected:**
- `n_samples: 20`
- `falsification_rate_mean` in range [0, 100]
- `raw_falsification_rates` has 20 values

**If this fails:** Review statistical aggregation
**If this works:** Proceed to Step 3

---

### Step 3: Production Re-run (90 min) - FINAL RESULTS

```bash
# Re-run full n=500 experiment
python experiments/run_final_experiment_6_1.py \
  --n_pairs 500 \
  --device cuda \
  --output_dir experiments/production_facenet_n500_FIXED \
  --seed 42 \
  2>&1 | tee experiments/production_facenet_n500_FIXED/run.log
```

**Monitor progress:**
```bash
# In another terminal
watch -n 30 'ls experiments/production_facenet_n500_FIXED/exp6_1_n500_*/visualizations/ 2>/dev/null | wc -l'
```

**After completion (expected: ~90 minutes):**

```bash
RESULTS=$(ls -d experiments/production_facenet_n500_FIXED/exp6_1_n500_* | tail -1)

# 1. Verify all methods present
echo "=== Methods ==="
jq '.methods | keys' $RESULTS/results.json

# 2. Verify sample counts
echo -e "\n=== Sample Counts ==="
jq '.methods | to_entries[] | "\(.key): \(.value.n_samples) samples"' $RESULTS/results.json -r

# 3. Verify falsification rates
echo -e "\n=== Falsification Rates ==="
jq '.methods | to_entries[] | "\(.key): \(.value.falsification_rate_mean)% ± \(.value.falsification_rate_std)%"' $RESULTS/results.json -r

# 4. Verify statistical tests
echo -e "\n=== Statistical Tests ==="
jq '.statistical_tests | keys | length' $RESULTS/results.json

# 5. Verify visualizations
echo -e "\n=== Visualizations ==="
ls $RESULTS/visualizations/*.png | wc -l
```

**Expected:**
- 5 methods with results
- 500 samples per method
- Reasonable falsification rates (likely 30-70% range)
- 10 statistical comparisons
- 500 visualization files

---

## SUCCESS CRITERIA

The fix is validated when:

1. ✅ All 5 attribution methods produce results
2. ✅ Each method has n_pairs samples
3. ✅ Falsification rates in valid range [0, 100]
4. ✅ No errors in execution logs
5. ✅ results.json fully populated (not empty)
6. ✅ Statistical tests computed
7. ✅ Visualizations created

**If ALL pass:** Bug fix successful, use results for dissertation
**If ANY fail:** Debug further, check logs, review code

---

## TROUBLESHOOTING

### Issue: Test 1 still shows empty results

**Check:**
```bash
# View the actual error
tail -50 experiments/test_fix_quick/exp6_1_n5_*/run.log

# Check if fix was applied
grep "BUG FIX" experiments/run_final_experiment_6_1.py
```

**Solution:** Verify file was saved, re-apply edits if needed

---

### Issue: Falsification rates are 0 or 100 for all methods

**Check:**
```bash
# View raw rates
jq '.methods[] | .raw_falsification_rates[:5]' $RESULTS/results.json
```

**Possible causes:**
- Attribution maps too uniform
- Thresholds (theta_high=0.7, theta_low=0.3) too restrictive
- Model not producing meaningful gradients

**Solution:** Review attribution visualizations, adjust thresholds

---

### Issue: Some methods missing from results

**Check logs:**
```bash
grep "CRITICAL\|ERROR" experiments/*/exp6_1_*/run.log
```

**Common causes:**
- Specific attribution method failing
- CUDA out of memory
- Invalid gradients

**Solution:** Review method-specific errors, increase error handling

---

## FILE LOCATIONS

**Fixed Script:**
```
/home/aaron/projects/xai/experiments/run_final_experiment_6_1.py
```

**Documentation:**
```
/home/aaron/projects/xai/BUG_REPORT_EXPERIMENT_6_1.md
/home/aaron/projects/xai/BUG_FIX_SUMMARY.md
/home/aaron/projects/xai/TESTING_PLAN_FIX.md
/home/aaron/projects/xai/FIX_EXPERIMENT_6_1.patch
/home/aaron/projects/xai/ACTION_PLAN.md (this file)
```

**Test Results (will be created):**
```
/home/aaron/projects/xai/experiments/test_fix_quick/
/home/aaron/projects/xai/experiments/test_fix_medium/
/home/aaron/projects/xai/experiments/production_facenet_n500_FIXED/
```

**Old (Buggy) Results:**
```
/home/aaron/projects/xai/experiments/production_facenet_n500/exp6_1_n500_20251018_214202/
```

---

## TIMELINE

| Task | Duration | Status |
|------|----------|--------|
| Bug analysis | 30 min | ✅ Complete |
| Fix implementation | 15 min | ✅ Complete |
| Documentation | 30 min | ✅ Complete |
| Test 1 (n=5) | 5 min | ⏳ Pending |
| Test 2 (n=20) | 15 min | ⏳ Pending |
| Test 3 (n=500) | 90 min | ⏳ Pending |
| Validation | 10 min | ⏳ Pending |
| **TOTAL** | **~3 hours** | **Progress: 50%** |

---

## RISK ASSESSMENT

**Risk Level:** LOW

**Reasons:**
1. Fix is surgical (only aggregation code changed)
2. Core computation unchanged (attributions, falsification tests)
3. Validation added to catch future issues
4. Can verify with small tests before full run

**Mitigation:**
- Test incrementally (n=5, n=20, n=500)
- Monitor during execution
- Keep old results for comparison

---

## AFTER SUCCESS

Once all tests pass:

1. **Archive old results:**
   ```bash
   mv experiments/production_facenet_n500 experiments/ARCHIVE_buggy_run_20251018
   ```

2. **Use fixed results for dissertation:**
   ```bash
   cp experiments/production_facenet_n500_FIXED/exp6_1_n500_*/results.json dissertation/data/
   ```

3. **Update dissertation text:**
   - Add results tables
   - Generate plots from results.json
   - Write analysis section

4. **Optional: Run n=1000 for stronger statistics**
   ```bash
   python experiments/run_final_experiment_6_1.py --n_pairs 1000 --device cuda --output_dir experiments/production_facenet_n1000
   ```

---

## CONTACT

**If stuck:**
1. Check BUG_REPORT_EXPERIMENT_6_1.md for detailed analysis
2. Check TESTING_PLAN_FIX.md for step-by-step testing
3. Review logs in experiments/*/exp6_1_*/run.log
4. Check GitHub issues (if using git)

---

**Status:** Ready to execute
**Next Action:** Run Test 1 (Step 1 above)
**Estimated Time to Resolution:** 2-3 hours
