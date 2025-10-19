# AGENT 2 REPRODUCIBILITY EXPERT - EXECUTIVE SUMMARY

**Date:** October 19, 2025
**Agent:** AGENT 2 (Reproducibility Expert)
**Mission:** Diagnose Exp 6.1 vs Exp 6.4 inconsistency (10.48% FR vs 0.0% FR)

---

## FINDING: CRITICAL BUG IDENTIFIED ✅

The inconsistency is **NOT** due to sampling variability or different face pairs.

The inconsistency **IS** due to an **implementation bug** in Experiment 6.4.

---

## THE BUG

**Location:** `experiments/run_real_experiment_6_4.py`, line 368-369

**Wrong Code:**
```python
is_falsified = falsification_result.get('falsified', False)  # ❌ WRONG KEY
pair_frs.append(1.0 if is_falsified else 0.0)
```

**What It Should Be:**
```python
falsification_rate = falsification_result.get('falsification_rate', 0.0)  # ✅ CORRECT
pair_frs.append(falsification_rate)
```

**Why This Causes 0.0% FR:**
- The function `falsification_test()` returns a dictionary with key `'is_falsified'` (boolean), not `'falsified'`
- Using `.get('falsified', False)` returns the default `False` for ALL 80 pairs
- This produces FR = 0.0% and std = 0.0% regardless of actual results

---

## EVIDENCE

1. **Standard deviation = 0.0 is impossible** unless all values are identical
   - Exp 6.1: std = 28.71% (realistic)
   - Exp 6.4: std = 0.0% (bug indicator)

2. **Code inspection confirms key mismatch**
   - Function returns `'is_falsified'` and `'falsification_rate'`
   - Exp 6.4 looks for `'falsified'` (doesn't exist)

3. **Face pair generation is identical** in both experiments
   - Same `load_lfw_pairs_sklearn()` function
   - Same seed (42)
   - Same LFW dataset

---

## RECOMMENDATION

**Fix the bug and re-run Experiment 6.4** (Option A)

**Why:**
- Fix is trivial (2-line code change)
- Re-run takes only 45 minutes GPU time
- Restores model-agnostic validation for RQ4
- Demonstrates rigor (found bug, fixed it, validated)
- Theorem 3.5 remains valid either way (this bug doesn't affect it)

**Total time investment:** ~1 hour (code fix + re-run + documentation)

---

## WHAT TO DO NEXT

1. **Edit file:** `experiments/run_real_experiment_6_4.py`
   - Line 368: Change `'falsified'` → `'falsification_rate'`
   - Line 369: Change to `pair_frs.append(falsification_rate)`

2. **Re-run:**
   ```bash
   python experiments/run_real_experiment_6_4.py --n_pairs 500 --device cuda --seed 42
   ```

3. **Verify:** Check that std > 0 and FR is in range [0%, 20%]

4. **Update docs:** Note the bug fix in dissertation

---

## IMPACT ON DISSERTATION

**Theorem 3.5 Validation:** ✅ UNAFFECTED
- Already validated by Exp 6.1 alone (p < 10^-112)
- This bug affects RQ4 (model-agnostic), not RQ1 (falsifiability)

**Timeline:** ✅ MINIMAL IMPACT
- 1 hour to fix and re-run
- Well within 1-2 week deadline

**Defense Readiness:** ✅ IMPROVED
- Shows you found and fixed a bug (demonstrates rigor)
- Turns potential weakness into strength

---

## FULL DETAILS

See: `/home/aaron/projects/xai/AGENT_2_REPRODUCIBILITY_FINDINGS.md`

---

**BOTTOM LINE:** This is a simple bug with a simple fix. Re-running will take 1 hour and significantly strengthen your dissertation's model-agnostic validation.
