# Dissertation Validation Experiments - Launch Report

**Launch Date:** October 19, 2025, 12:32 AM CDT
**Status:** ALL 4 EXPERIMENTS RUNNING SUCCESSFULLY
**Total Expected Runtime:** 10-15 hours (can run overnight)

---

## EXPERIMENTS LAUNCHED

### Experiment 6.2: Margin Analysis (n=500)
**Hypothesis:** Falsifiability varies by decision margin width
**Status:** RUNNING
**PID:** 1848475
**Command:**
```bash
/home/aaron/projects/xai/venv/bin/python experiments/run_real_experiment_6_2.py \
  --n_pairs 500 \
  --device cuda \
  --output_dir experiments/production_n500_exp6_2_20251019_003231 \
  --seed 42
```
**Log File:** `/home/aaron/projects/xai/logs/exp6_2_n500.log`
**Output Dir:** `/home/aaron/projects/xai/experiments/production_n500_exp6_2_20251019_003231/`
**Estimated Completion:** ~2-4 hours from start
**Current Progress:** Processing Stratum 2 (Moderate) - 125 pairs
**CPU Usage:** 98.3% | **Memory:** 1.4%

---

### Experiment 6.3: Attribute Falsifiability (n=300)
**Hypothesis:** Falsifiability varies by facial attributes
**Status:** RUNNING
**PID:** 1849124
**Command:**
```bash
/home/aaron/projects/xai/venv/bin/python experiments/run_real_experiment_6_3.py \
  --n_samples 300 \
  --K 100 \
  --device cuda \
  --output_dir experiments/production_n300_exp6_3_20251019_003236 \
  --seed 42
```
**Log File:** `/home/aaron/projects/xai/logs/exp6_3_n300.log`
**Output Dir:** `/home/aaron/projects/xai/experiments/production_n300_exp6_3_20251019_003236/`
**Estimated Completion:** ~3-6 hours from start
**Current Progress:** Detecting attributes (129/300 complete, ~43%)
**CPU Usage:** 2560% (multi-threaded) | **Memory:** 1.4%

---

### Experiment 6.4: Model-Agnostic Testing (n=500)
**Hypothesis:** Falsifiability holds across different architectures
**Status:** RUNNING
**PID:** 1848607
**Command:**
```bash
/home/aaron/projects/xai/venv/bin/python experiments/run_real_experiment_6_4.py \
  --n_pairs 500 \
  --device cuda \
  --output_dir experiments/production_n500_exp6_4_20251019_003233 \
  --seed 42
```
**Log File:** `/home/aaron/projects/xai/logs/exp6_4_n500.log`
**Output Dir:** `/home/aaron/projects/xai/experiments/production_n500_exp6_4_20251019_003233/`
**Estimated Completion:** ~4-8 hours from start
**Current Progress:** Testing FaceNet (27/500 pairs, ~5%)
**CPU Usage:** 180% | **Memory:** 1.9%
**Note:** Will test 3 models: FaceNet, VGG-Face, ResNet-50

---

### Experiment 6.5: Convergence & Sample Size (5000 trials)
**Hypothesis:** REAL algorithm converges reliably
**Status:** RUNNING
**PID:** 1849191
**Command:**
```bash
/home/aaron/projects/xai/venv/bin/python experiments/run_real_experiment_6_5.py \
  --n_inits 5000 \
  --max_iters 100 \
  --n_bootstrap 100 \
  --device cuda \
  --save_dir experiments/production_exp6_5_20251019_003234 \
  --seed 42
```
**Log File:** `/home/aaron/projects/xai/logs/exp6_5_convergence.log`
**Output Dir:** `/home/aaron/projects/xai/experiments/production_exp6_5_20251019_003234/`
**Estimated Completion:** ~4-6 hours from start
**Current Progress:** Running REAL optimizations (11/5000 trials, ~0.2%)
**CPU Usage:** 103% | **Memory:** 2.0%
**Note:** Each trial runs 100 iterations max, convergence threshold = 0.01

---

## SYSTEM RESOURCES

**GPU:** NVIDIA GeForce RTX 3090
- **Temperature:** 53°C
- **GPU Utilization:** 93%
- **Memory Used:** 5658 MiB / 24576 MiB (23%)

**Status:** All experiments using CUDA efficiently, no resource contention

---

## MONITORING INSTRUCTIONS

### Manual Monitoring Script
Run the monitoring script anytime to check status:
```bash
/home/aaron/projects/xai/scripts/monitor_experiments.sh
```

### Watch Mode (auto-refresh every 60 seconds)
```bash
watch -n 60 /home/aaron/projects/xai/scripts/monitor_experiments.sh
```

### Check Individual Logs
```bash
# Experiment 6.2
tail -f /home/aaron/projects/xai/logs/exp6_2_n500.log

# Experiment 6.3
tail -f /home/aaron/projects/xai/logs/exp6_3_n300.log

# Experiment 6.4
tail -f /home/aaron/projects/xai/logs/exp6_4_n500.log

# Experiment 6.5
tail -f /home/aaron/projects/xai/logs/exp6_5_convergence.log
```

### Check Process Status
```bash
ps aux | grep "run_real_experiment_6_[2-5]" | grep -v grep
```

### GPU Monitoring
```bash
nvidia-smi
# Or watch mode:
watch -n 10 nvidia-smi
```

---

## EXPECTED RESULTS

### Experiment 6.2 Output
**Files:**
- `results.json` - FR rates per margin stratum
- `margin_stratification_plot.png` - Visual analysis
- `summary.txt` - Statistical summary

**Key Metrics:**
- Falsifiability rate per stratum (Narrow/Moderate/Wide/Very Wide)
- Trend analysis (should decrease with wider margins)
- Statistical significance tests

---

### Experiment 6.3 Output
**Files:**
- `results.json` - FR rates by facial attributes
- `attribute_analysis_plot.png` - Comparison across attributes
- `summary.txt` - Attribute breakdown

**Key Metrics:**
- Falsifiability rate for: Age, Gender, Ethnicity, Occlusion
- Variation analysis
- Correlation with attribute confidence

---

### Experiment 6.4 Output
**Files:**
- `results.json` - FR rates per model
- `model_comparison_plot.png` - Cross-model analysis
- `summary.txt` - Model-agnostic validation

**Key Metrics:**
- Falsifiability rates for:
  - FaceNet (primary)
  - VGG-Face
  - ResNet-50
- Consistency across architectures
- Statistical significance

---

### Experiment 6.5 Output
**Files:**
- `convergence_analysis.json` - Convergence statistics
- `convergence_plot.png` - Convergence curves
- `sample_size_analysis.json` - Bootstrap results
- `summary.txt` - Statistical summary

**Key Metrics:**
- Convergence rate (% trials that converge)
- Mean/median iterations to convergence
- Sample size stability analysis
- Bootstrap confidence intervals

---

## TROUBLESHOOTING

### If an experiment crashes:

1. **Check the log file** for error messages
2. **Check GPU memory:** `nvidia-smi`
3. **Restart individual experiment:**

```bash
# Example: Restart Experiment 6.2
nohup /home/aaron/projects/xai/venv/bin/python \
  experiments/run_real_experiment_6_2.py \
  --n_pairs 500 \
  --device cuda \
  --output_dir experiments/production_n500_exp6_2_RESTART_$(date +%Y%m%d_%H%M%S) \
  --seed 42 \
  > logs/exp6_2_n500_restart.log 2>&1 &
```

### Known Issues

**Experiment 6.4 Warnings:**
- Some pairs may fail with "No high-attribution pixels found"
- This is expected for uniform attribution maps
- Experiment continues processing remaining pairs
- Final results will report valid pair count

**GPU Memory:**
- All experiments fit in 6GB GPU RAM
- 3090 (24GB) has plenty of headroom
- If memory issues occur, restart individual experiments

---

## COMPLETION CHECKLIST

When all experiments finish, verify:

- [ ] All 4 `results.json` files exist
- [ ] No errors in log files (grep for "ERROR" or "FAILED")
- [ ] All plots generated successfully
- [ ] Results are statistically significant
- [ ] Ready to compile into dissertation Chapter 6

---

## NEXT STEPS (After Completion)

1. **Collect all results:**
   ```bash
   find experiments/production_* -name "results.json" -type f
   ```

2. **Review plots:**
   ```bash
   find experiments/production_* -name "*.png" -type f
   ```

3. **Compile statistics** into dissertation tables

4. **Update Chapter 6** with empirical results

5. **Run final validation check:**
   ```bash
   python experiments/validate_all_results.py
   ```

---

## LAUNCH SUMMARY

LAUNCH SUCCESS RATE: **4/4 (100%)**

All validation experiments are running successfully and will provide comprehensive empirical evidence for:
- **H6a-H6b:** Margin-dependent falsifiability
- **H6c-H6d:** Attribute-specific variation
- **H6e:** Model-agnostic consistency
- **H5a:** Convergence reliability

**Estimated total completion time:** 10-15 hours

**Launch completed:** October 19, 2025, 12:34 AM CDT
**Expected finish:** October 19, 2025, 10:00 AM - 3:00 PM CDT

---

**Status: ALL SYSTEMS NOMINAL - RUNNING OVERNIGHT**
