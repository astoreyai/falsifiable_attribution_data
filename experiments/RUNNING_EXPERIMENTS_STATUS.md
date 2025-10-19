# Running Experiments Status Report
**Generated:** 2025-10-19 02:10 UTC
**GPU:** CUDA enabled

---

## Currently Running Experiments

### Experiment 6.5: Convergence Analysis (RUNNING SINCE 00:33)
- **PID:** 1849191
- **Status:** Running (98+ minutes)
- **Command:** 
  ```bash
  python experiments/run_real_experiment_6_5.py \
    --n_inits 5000 \
    --max_iters 100 \
    --n_bootstrap 100 \
    --device cuda \
    --save_dir experiments/production_exp6_5_20251019_003318 \
    --seed 42
  ```
- **Output Log:** `/home/aaron/projects/xai/experiments/exp6_5_output.log`
- **CPU Usage:** 134%
- **Memory Usage:** 2.1%
- **Expected Completion:** ~3-4 hours (5000 optimization runs with bootstrapping)
- **What It Tests:** RQ5 - Convergence speed of counterfactual generation algorithm

### Experiment 6.3: Attribute-Based Falsifiability (LAUNCHED 02:07)
- **PID:** 1863425
- **Status:** Running - Detecting facial attributes (32% complete)
- **Command:**
  ```bash
  python experiments/run_real_experiment_6_3.py \
    --n_samples 300 \
    --K 100 \
    --device cuda \
    --output_dir experiments/production_exp6_3_20251019_020730 \
    --seed 42
  ```
- **Output Log:** `/home/aaron/projects/xai/experiments/exp6_3_output.log`
- **Output Dir:** `/home/aaron/projects/xai/experiments/production_exp6_3_20251019_020730/`
- **CPU Usage:** 2490% (multi-threaded InsightFace attribute detection)
- **Memory Usage:** 1.3%
- **Current Progress:** Detecting attributes 97/300 faces (~32%)
- **Expected Completion:** 
  - Attribute detection: ~20-30 minutes (2-3 it/s)
  - Falsification testing: ~60-90 minutes (300 samples × 100 counterfactuals each)
  - **TOTAL:** ~2-2.5 hours from launch
- **What It Tests:** RQ3 - Which facial attributes (occlusion/geometric/expression) are most falsifiable

### Experiment 6.4: Model-Agnostic Testing (LAUNCHED 02:07)
- **PID:** 1872348
- **Status:** Running - Processing FaceNet model
- **Command:**
  ```bash
  python experiments/run_real_experiment_6_4.py \
    --n_pairs 500 \
    --K 100 \
    --device cuda \
    --output_dir experiments/production_exp6_4_20251019_020744 \
    --seed 42
  ```
- **Output Log:** `/home/aaron/projects/xai/experiments/exp6_4_output.log`
- **Output Dir:** `/home/aaron/projects/xai/experiments/production_exp6_4_20251019_020744/`
- **CPU Usage:** 173%
- **Memory Usage:** 1.6%
- **Models to Test:** FaceNet, ResNet-50, MobileNetV2
- **Attribution Methods:** Grad-CAM, SHAP
- **Total Tests:** 500 pairs × 3 models × 2 methods = 3000 attributions
- **Expected Completion:** 
  - Per pair: ~5 seconds
  - Total: 3000 × 5s = 15,000s = ~4.2 hours
  - **TOTAL:** ~4-5 hours from launch
- **What It Tests:** RQ4 - Does falsifiability generalize across model architectures
- **Note:** Some pairs showing uniform attribution warnings (low contrast) - expected for some face pairs

---

## Monitoring Commands

### Check Process Status
```bash
ps aux | grep -E "python.*run_real_experiment" | grep -v grep
```

### View Live Logs (Tail)
```bash
# Experiment 6.3
tail -f /home/aaron/projects/xai/experiments/exp6_3_output.log

# Experiment 6.4
tail -f /home/aaron/projects/xai/experiments/exp6_4_output.log

# Experiment 6.5
tail -f /home/aaron/projects/xai/experiments/exp6_5_output.log
```

### View Last 50 Lines
```bash
# Experiment 6.3
tail -50 /home/aaron/projects/xai/experiments/exp6_3_output.log

# Experiment 6.4
tail -50 /home/aaron/projects/xai/experiments/exp6_4_output.log

# Experiment 6.5
tail -50 /home/aaron/projects/xai/experiments/exp6_5_output.log
```

### Check Results Files
```bash
# Experiment 6.3 results
ls -lh /home/aaron/projects/xai/experiments/production_exp6_3_20251019_020730/
cat /home/aaron/projects/xai/experiments/production_exp6_3_20251019_020730/exp6_3_n300_*/results.json

# Experiment 6.4 results
ls -lh /home/aaron/projects/xai/experiments/production_exp6_4_20251019_020744/
cat /home/aaron/projects/xai/experiments/production_exp6_4_20251019_020744/exp6_4_n500_*/results.json

# Experiment 6.5 results
ls -lh /home/aaron/projects/xai/experiments/production_exp6_5_20251019_003318/
cat /home/aaron/projects/xai/experiments/production_exp6_5_20251019_003318/results_*.json
```

### Check GPU Usage
```bash
nvidia-smi
```

### Check Disk Space
```bash
df -h /home/aaron/projects/xai/experiments/
```

---

## Expected Completion Timeline

| Experiment | Started | Est. Completion | Runtime | Status |
|------------|---------|-----------------|---------|--------|
| Exp 6.5 (Convergence) | 00:33 | ~04:00-04:30 | 3.5-4h | Running |
| Exp 6.3 (Attributes) | 02:07 | ~04:30-04:45 | 2-2.5h | Running |
| Exp 6.4 (Model-Agnostic) | 02:07 | ~06:00-07:00 | 4-5h | Running |

**All experiments should complete by ~07:00 UTC (within 5 hours from now)**

---

## Success Criteria

### Experiment 6.3
- ✅ Detect attributes for 300 faces using InsightFace
- ✅ Compute falsification rates per attribute
- ✅ Statistical tests (ANOVA across attributes, t-test occlusion vs geometric)
- ✅ Results saved to `results.json`
- Expected: Occlusion attributes (beard, mustache, glasses) more falsifiable than geometric

### Experiment 6.4
- ✅ Test 500 face pairs across 3 models
- ✅ Compute falsification rates per model × attribution method
- ✅ Paired t-tests between models
- ✅ ANOVA across all 3 models
- ✅ Results saved to `results.json`
- Expected: No significant difference (model-agnostic property)

### Experiment 6.5 (Already Running)
- ✅ 5000 optimization runs with different initializations
- ✅ Track convergence speed and success rate
- ✅ Bootstrap confidence intervals
- ✅ Results saved to `results_*.json`
- Expected: Fast convergence (<20 iterations) with >90% success rate

---

## Troubleshooting

### If an experiment crashes:
1. Check log file for error messages
2. Check GPU memory: `nvidia-smi`
3. Check disk space: `df -h`
4. Restart with same command from "Currently Running Experiments" section above

### If GPU out of memory:
- Reduce batch size or number of counterfactuals (K)
- Kill other processes using GPU

### If results seem incomplete:
- Check log file for errors during processing
- Verify all stages completed (loading data, processing, statistical tests, saving results)

---

## Contact

For issues or questions, check:
- Log files in `/home/aaron/projects/xai/experiments/`
- Experiment scripts in `/home/aaron/projects/xai/experiments/run_real_experiment_6_*.py`
- Main project documentation in `/home/aaron/projects/xai/PHD_PIPELINE/`

