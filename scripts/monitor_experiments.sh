#!/bin/bash
# Experiment Monitoring Script
# Monitors the 4 running dissertation validation experiments

echo "================================================================================"
echo "  DISSERTATION VALIDATION EXPERIMENTS - STATUS MONITOR"
echo "================================================================================"
echo ""
echo "Last updated: $(date)"
echo ""

# Experiment PIDs (update these if restarted)
EXP_6_2_PID=1848475
EXP_6_3_PID=1849124
EXP_6_4_PID=1848607
EXP_6_5_PID=1849191

echo "=== PROCESS STATUS ==="
echo ""
echo "Experiment 6.2 (Margin Analysis, n=500):"
if ps -p $EXP_6_2_PID > /dev/null 2>&1; then
    ps aux | grep $EXP_6_2_PID | grep -v grep | awk '{print "  ✅ RUNNING - PID:", $2, "CPU:", $3"%, MEM:", $4"%"}'
else
    echo "  ❌ NOT RUNNING"
fi

echo ""
echo "Experiment 6.3 (Attribute Falsifiability, n=300):"
if ps -p $EXP_6_3_PID > /dev/null 2>&1; then
    ps aux | grep $EXP_6_3_PID | grep -v grep | awk '{print "  ✅ RUNNING - PID:", $2, "CPU:", $3"%, MEM:", $4"%"}'
else
    echo "  ❌ NOT RUNNING"
fi

echo ""
echo "Experiment 6.4 (Model-Agnostic, n=500):"
if ps -p $EXP_6_4_PID > /dev/null 2>&1; then
    ps aux | grep $EXP_6_4_PID | grep -v grep | awk '{print "  ✅ RUNNING - PID:", $2, "CPU:", $3"%, MEM:", $4"%"}'
else
    echo "  ❌ NOT RUNNING"
fi

echo ""
echo "Experiment 6.5 (Convergence, 5000 trials):"
if ps -p $EXP_6_5_PID > /dev/null 2>&1; then
    ps aux | grep $EXP_6_5_PID | grep -v grep | awk '{print "  ✅ RUNNING - PID:", $2, "CPU:", $3"%, MEM:", $4"%"}'
else
    echo "  ❌ NOT RUNNING"
fi

echo ""
echo "=== LOG FILE SIZES ==="
ls -lh logs/exp6_*.log 2>/dev/null | awk '{print "  ", $9, "-", $5}'

echo ""
echo "=== RECENT LOG ACTIVITY ==="
echo ""
echo "Exp 6.2 (last 3 lines):"
tail -3 logs/exp6_2_n500.log 2>/dev/null | sed 's/^/  /'

echo ""
echo "Exp 6.3 (last 3 lines):"
tail -3 logs/exp6_3_n300.log 2>/dev/null | sed 's/^/  /'

echo ""
echo "Exp 6.4 (last 3 lines):"
tail -3 logs/exp6_4_n500.log 2>/dev/null | sed 's/^/  /'

echo ""
echo "Exp 6.5 (last 3 lines):"
tail -3 logs/exp6_5_convergence.log 2>/dev/null | sed 's/^/  /'

echo ""
echo "=== OUTPUT DIRECTORIES ==="
echo ""
echo "Results files found:"
find experiments/production_* -name "results.json" -o -name "*.json" 2>/dev/null | head -20 | sed 's/^/  /'

echo ""
echo "=== GPU USAGE ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "  nvidia-smi not available"
fi

echo ""
echo "=== ESTIMATED COMPLETION TIMES ==="
echo ""
echo "  Exp 6.2 (n=500):    ~2-4 hours from start (margin stratification)"
echo "  Exp 6.3 (n=300):    ~3-6 hours from start (attribute detection + tests)"
echo "  Exp 6.4 (n=500):    ~4-8 hours from start (3 models: FaceNet, VGG, ResNet)"
echo "  Exp 6.5 (5000):     ~4-6 hours from start (5000 REAL optimizations)"
echo ""
echo "  Expected total runtime: 10-15 hours (can run overnight)"
echo ""
echo "================================================================================"
