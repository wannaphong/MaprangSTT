#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv

echo ""
echo "=== Tmux Sessions ==="
tmux list-sessions 2>/dev/null || echo "No active tmux sessions"

echo ""
echo "=== Training Process ==="
ps aux | grep -v grep | grep -E "cactus whistle finetune" || echo "No training process active"

echo ""
echo "=== Tmux Pane Output (session: whistle) ==="
tmux capture-pane -pt whistle -S -30 2>/dev/null || echo "No tmux pane found"
