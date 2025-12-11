#!/bin/bash

# Usage: ./launch_parallel.sh [NUM_DIVISIONS] [GPU_ID]
# Example: ./launch_parallel.sh 4 1

NUM_DIVS=${1:-4}    # Default to 4 divisions
GPU_ID=${2:-0}      # Default to GPU 0
SESSION_NAME="pretok_gpu${GPU_ID}"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION_NAME already exists. Attach using: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Launching $NUM_DIVS workers on GPU $GPU_ID in tmux session '$SESSION_NAME'..."

# Create a new detached session with the first window named 'worker_0'
# We use windows (tabs) instead of panes to avoid "no space" errors completely.
tmux new-session -d -s $SESSION_NAME -n "worker_0"

# Loop to create windows and run commands
for ((i=0; i<NUM_DIVS; i++)); do
    # If not the first iteration, create a new window
    if [ $i -gt 0 ]; then
        tmux new-window -t $SESSION_NAME -n "worker_$i"
    fi

    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python pretokenize_split.py --division $i --num_divisions $NUM_DIVS"
    
    # Send command to the specific window
    tmux send-keys -t "${SESSION_NAME}:worker_${i}" "$CMD" C-m
    
    echo "  Started Worker $i (Window: worker_$i)"
done

echo "All workers started."
echo "To view: tmux attach -t $SESSION_NAME"
echo "To switch windows inside tmux: Ctrl+B then <number> (0-9)"
echo "To detach and leave running: Ctrl+B then d"

# Automatically attach
tmux attach-session -t $SESSION_NAME