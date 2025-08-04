#!/bin/bash

# ========== CONFIGURATION ==========
# Your HPC login details
HPC_USER="moncek"
HPC_HOST="r3d3.karlin.mff.cuni.cz"   # e.g., hpc.example.edu
HPC_PATH="~/Documents"

# Local project path (default: current directory)
LOCAL_PATH="$(pwd)"

# ===================================
# Usage: ./sync-to-hpc.sh [--pull]
# Default: push local → HPC
# --pull: pull HPC → local

# Files & folders to exclude from syncing
EXCLUDES=(
    --exclude ".git/"
    --exclude ".DS_Store"
    --exclude "*.jl.cov"
    --exclude "*.jl.*.cov"
    --exclude "*.jl.mem"
    --exclude "*.o"
    --exclude "*.so"
    --exclude "*.dylib"
    --exclude "Manifest.toml"   # If you want HPC to resolve deps itself
    --exclude "deps/"
)

# Rsync command
RSYNC_CMD="rsync -avz --delete ${EXCLUDES[@]}"

# Push (local → HPC)
push_to_hpc() {
    echo "Syncing LOCAL → HPC..."
    $RSYNC_CMD "$LOCAL_PATH/" "$HPC_USER@$HPC_HOST:$HPC_PATH/"
}

# Pull (HPC → local)
pull_from_hpc() {
    echo "Syncing HPC → LOCAL..."
    $RSYNC_CMD "$HPC_USER@$HPC_HOST:$HPC_PATH/" "$LOCAL_PATH/"
}

# Main logic
if [[ "$1" == "--pull" ]]; then
    pull_from_hpc
else
    push_to_hpc
fi
