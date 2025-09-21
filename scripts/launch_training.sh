#!/bin/bash

# Launch script for multi-GPU distributed training
# Usage: ./launch_training.sh [num_gpus] [strategy] [batch_size]

NUM_GPUS=${1:-8}
STRATEGY=${2:-ddp}
BATCH_SIZE=${3:-32}
ITERATIONS=${4:-100}

echo "Launching distributed training with:"
echo "  GPUs: $NUM_GPUS"
echo "  Strategy: $STRATEGY"
echo "  Batch Size: $BATCH_SIZE"
echo "  Iterations: $ITERATIONS"

# Single node training
if [ $NUM_GPUS -le 8 ]; then
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        distributed_training.py \
        --strategy=$STRATEGY \
        --batch-size=$BATCH_SIZE \
        --iterations=$ITERATIONS \
        --mixed-precision
else
    # Multi-node training
    NUM_NODES=$((($NUM_GPUS + 7) / 8))
    GPUS_PER_NODE=$((NUM_GPUS < 8 ? NUM_GPUS : 8))
    
    echo "Multi-node training: $NUM_NODES nodes, $GPUS_PER_NODE GPUs per node"
    
    # Note: For multi-node, you need to set MASTER_ADDR to the IP of node 0
    # and run this script on each node with appropriate --node_rank
    
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=${NODE_RANK:-0} \
        --master_addr=${MASTER_ADDR:-localhost} \
        --master_port=29500 \
        distributed_training.py \
        --strategy=$STRATEGY \
        --batch-size=$BATCH_SIZE \
        --iterations=$ITERATIONS \
        --mixed-precision
fi
