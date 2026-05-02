#!/bin/bash
# Start a Cricket RPC server on the host.
# Usage: ./start_server.sh <rpc_version> <sm_count>
# Example: ./start_server.sh 1 142   (server #1 with all SMs)
# Example: ./start_server.sh 2 71    (server #2 with 71 SMs)

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <rpc_version> <sm_count>"
    exit 1
fi

cd ~/flyt
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_khab/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_khab/log
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
LD_LIBRARY_PATH=bin-server:bin-server/libs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    bin-server/cricket-rpc-server "$1" 0 "$2" 4294967296
