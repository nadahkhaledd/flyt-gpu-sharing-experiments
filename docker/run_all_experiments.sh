#!/bin/bash
set -e
LOGS=~/flyt-experiments/matmul/logs
mkdir -p $LOGS

echo "=== Single container, full GPU ==="
echo "Start server: ./start_server.sh 1 142"
echo "Press Enter when server is ready..."
read
docker run --rm --network=host -e CRICKET_SERVER="localhost,1" \
    flyt-matmul:v1 | tee $LOGS/docker_flyt_full_gpu.log

echo ""
echo "=== Concurrent 50-50 split ==="
echo "Stop server. Start: ./start_server.sh 1 71 and ./start_server.sh 2 71"
echo "Press Enter when both servers are ready..."
read
docker run --rm --network=host -e CRICKET_SERVER="localhost,1" \
    flyt-matmul:v1 > $LOGS/docker_flyt_50_50_job1.log 2>&1 &
docker run --rm --network=host -e CRICKET_SERVER="localhost,2" \
    flyt-matmul:v1 > $LOGS/docker_flyt_50_50_job2.log 2>&1 &
wait
echo "Job 1:"; grep "Avg GFLOPS\|Verification" $LOGS/docker_flyt_50_50_job1.log
echo "Job 2:"; grep "Avg GFLOPS\|Verification" $LOGS/docker_flyt_50_50_job2.log

echo ""
echo "=== Concurrent 30-70 split ==="
echo "Stop servers. Start: ./start_server.sh 1 43 and ./start_server.sh 2 99"
echo "Press Enter when both servers are ready..."
read
docker run --rm --network=host -e CRICKET_SERVER="localhost,1" \
    flyt-matmul:v1 > $LOGS/docker_flyt_30_70_job1.log 2>&1 &
docker run --rm --network=host -e CRICKET_SERVER="localhost,2" \
    flyt-matmul:v1 > $LOGS/docker_flyt_30_70_job2.log 2>&1 &
wait
echo "Job 1 (30%):"; grep "Avg GFLOPS\|Verification" $LOGS/docker_flyt_30_70_job1.log
echo "Job 2 (70%):"; grep "Avg GFLOPS\|Verification" $LOGS/docker_flyt_30_70_job2.log

echo ""
echo "All experiments complete. Logs in $LOGS"
