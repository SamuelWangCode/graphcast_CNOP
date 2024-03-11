#!/bin/bash

while :
do
    # 检查所有GPU，找出超过19GB可用内存的GPU
    available_gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{if ($1 > 19000) print NR-1}')
    
    # 如果找到了，立即提交作业
    if [ ! -z "$available_gpu" ]; then
        # 假设您的作业提交脚本是submit_job.sh
        echo "发现可用的GPU：$available_gpu, 提交作业..."
        export CUDA_VISIBLE_DEVICES=$available_gpu
        nohup ./run_my_program.sh &
        echo "程序已提交！"
        break
    else
        echo "没有找到可用的GPU。"
        sleep 5
    fi
done