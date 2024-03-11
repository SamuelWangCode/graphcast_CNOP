#!/bin/bash

# 定义日志文件路径
LOG_DIR="./CNOP/logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

# 执行generate_perturbation.py并将输出写入日志
echo "Running generate_perturbation.py..."
python ./CNOP/generate_perturbation.py > "$LOG_DIR/generate_perturbation.log" 2>&1

# 确保上一个程序执行成功再执行下一个
if [ $? -eq 0 ]; then
    echo "Running generate_sample.py..."
    python ./CNOP/generate_sample.py > "$LOG_DIR/generate_sample.log" 2>&1
else
    echo "generate_perturbation.py failed, exiting."
    exit 1
fi

# 同样地，检查generate_sample.py的执行结果
if [ $? -eq 0 ]; then
    echo "Running normalization.py..."
    python ./CNOP/normalization.py > "$LOG_DIR/normalization.log" 2>&1
else
    echo "generate_sample.py failed, exiting."
    exit 1
fi

echo "All programs have run successfully."

