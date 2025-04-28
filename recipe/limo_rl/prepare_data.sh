#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")"

# 创建数据目录（如果不存在）
mkdir -p ../../../data/limo_rl
mkdir -p ../../../data/limo_rl/skywork_or1
mkdir -p ../../../data/limo_rl/math500
mkdir -p ../../../data/limo_rl/aime24

# 运行数据准备脚本
echo "开始准备数据..."
echo "运行 skywork_or1.py..."
python src/data/skywork_or1.py --local_dir ../../../data/limo_rl/skywork_or1

echo "运行 test_math500.py..."
python src/data/test_math500.py --local_dir ../../../data/limo_rl/math500

echo "运行 test_aime24.py..."
python src/data/test_aime24.py --local_dir ../../../data/limo_rl/aime24

echo "数据准备完成！"
