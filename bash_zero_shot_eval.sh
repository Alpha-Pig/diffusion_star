#!/bin/bash

# 激活环境
source /data_train/yeqigao/miniconda3/bin/activate star_1

workspaceFolder=$(pwd)
echo "workspaceFolder: $workspaceFolder"

# 获取当前工作目录
workspaceFolder=$(pwd)

# 检查是否传递了 CHECKPOINT 参数
if [ -z "$1" ]; then
  echo "ERROR: CHECKPOINT path is required!"
  echo "Usage: $0 <checkpoint-path>"
  exit 1
fi

CHECKPOINT=$1  # 使用传入的 CHECKPOINT 路径

# 获取倒数第一层级（epoch）
EPOCH=$(basename "$CHECKPOINT")

# 获取倒数第三层级（project_title）
PROJECT_TITLE_BASE=$(basename $(dirname $(dirname "$CHECKPOINT")))

# 将 epoch 拼接到 project_title 中
PROJECT_TITLE="${PROJECT_TITLE_BASE}_${EPOCH}"

# 打印目录层级以验证
echo "Epoch: $EPOCH"
echo "Project Title: $PROJECT_TITLE"

SCRIPT_PATH="${workspaceFolder}/zero-shotcot-eval.py"
BATCH_IDX=0
DEVICE_BATCH_SIZE=8
MAX_IDX=128
N_VOTES=8
TEMP=0.9
START_FINAL_ANSWER_IDX=500
ANSWER_LENGTH=12
ROOT_PREFIX="/data_train/yeqigao/code/llms_factory/"
FINAL_ANSWER_TEXT="Therefore, the answer (arabic numerals) is"
ZERO_SHOT_COT_PROMPT="A: Let's think step by step."
N_AHEAD=8

# 运行分布式评估脚本
python \
    $SCRIPT_PATH \
    --batch_idx $BATCH_IDX \
    --device_batch_size $DEVICE_BATCH_SIZE \
    --max_idx $MAX_IDX \
    --n_votes $N_VOTES \
    --temp $TEMP \
    --start_final_answer_idx $START_FINAL_ANSWER_IDX \
    --answer_length $ANSWER_LENGTH \
    --root_prefix $ROOT_PREFIX \
    --checkpoint $CHECKPOINT \
    --project_title $PROJECT_TITLE \
    --final_answer_text "$FINAL_ANSWER_TEXT" \
    --zero_shot_cot_prompt "$ZERO_SHOT_COT_PROMPT" \
    --n_ahead $N_AHEAD
