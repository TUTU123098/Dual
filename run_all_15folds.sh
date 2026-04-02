#!/usr/bin/env bash

DATASETS=("AOPP" "AnOxPP" "AnOxPePred")

for ds in "${DATASETS[@]}"; do
    for fold in {0..4}; do
        echo "====================================="
        echo "开始训练: $ds  fold $fold"
        echo "====================================="
        
        python main.py --dataset "$ds" --fold "$fold" > "outlog/${ds}_fold${fold}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$ds fold $fold] 训练完成"
        else
            echo "[$ds fold $fold] 出现错误，请查看 outlog/${ds}_fold${fold}.log"
        fi
    done
done

echo "所有 15 个五折训练已完成！"
