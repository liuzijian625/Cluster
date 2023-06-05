#!/bin/bash

# 定义一个数组，包含所有要修改的值
normalize_methods=("min-max" "z-score")
decomposition_methods=("PCA" "t-SNE" "UMAP")

sed -i "s/is_normalize.*/is_normalize: False/" config/config.yaml

for decomposition_method in "${decomposition_methods[@]}"
do
    sed -i "s/decomposition_method.*/decomposition_method: \"$decomposition_method\"/" config/config.yaml
    python main.py
done

sed -i "s/is_normalize.*/is_normalize: True/" config/config.yaml

# 使用循环来分别修改 normalize_method 的值
for normalize_method in "${normalize_methods[@]}"
do
    for decomposition_method in "${decomposition_methods[@]}"
    do
        sed -i "s/normalize_method.*/normalize_method: \"$normalize_method\"/" config/config.yaml
        sed -i "s/decomposition_method.*/decomposition_method: \"$decomposition_method\"/" config/config.yaml
        python main.py
    done
done