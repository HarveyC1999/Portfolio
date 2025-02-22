# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:37:23 2025

@author: HarveyC
"""
## 合併多個.safetensors
from safetensors.torch import save_file, load_file
import torch


def combine_safetensors_files(file_paths, output_file):
    combined_tensors = {}

    for file_path in file_paths:
        # 加載 safetensors 文件
        tensors = load_file(file_path)
        # combined_tensors 
        for key, tensor in tensors.items():
            if key in combined_tensors:
                combined_tensors[key] = torch.cat((combined_tensors[key], tensor), dim=0)  # 按第0维拼接
            else:
                combined_tensors[key] = tensor

    # 保存
    save_file(combined_tensors, output_file)


# 待合併列表
safetensors_files = [
    'model-00001-of-00004.safetensors',
    'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors',
    'model-00004-of-00004.safetensors'
]

# 输出safetensors 文件
combine_safetensors_files(safetensors_files, 'combined_model.safetensors')

## 轉.bin
import torch
from safetensors.torch import load_file

# 加載 safetensors 文件
def load_safetensors_to_pytorch(file_path):
    return load_file(file_path)

# 保存為 .bin 格式
def save_as_bin(tensors, bin_file_path):
    torch.save(tensors, bin_file_path)

safetensors_file = 'combined_model.safetensors' # 轉換前path
bin_file_path = 'model.bin' # 轉換後path
tensors = load_safetensors_to_pytorch(safetensors_file)
save_as_bin(tensors, bin_file_path)