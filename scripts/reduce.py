import os
import torch
import torch.nn.functional as F
from glob import glob

vector_dir = "../outputs/vectors"
vector_files = glob(os.path.join(vector_dir, "*.pt"))

# 读取向量文件
file_names =[os.path.basename(file) for file in vector_files]
raw_vectors = [torch.load(file, map_location="cpu") for file in vector_files]
# 将向量展平为一维
falttened_vectors = [vec.view(-1) for vec in raw_vectors]

similarity_threshold = 0.85
unique_pool = []
unique_files = []

# 逐个比较向量，进行去重
for i, current_vec in enumerate(falttened_vectors):
    is_duplicate = False
    for j, unique_vec in enumerate(unique_pool):
        sim = F.cosine_similarity(current_vec.unsqueeze(0), unique_vec.unsqueeze(0)).item()
        if sim > similarity_threshold:
            is_duplicate = True
            break
    if not is_duplicate:
        unique_pool.append(current_vec)
        unique_files.append(file_names[i])

print("Unique Vectors:")
for file in unique_files:
    print(file)