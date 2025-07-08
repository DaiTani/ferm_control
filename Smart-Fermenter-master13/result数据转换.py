import numpy as np
import pandas as pd

# 加载 .npz 文件
data = np.load('data.npz')

# 打印 .npz 文件中的所有键
print("Keys in .npz file:", data.files)

# 创建一个空的字典，用于存储所有数据
data_dict = {}

# 遍历 .npz 文件中的每个数组
for key in data.files:
    array = data[key]
    print(f"Processing {key} with shape {array.shape}")

    # 检查数组是否为空
    if array.shape == ():
        print(f"Warning: {key} is empty. Skipping...")
        continue

    # 如果是一维数组，转换为二维
    if array.ndim == 1:
        array = array.reshape(-1, 1)  # 将一维数组转换为二维 (n, 1)
    elif array.ndim > 2:
        print(f"Warning: {key} has more than 2 dimensions. Skipping...")
        continue

    # 将数组存储到字典中，键为数组名称，值为数组数据
    data_dict[key] = array

# 将所有数组合并为一个 DataFrame
# 使用 pd.concat 将列拼接在一起
df = pd.concat([pd.DataFrame(data_dict[key], columns=[key]) for key in data_dict], axis=1)

# 添加编号列，从 1 开始
df.insert(0, 'Index', range(1, len(df) + 1))  # 在第一列插入从 1 开始的编号

# 将 DataFrame 保存为单个 .csv 文件
df.to_csv('result.csv', index=False)
print("All data saved to result.csv")