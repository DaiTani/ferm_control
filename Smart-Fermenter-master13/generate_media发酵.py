import random
import pandas as pd
import os
from datetime import datetime

# 定义培养基成分及其取值范围
components = {
    "酵母粉": (2, 8),
    "蛋白胨": (6, 14),
    "甘油": (2, 8),
    "葡萄糖": (0, 2.5),
    "硫酸镁": (0.5, 3.5),  # 注意这里有一个空格
    "硫酸氢二钠": (5, 10),
    "磷酸二氢钾": (2, 5),
    "氯化铵": (2, 5),
    "硫酸钠": (0, 2.5),
    "柠檬酸": (0, 2.5)
}

# 随机生成96组培养基数据，按照新的取值间隔规则
def generate_random_media(num_samples=96):
    data = []
    components_list = list(components.keys())
    samples_per_component = num_samples // len(components_list)  # 每个成分12个样本
    
    for comp_idx, component in enumerate(components_list):
        min_val, max_val = components[component]
        # 生成降序值列表
        if max_val - min_val > 4:
            step = 1
            values = list(range(max_val, min_val-1, -step))
        elif max_val - min_val > 2:
            step = 0.5
            values = list(range(int(max_val * 2), int(min_val * 2) - 1, -1))  # 处理小数步长
            values = [v / 2 for v in values]
        else:
            step = 0.5
            values = list(range(int(max_val * 2), int(min_val * 2) - 1, -1))  # 处理小数步长
            values = [v / 2 for v in values]
        
        # 循环填充至所需样本数
        selected_values = []
        while len(selected_values) < samples_per_component:
            selected_values.extend(values)
        selected_values = selected_values[:samples_per_component]
        
        # 创建样本（组编号作为第一列）
        for val in selected_values:
            # 先设置组编号
            sample = {"组编号": comp_idx * samples_per_component + len(data) + 1}
            
            # 设置默认值
            default_values = {
                "酵母粉": 5,
                "蛋白胨": 10,
                "甘油": 5,
                "葡萄糖": 0.5,
                "硫酸镁": 2,  # 注意这里有一个空格
                "硫酸氢二钠": 9,
                "磷酸二氢钾": 3.4,
                "氯化铵": 2.7,
                "硫酸钠": 0.7,
                "柠檬酸": 1
            }
            
            # 设置其他成分的值为默认值
            for c in components_list:
                sample[c] = default_values[c]
            
            # 当前成分设置为梯度值
            sample[component] = val
            
            # 将样本添加到数据列表
            data.append(sample)
    
    return data

# 随机打乱顺序并重新编号
def shuffle_and_renumber(data):
    # 随机打乱顺序
    random.shuffle(data)
    
    # 重新编号
    for i, sample in enumerate(data):
        sample["组编号"] = i + 1
    
    return data

# 检查文件是否存在，若存在则备份
def backup_existing_file(file_name):
    if os.path.exists(file_name):
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建备份文件名
        backup_file_name = file_name.replace(".xlsx", f"_backup_{timestamp}.xlsx")
        # 备份原文件
        os.rename(file_name, backup_file_name)
        print(f"原文件已备份为 {backup_file_name}")

# 将数据保存到Excel文件
def save_to_excel(data, file_name="培养基数据-发酵.xlsx"):
    # 检查文件是否存在并备份
    backup_existing_file(file_name)
    
    # 保存新文件
    df = pd.DataFrame(data)
    df.to_excel(file_name, index=False)
    print(f"数据已成功保存到 {file_name}")

# 主函数
if __name__ == "__main__":
    # 生成96组随机培养基数据
    media_data = generate_random_media(96)
    
    # 随机打乱顺序并重新编号
    media_data = shuffle_and_renumber(media_data)
    
    # 保存到Excel文件
    save_to_excel(media_data)