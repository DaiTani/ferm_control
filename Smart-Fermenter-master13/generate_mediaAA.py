import random
import pandas as pd
import os
from datetime import datetime

# 定义培养基成分及其取值范围
components = {
    "酵母浸粉": (5, 10),
    "蛋白胨": (10, 15),
    "葡萄糖": (5, 15),
    "(NH4)2SO4": (5, 10),
    "K2HPO4·3H2O": (5, 10),
    "柠檬酸·H2O": (1, 1.5),
    "硫酸镁": (0.5, 1),
    "微量元素": (1, 3)
}

# 随机生成96组培养基数据，按照新的取值间隔规则
def generate_random_media(num_samples=96):
    data = []
    for i in range(1, num_samples + 1):
        sample = {"组编号": i}
        for component, (min_val, max_val) in components.items():
            if max_val - min_val > 5:  # 最大范围超过5
                value = round(random.choice([x * 0.5 for x in range(int(min_val * 2), int(max_val * 2) + 1)]), 1)
            else:  # 最大范围小于等于5
                value = round(random.choice([x * 0.25 for x in range(int(min_val * 4), int(max_val * 4) + 1)]), 2)
            sample[component] = value
        data.append(sample)
    return data

# 对数据进行排序并更新组编号
def sort_and_update_group_numbers(data):
    # 将数据转换为DataFrame以便排序
    df = pd.DataFrame(data)
    
    # 按照所有列依次排序
    sorted_df = df.sort_values(by=list(components.keys()))
    
    # 更新组编号
    sorted_df["组编号"] = range(1, len(sorted_df) + 1)
    
    # 返回排序后的数据
    return sorted_df.to_dict(orient="records")

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
def save_to_excel(data, file_name="培养基数据.xlsx"):
    # 检查文件是否存在并备份
    backup_existing_file(file_name)
    
    # 保存新文件
    df = pd.DataFrame(data)
    df.to_excel(file_name, index=False)
    print(f"数据已成功保存到 {file_name}")

# 重新编号规则
def reassign_group_numbers(data):
    new_data = []
    rows_per_column = 12  # 每列12个编号
    columns = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    for i, sample in enumerate(data):
        # 计算行号和列号
        column_index = i // rows_per_column
        row_index = i % rows_per_column + 1
        
        # 构造新的编号
        new_sample = sample.copy()
        new_sample["组编号"] = f"{columns[column_index]}{row_index}"
        new_data.append(new_sample)
    
    return new_data

# 新增功能：将48行数据复制为96行数据，并重新编号
def duplicate_and_renumber(data):
    duplicated_data = []
    rows_per_column = 12  # 每列12个编号
    columns = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # 将48行数据复制为96行数据，每行重复一次
    for i in range(len(data)):
        duplicated_data.append(data[i])  # 添加原始行
        duplicated_data.append(data[i])  # 再次添加原始行（复制）
    
    # 重新编号
    new_data = []
    for i, sample in enumerate(duplicated_data):
        # 计算行号和列号
        column_index = i // rows_per_column
        row_index = i % rows_per_column + 1
        
        # 构造新的编号
        new_sample = sample.copy()
        new_sample["组编号"] = f"{columns[column_index]}{row_index}"
        new_data.append(new_sample)
    
    return new_data

# 主函数
if __name__ == "__main__":
    # 生成96组随机培养基数据
    media_data = generate_random_media(96)
    
    # 对数据进行排序并更新组编号
    sorted_media_data = sort_and_update_group_numbers(media_data)
    
    # 将96组数据分为两部分，每部分48组
    part1 = sorted_media_data[:48]
    part2 = sorted_media_data[48:]
    
    # 重新编号
    part1_renumbered = reassign_group_numbers(part1)
    part2_renumbered = reassign_group_numbers(part2)
    
    # 将48行数据复制为96行数据，并重新编号
    part1_duplicated = duplicate_and_renumber(part1_renumbered)
    part2_duplicated = duplicate_and_renumber(part2_renumbered)
    
    # 保存到两个Excel文件
    save_to_excel(part1_duplicated, "培养基数据1_扩展.xlsx")
    save_to_excel(part2_duplicated, "培养基数据2_扩展.xlsx")