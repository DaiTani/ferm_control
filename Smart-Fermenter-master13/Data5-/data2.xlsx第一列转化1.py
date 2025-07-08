import os
import pandas as pd
import re
from datetime import datetime

def is_already_processed(time_str):
    """检查时间戳是否已经是HH:MM:SS格式"""
    return re.match(r'^\d{2}:\d{2}:\d{2}$', str(time_str)) is not None

def parse_timestamp(df):
    """智能解析时间戳列，支持多种格式"""
    try:
        # 尝试ISO格式 (2022-06-08T08:21:17)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S')
    except ValueError:
        # 尝试原格式 (16-11-2022 08:34:24)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    return df

def process_excel_file(file_path):
    """处理单个Excel文件"""
    try:
        df = pd.read_excel(file_path, sheet_name=None)  # 读取所有工作表
        
        processed = False
        for sheet_name in df:
            current_sheet = df[sheet_name]
            if 'Timestamp' not in current_sheet.columns:
                continue
                
            # 检查是否已处理
            if current_sheet['Timestamp'].apply(is_already_processed).all():
                print(f"跳过已处理文件: {file_path}")
                return
            
            # 处理时间戳
            current_sheet = parse_timestamp(current_sheet)
            start_time = current_sheet['Timestamp'].iloc[0]
            current_sheet['Timestamp'] = (current_sheet['Timestamp'] - start_time).apply(
                lambda x: f"{x.components.hours:02}:{x.components.minutes:02}:{x.components.seconds:02}"
            )
            df[sheet_name] = current_sheet
            processed = True
        
        if processed:
            # 备份原文件
            backup_path = file_path.replace('data.xlsx', 'data_origin.xlsx')
            os.rename(file_path, backup_path)
            # 保存新文件
            with pd.ExcelWriter(file_path) as writer:
                for sheet_name, data in df.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"处理完成: {file_path}")
            
    except Exception as e:
        print(f"处理失败: {file_path} - 错误: {str(e)}")

def main():
    # 遍历当前目录及所有子目录
    for root, dirs, files in os.walk('.'):
        if 'data_origin.xlsx' in files:  # 跳过备份文件
            continue
        if 'data.xlsx' in files:
            file_path = os.path.join(root, 'data.xlsx')
            process_excel_file(file_path)

if __name__ == "__main__":
    main()
    print("所有文件处理完成！")