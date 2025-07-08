import os
import pandas as pd
import re
from datetime import datetime

# 标准列顺序定义
COLUMN_ORDER = [
    'Timestamp', 'dm_air', 'm_ls_opt_do', 'm_ph',
    'm_stirrer', 'm_temp', 'dm_o2', 'dm_spump1',
    'dm_spump2', 'dm_spump3', 'dm_spump4', 
    'induction', 'od_600'
]

def is_already_processed(time_str):
    """检查时间戳是否已经是数值格式（分钟）"""
    try:
        float(time_str)
        return True
    except:
        return False

def parse_timestamp(df):
    """智能解析时间戳列，支持多种格式"""
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S')
    except ValueError:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
        except:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def process_timestamp(df):
    """处理时间戳列，转换为从0开始的分钟数"""
    processed = False
    for sheet_name in df:
        current_sheet = df[sheet_name]
        if 'Timestamp' not in current_sheet.columns:
            continue
            
        if current_sheet['Timestamp'].apply(is_already_processed).all():
            continue

        current_sheet = parse_timestamp(current_sheet)
        start_time = current_sheet['Timestamp'].iloc[0]
        time_diff = current_sheet['Timestamp'] - start_time
        current_sheet['Timestamp'] = time_diff.dt.total_seconds() / 60  # 转换为分钟数
        df[sheet_name] = current_sheet
        processed = True
    return df, processed

def process_columns(df, file_path):
    """处理列顺序和缺失检测"""
    modified = False
    for sheet_name in df:
        current_sheet = df[sheet_name]
        original_columns = current_sheet.columns.tolist()
        
        missing = [col for col in COLUMN_ORDER if col not in current_sheet]
        if missing:
            print(f"[警告] 文件 {file_path} 工作表 [{sheet_name}] 缺失列：{', '.join(missing)}")

        ordered = [col for col in COLUMN_ORDER if col in current_sheet]
        extras = [col for col in current_sheet if col not in COLUMN_ORDER]
        new_columns = ordered + extras

        if new_columns != original_columns:
            df[sheet_name] = current_sheet[new_columns]
            modified = True
    return df, modified

def process_excel_file(file_path):
    """处理单个Excel文件"""
    try:
        df = pd.read_excel(file_path, sheet_name=None)
        df, ts_processed = process_timestamp(df)
        df, col_processed = process_columns(df, file_path)

        if ts_processed or col_processed:
            backup_path = file_path.replace('data.xlsx', 'data_origin.xlsx')
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
            
            with pd.ExcelWriter(file_path) as writer:
                for sheet_name, data in df.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"√ 成功处理：{file_path}")
        else:
            print(f"○ 无需修改：{file_path}")

    except Exception as e:
        print(f"× 处理失败：{file_path}\n   错误信息：{str(e)}")

def main():
    for root, dirs, files in os.walk('.'):
        if 'data_origin.xlsx' in files:
            continue
            
        if 'data.xlsx' in files:
            file_path = os.path.join(root, 'data.xlsx')
            process_excel_file(file_path)

if __name__ == "__main__":
    print("=== Excel文件处理程序 ===")
    print("正在扫描目录...")
    main()
    print("处理完成！")