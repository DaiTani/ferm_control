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

def process_timestamp(df):
    """将时间戳列转换为从0开始的分钟数"""
    processed = False 
    for sheet_name in df:
        current_sheet = df[sheet_name]
        if 'Timestamp' not in current_sheet.columns: 
            continue 

        try:
            # 将时间字符串转换为时间增量 
            timedelta_series = pd.to_timedelta(current_sheet['Timestamp']) 
            # 计算总分钟数并替换原列 
            current_sheet['Timestamp'] = timedelta_series.dt.total_seconds()  / 60 
            df[sheet_name] = current_sheet 
            processed = True 
        except Exception as e:
            print(f"工作表 [{sheet_name}] 时间戳转换失败: {str(e)}")
            continue 

    return df, processed 

def process_columns(df, file_path):
    """处理列顺序和缺失检测"""
    modified = False 
    for sheet_name in df:
        current_sheet = df[sheet_name]
        original_columns = current_sheet.columns.tolist() 
        
        # 检测缺失列 
        missing = [col for col in COLUMN_ORDER if col not in current_sheet]
        if missing:
            print(f"[警告] 文件 {file_path} 工作表 [{sheet_name}] 缺失列：{', '.join(missing)}")

        # 构建新列顺序 
        ordered = [col for col in COLUMN_ORDER if col in current_sheet]
        extras = [col for col in current_sheet if col not in COLUMN_ORDER]
        new_columns = ordered + extras 

        # 重新排序 
        if new_columns != original_columns:
            df[sheet_name] = current_sheet[new_columns]
            modified = True 
    return df, modified 

def replace_zero_values(df):
    """替换非Timestamp和非od_600列中的0为0.0001"""
    modified = False 
    for sheet_name in df:
        current_sheet = df[sheet_name]
        for col in current_sheet.columns: 
            if col not in ['Timestamp', 'od_600']:
                # 检查是否为数值型列 
                if pd.api.types.is_numeric_dtype(current_sheet[col]): 
                    mask = current_sheet[col] == 0 
                    if mask.any(): 
                        current_sheet.loc[mask,  col] = 0.0001 
                        modified = True 
        df[sheet_name] = current_sheet
    return df, modified 

def interpolate_od_600(df):
    """对od_600列中的0值进行渐进数值填充"""
    modified = False 
    for sheet_name in df:
        current_sheet = df[sheet_name]
        if 'od_600' in current_sheet.columns: 
            # 检查是否为数值型列 
            if pd.api.types.is_numeric_dtype(current_sheet['od_600']): 
                # 使用线性插值填充od_600中的零值 
                # 先将零值标记为NaN 
                od_series = current_sheet['od_600'].replace(0, pd.NA)
                # 进行插值 
                interpolated = od_series.interpolate(method='linear',  limit_direction='both')
                # 替换回原列 
                current_sheet['od_600'] = interpolated.fillna(0)   # 如果仍有NaN，保留为零（边界情况）
                modified = True 
        df[sheet_name] = current_sheet
    return df, modified 

def process_specific_columns(df):
    """处理dm_o2、dm_air、dm_spump1、dm_spump2、dm_spump3、dm_spump4列的数据"""
    modified = False
    target_columns = ['dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2', 'dm_spump3', 'dm_spump4']
    for sheet_name in df:
        current_sheet = df[sheet_name]
        for col in target_columns:
            if col in current_sheet.columns:
                # 检查是否为数值型列
                if pd.api.types.is_numeric_dtype(current_sheet[col]):
                    col_data = current_sheet[col].copy()
                    for i in range(len(col_data) - 1, 0, -1):
                        col_data.iloc[i] = col_data.iloc[i] - col_data.iloc[i - 1]
                    col_data.iloc[0] = 0.0001
                    current_sheet[col] = col_data
                    modified = True
        df[sheet_name] = current_sheet
    return df, modified

def process_excel_file(file_path):
    """处理单个Excel文件"""
    try:
        # 读取所有工作表 
        df = pd.read_excel(file_path,  sheet_name=None)
        
        # 处理时间戳 
        df, ts_processed = process_timestamp(df)
        
        # 处理列顺序 
        df, col_processed = process_columns(df, file_path)

        # 处理特定列
        df, specific_col_processed = process_specific_columns(df)
 
        # 替换非指定列的零值 
        df, zero_replaced = replace_zero_values(df)

        # 处理od_600列的零值 
        df, od_interpolated = interpolate_od_600(df)

        if ts_processed or col_processed or zero_replaced or od_interpolated or specific_col_processed:
            # 备份原文件 
            backup_path = file_path.replace('data.xlsx',  'data_origin.xlsx') 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            if os.path.exists(backup_path): 
                backup_path_with_time = file_path.replace('data.xlsx',  f'data_origin_{timestamp}.xlsx')
                os.rename(file_path,  backup_path_with_time)
                print(f"备份文件已存在，生成新备份：{backup_path_with_time}")
            else:
                os.rename(file_path,  backup_path)
                print(f"备份文件已创建：{backup_path}")
            
            # 保存新文件 
            with pd.ExcelWriter(file_path) as writer:
                for sheet_name, data in df.items(): 
                    data.to_excel(writer,  sheet_name=sheet_name, index=False)
            print(f"√ 成功处理：{file_path}")
        else:
            print(f"○ 无需修改：{file_path}")

    except Exception as e:
        print(f"× 处理失败：{file_path}\n   错误信息：{str(e)}")

def main():
    # 遍历当前目录及所有子目录 
    for root, dirs, files in os.walk('.'): 
        if 'data.xlsx'  in files:
            file_path = os.path.join(root,  'data.xlsx') 
            process_excel_file(file_path)

if __name__ == "__main__":
    print("=== Excel文件处理程序 ===")
    print("正在扫描目录...")
    main()
    print("处理完成！")