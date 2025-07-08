# 新建脚本 generate_scaler.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 加载数据
df = pd.read_excel("./Data5/data.xlsx")

# 提取12个特征（包含od_600）
features = df[[
    'od_600', 'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
    'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2', 'dm_spump3', 
    'dm_spump4', 'induction'
]].values.astype(np.float32)

# 处理缺失值和零值
features = np.nan_to_num(features, nan=0.001)
features[:, 5:11] = np.where(features[:, 5:11] <= 0, 0.001, features[:, 5:11])

# 训练并保存新标准化器
scaler = StandardScaler()
scaler.fit(features)
joblib.dump(scaler, "./Data5/scaler_new_data_v2.save")