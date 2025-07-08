import torch
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import joblib
import argparse
import os
import math
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.preprocessing import StandardScaler

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 模型定义 -------------------------
class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.pre_fc = torch.nn.Linear(input_dim, 16)
        self.lstm = torch.nn.LSTM(
            input_size=16,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        x = torch.relu(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 命令行参数解析 ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='智能发酵优化系统')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--Timestamp', type=str, required=True)
    return parser.parse_args()

# ---------------------- 增强型日期处理预处理 ----------------------
def load_and_preprocess_data(dataset_path, timestamp, window_size=20):
    """支持多种日期格式的预处理"""
    try:
        data_path = os.path.join(dataset_path, "data.xlsx")
        df = pd.read_excel(data_path)
        
        # 灵活解析数据文件中的时间戳（兼容日-月-年和月-日-年格式）
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'],
            dayfirst=True,  # 关键修改：优先解析为日-月格式
            format='mixed',  # 自动推断格式
            errors='coerce'
        )
        
        # 解析用户输入的时间戳（明确指定日-月-年格式）
        target_time = pd.to_datetime(
            timestamp,
            dayfirst=True,
            format="%d-%m-%Y %H:%M:%S"  # 精确匹配"16-11-2022 18:25:22"格式
        )
        
        # 验证时间有效性
        if pd.isnull(target_time):
            raise ValueError(f"无法解析输入时间: {timestamp}")
        
        # 时间对齐
        time_diff = (df['Timestamp'] - target_time).abs()
        closest_idx = time_diff.idxmin()
        
        # 原始特征提取
        raw_features = df[[
            'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
            'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
            'dm_spump3', 'dm_spump4', 'induction'
        ]].values.copy()
        
        # 累积量差分处理（与训练时完全一致）
        cumulative_cols = [4,5,6,7,8,9]
        for col in cumulative_cols:
            diffs = np.diff(raw_features[:, col], prepend=0)
            raw_features[:, col] = np.where(diffs <= 0, 0.001, diffs)
        
        # 构建初始序列
        start_idx = max(0, closest_idx - window_size + 1)
        sequence = raw_features[start_idx:closest_idx+1]
        
        # 序列填充
        if len(sequence) < window_size:
            pad_size = window_size - len(sequence)
            padding = np.zeros((pad_size, 11))
            sequence = np.vstack([padding, sequence])
        
        return sequence[-window_size+1:], raw_features[closest_idx]

    except Exception as e:
        print(f"[数据预处理错误] {str(e)}")
        exit(1)

# ---------------------- 模型加载 ----------------------
def load_model(model_path, input_dim, hidden_dim, output_dim, n_layers):
    model = LSTMPredictor(input_dim, hidden_dim, output_dim, n_layers)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[模型加载错误] {str(e)}")
        exit(1)

# ---------------------- 物理边界定义 ----------------------
PHYSICAL_BOUNDS = [
    (0.1, 100.0),    # m_ls_opt_do (%)
    (25.0, 40.0),    # m_temp (℃)
    (100.0, 1000.0), # m_stirrer (rpm)
    (0.1, 1000.0),   # dm_o2 (L/h)
    (0.1, 1000.0),   # dm_air (L/h)
    (0.1, 1000.0),   # dm_spump1 (mL/h)
    (0.1, 1000.0),   # dm_spump2 (mL/h)
    (0.1, 1000.0),   # dm_spump3 (mL/h)
    (0.1, 1000.0),   # dm_spump4 (mL/h)
    (0, 1)           # induction (0/1)
]

# ---------------------- 边界强制约束 ----------------------
def enforce_constraints(X):
    """确保参数在物理边界内"""
    X = np.array(X, dtype=np.float32)
    # 处理前9个连续参数
    for i in range(9):
        X[i] = np.clip(X[i], PHYSICAL_BOUNDS[i][0], PHYSICAL_BOUNDS[i][1])
    # 处理induction参数
    X[9] = 1 if X[9] >= 0.5 else 0
    return X

# ---------------------- 优化目标函数 ----------------------
def create_objective(model, scaler, initial_sequence, last_raw_values):
    """创建带约束的目标函数"""
    window_size = 20
    
    def objective(X):
        try:
            X = enforce_constraints(X)  # 强制参数约束
            
            # 构建完整输入行（考虑累积量特征）
            new_row = np.array([
                last_raw_values[0],   # m_ph保持最后观测值
                X[0],   # m_ls_opt_do
                X[1],   # m_temp
                X[2],   # m_stirrer
                X[3] + last_raw_values[4],   # dm_o2（累积量）
                X[4] + last_raw_values[5],   # dm_air
                X[5] + last_raw_values[6],   # dm_spump1
                X[6] + last_raw_values[7],   # dm_spump2
                X[7] + last_raw_values[8],   # dm_spump3
                X[8] + last_raw_values[9],   # dm_spump4
                X[9]    # induction
            ], dtype=np.float32)
            
            # 生成完整序列
            full_sequence = np.vstack([initial_sequence, new_row])
            
            # 应用与训练相同的预处理
            processed_sequence = full_sequence.copy()
            cumulative_cols = [4,5,6,7,8,9]
            for col in cumulative_cols:
                diffs = np.diff(processed_sequence[:, col], prepend=0)
                processed_sequence[:, col] = np.where(diffs <= 0, 0.001, diffs)
            
            # 归一化
            scaled_sequence = scaler.transform(processed_sequence)
            
            # 转换为张量
            input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                hidden = model.init_hidden(1)
                prediction, _ = model(input_tensor, hidden)
                prediction = prediction.item()
            
            # 限制预测值在合理范围内
            prediction = max(0.0, min(10.0, prediction))
            return -prediction  # 最小化负值相当于最大化OD600
            
        except Exception as e:
            print(f"[目标函数错误] {str(e)}")
            return float('inf')
    
    return objective

# ---------------------- 优化主流程 ----------------------
def optimize_inputs(args):
    print("\n" + "="*40)
    print(" 智能发酵优化系统 v3.2".center(40))
    print("="*40)
    
    # 文件验证
    required_files = {
        "模型权重文件": args.weights,
        "数据文件": os.path.join(args.dataset, "data.xlsx"),
        "归一化文件": os.path.join(args.dataset, "scaler_new_data.save")
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"[错误] {name}不存在：{path}")
            exit(1)
    
    # 加载组件
    try:
        scaler = joblib.load(required_files["归一化文件"])
        model = load_model(args.weights, 11, args.hidden_dim, 1, args.num_layers)
        initial_sequence, last_raw = load_and_preprocess_data(args.dataset, args.Timestamp)
    except Exception as e:
        print(f"[初始化错误] {str(e)}")
        exit(1)
    
    # 遗传算法参数（优化版）
    algorithm_params = {
        'max_num_iteration': 500,
        'population_size': 100,
        'mutation_probability': 0.25,
        'elit_ratio': 0.15,
        'crossover_probability': 0.85,
        'crossover_type': 'two_point',
        'parents_portion': 0.3,
        'max_iteration_without_improv': 150,
        'variable_type': ['real']*9 + ['int'],
        'variable_boundaries': PHYSICAL_BOUNDS
    }
    
    print("\n[状态] 开始参数优化...")
    start_time = time.time()
    
    # 创建目标函数
    objective = create_objective(model, scaler, initial_sequence, last_raw)
    
    # 运行优化
    optimizer = ga(
        function=objective,
        dimension=10,
        algorithm_parameters=algorithm_params
    )
    
    optimizer.run()
    
    # 处理结果
    optimization_time = time.time() - start_time
    best_params = enforce_constraints(optimizer.output_dict['variable'])
    best_od = -optimizer.output_dict['function']
    
    # 显示结果
    print("\n" + "="*40)
    print(f" 优化结果 ".center(40, '='))
    print(f" 预测OD600: {best_od:.4f}")
    print(f" 总耗时: {optimization_time:.2f}秒")
    print("-"*40)
    print(f" 优化参数 ".center(40, '-'))
    
    param_info = [
        ("溶氧设定值", best_params[0], "%"),
        ("温度", best_params[1], "℃"),
        ("搅拌速率", best_params[2], "rpm"),
        ("O2流量", best_params[3], "L/h"),
        ("空气流量", best_params[4], "L/h"),
        ("补料泵1", best_params[5], "mL/h"),
        ("补料泵2", best_params[6], "mL/h"),
        ("补料泵3", best_params[7], "mL/h"),
        ("补料泵4", best_params[8], "mL/h"),
        ("诱导剂", best_params[9], "状态")
    ]
    
    for name, value, unit in param_info:
        if name == "诱导剂":
            state = "开启" if value >= 0.5 else "关闭"
            print(f" {name:<10}：{state}")
        else:
            print(f" {name:<10}：{value:.2f} {unit}")
    
    return best_params, best_od

if __name__ == "__main__":
    args = parse_args()
    try:
        optimized_params, best_od = optimize_inputs(args)
    except KeyboardInterrupt:
        print("\n[警告] 用户中断操作！")
        exit(0)
    except Exception as e:
        print(f"\n[严重错误] 程序异常终止: {str(e)}")
        exit(1)