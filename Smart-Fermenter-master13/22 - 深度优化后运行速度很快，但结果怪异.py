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

# ---------------------- 数据预处理函数 ----------------------
def load_and_preprocess_data(dataset_path, timestamp, window_size=20):
    """加载并预处理历史数据，返回初始序列"""
    try:
        data_path = os.path.join(dataset_path, "data.xlsx")
        df = pd.read_excel(data_path)
        
        # 时间对齐
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        target_time = pd.to_datetime(timestamp, dayfirst=True)
        time_diff = (df['Timestamp'] - target_time).abs()
        closest_idx = time_diff.idxmin()
        
        # 提取特征（与训练时完全一致的预处理）
        features = df[[
            'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
            'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
            'dm_spump3', 'dm_spump4', 'induction'
        ]].values
        
        # 处理累积量特征（与训练时保持一致）
        cumulative_indices = [4,5,6,7,8,9]
        for idx in cumulative_indices:
            diff = np.diff(features[:, idx], prepend=0)
            features[:, idx] = np.where(diff == 0, 0.001, diff)
        
        # 获取初始序列（窗口大小-1）
        start_idx = max(0, closest_idx - window_size + 1)
        initial_sequence = features[start_idx:closest_idx+1]
        
        # 填充不足的窗口
        if len(initial_sequence) < window_size:
            padding = np.zeros((window_size - len(initial_sequence), 11))
            initial_sequence = np.vstack([padding, initial_sequence])
        
        return initial_sequence[-window_size+1:]  # 返回最后window_size-1个样本
    
    except Exception as e:
        print(f"[错误] 数据预处理失败: {str(e)}")
        exit(1)

# ---------------------- 模型加载函数 ----------------------
def load_model(model_path, input_dim, hidden_dim, output_dim, n_layers):
    model = LSTMPredictor(input_dim, hidden_dim, output_dim, n_layers)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[错误] 模型加载失败: {str(e)}")
        exit(1)

# ---------------------- 目标函数（修正核心）---------------------
def create_objective_function(model, scaler, initial_sequence):
    """闭包工厂函数，保持初始序列状态"""
    window_size = 20  # 与训练时的窗口大小一致
    
    def objective(X):
        # 参数校验
        if len(X) != 10:
            raise ValueError(f"参数数量错误，需要10个参数，得到{len(X)}个")
        
        # 整合新参数到输入序列
        new_row = np.array([
            initial_sequence[-1][0],   # m_ph保持不变
            X[0],   # m_ls_opt_do
            X[1],   # m_temp
            X[2],   # m_stirrer
            X[3],   # dm_o2
            X[4],   # dm_air
            X[5],   # dm_spump1
            X[6],   # dm_spump2
            X[7],   # dm_spump3
            X[8],   # dm_spump4
            X[9]    # induction
        ], dtype=np.float32)
        
        # 创建完整输入序列
        full_sequence = np.vstack([initial_sequence, new_row])
        scaled_sequence = scaler.transform(full_sequence)
        
        # 转换为模型输入格式
        input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(device)
        
        # 预测OD600
        with torch.no_grad():
            hidden = model.init_hidden(1)
            prediction, _ = model(input_tensor, hidden)
        
        # 返回负值用于最小化（实际最大化OD600）
        return -prediction.item()
    
    return objective

# ---------------------- 参数边界定义 ----------------------
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

# ---------------------- 优化主函数 ----------------------
def optimize_inputs(args):
    print("\n" + "="*40)
    print(" 智能发酵参数优化系统 v2.0")
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
        initial_sequence = load_and_preprocess_data(args.dataset, args.Timestamp)
    except Exception as e:
        print(f"[初始化错误] {str(e)}")
        exit(1)
    
    # 遗传算法参数
    algorithm_params = {
        'max_num_iteration': 50,
        'population_size': 25,
        'mutation_probability': 0.15,
        'elit_ratio': 0.1,
        'crossover_probability': 0.7,
        'crossover_type': 'uniform',
        'parents_portion': 0.3,
        'max_iteration_without_improv': 30,
        'variable_type': ['real']*9 + ['int'],
        'variable_boundaries': PHYSICAL_BOUNDS
    }
    
    print("\n[状态] 开始参数优化...")
    start_time = time.time()
    
    # 创建目标函数闭包
    objective = create_objective_function(model, scaler, initial_sequence)
    
    # 运行优化
    optimizer = ga(
        function=objective,
        dimension=10,
        algorithm_parameters=algorithm_params
    )
    
    optimizer.run()
    
    # 结果处理
    optimization_time = time.time() - start_time
    best_params = optimizer.output_dict['variable']
    best_od = -optimizer.output_dict['function']
    
    # 显示结果
    print("\n" + "="*40)
    print(" 优化结果 ".center(40, '='))
    print(f" 预测OD600: {best_od:.4f}")
    print(f" 总耗时: {optimization_time:.2f}秒")
    print("-"*40)
    print(" 优化参数 ".center(40, '-'))
    
    param_names = [
        "溶氧设定值(%)",
        "温度(℃)", 
        "搅拌速率(rpm)",
        "O2流量(L/h)",
        "空气流量(L/h)",
        "补料泵1(mL/h)",
        "补料泵2(mL/h)",
        "补料泵3(mL/h)",
        "补料泵4(mL/h)",
        "诱导剂状态"
    ]
    
    for name, value in zip(param_names, best_params):
        print(f" {name:<15}: {value:.4f}" if name != "诱导剂状态" else 
              f" {name:<15}: {'开启' if value > 0.5 else '关闭'}")

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