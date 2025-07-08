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
def extract_initial_inputs(dataset_path, timestamp):
    try:
        data_path = os.path.join(dataset_path, "data.xlsx")
        df = pd.read_excel(data_path)
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        target_time = pd.to_datetime(timestamp, dayfirst=True)
        
        time_diff = (df['Timestamp'] - target_time).abs()
        closest_idx = time_diff.idxmin()
        row = df.iloc[closest_idx]
        
        params = row[[
            'm_ls_opt_do', 'm_temp', 'm_stirrer', 
            'dm_o2', 'dm_air', 'dm_spump1', 
            'dm_spump2', 'dm_spump3', 'dm_spump4', 
            'induction'
        ]].values.astype(np.float32)
        
        # 防止初始参数出现零值
        params = np.where(params == 0, 0.001, params)
        return params
    
    except Exception as e:
        print(f"[错误] 参数提取失败: {str(e)}")
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

# ---------------------- 目标函数 ----------------------
def objective_function(X, model, scaler, dataset_path):
    if not hasattr(objective_function, 'test_loader'):
        try:
            data_path = os.path.join(dataset_path, "data.xlsx")
            raw_df = pd.read_excel(data_path)
            
            features = raw_df[[
                'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
                'dm_spump3', 'dm_spump4', 'induction'
            ]].values
            
            # 处理累积量特征并防止零值
            cumulative_indices = [4,5,6,7,8,9]
            for idx in cumulative_indices:
                diff = np.diff(features[:, idx], prepend=0)
                features[:, idx] = np.where(diff == 0, 0.001, diff)
            
            scaled_features = scaler.transform(features)
            
            def create_sequences(data, window_size=20):
                sequences = []
                for i in range(len(data) - window_size + 1):
                    sequences.append(data[i:i+window_size])
                return np.array(sequences)
            
            X_seq = create_sequences(scaled_features)
            y_seq = raw_df['od_600'].values[19:]
            
            # 确保有有效数据
            if len(X_seq) == 0 or len(y_seq) == 0:
                raise ValueError("生成的数据序列为空，请检查数据文件和时间窗口设置")
            
            dataset = TensorDataset(
                torch.FloatTensor(X_seq),
                torch.FloatTensor(y_seq.reshape(-1, 1))
            )
            
            objective_function.test_loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False
            )
            
        except Exception as e:
            print(f"[错误] 数据预处理失败: {str(e)}")
            exit(1)
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in objective_function.test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            hidden = model.init_hidden(inputs.size(0))
            outputs, _ = model(inputs, hidden)
            
            loss = torch.sqrt(torch.nn.functional.mse_loss(outputs, targets))
            total_loss += loss.item()
    
    # 防止除以零错误
    num_batches = len(objective_function.test_loader)
    if num_batches == 0:
        raise ValueError("测试数据加载器为空，请检查数据预处理")
    
    avg_rmse = total_loss / num_batches
    return -avg_rmse

# ---------------------- 优化主函数 ----------------------
def optimize_inputs(args):
    print("\n" + "="*40)
    print(" 智能发酵参数优化系统 v1.2")
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
    
    try:
        scaler = joblib.load(required_files["归一化文件"])
    except Exception as e:
        print(f"[错误] 归一化器加载失败: {str(e)}")
        exit(1)
    
    model = load_model(
        model_path=args.weights,
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    )
    
    initial_params = extract_initial_inputs(args.dataset, args.Timestamp)
    
    # 参数边界（强化安全范围）
    epsilon = 1e-6  # 安全系数
    varbound = np.array([
        [max(epsilon, initial_params[0]-10), initial_params[0]+10],
        [initial_params[1]-1+epsilon, initial_params[1]+1-epsilon],
        [max(epsilon, initial_params[2]-100), initial_params[2]+100],
        [max(epsilon, initial_params[3]-100), initial_params[3]+100],
        [max(epsilon, initial_params[4]-100), initial_params[4]+100],
        [max(epsilon, initial_params[5]-1), initial_params[5]+1],
        [max(epsilon, initial_params[6]-1), initial_params[6]+1],
        [max(epsilon, initial_params[7]-1), initial_params[7]+1],
        [max(epsilon, initial_params[8]-5), initial_params[8]+5],
        [epsilon, 1-epsilon]
    ], dtype=np.float32)
    
    algorithm_params = {
        'max_num_iteration': 200,
        'population_size': 100,
        'mutation_probability': 0.2,
        'elit_ratio': 0.1,
        'crossover_probability': 0.7,
        'crossover_type': 'uniform',
        'parents_portion': 0.3,
        'max_iteration_without_improv': 50
    }
    
    print("\n[状态] 开始参数优化...")
    start_time = time.time()
    
    optimizer = ga(
        function=lambda X: objective_function(X, model, scaler, args.dataset),
        dimension=len(varbound),
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_params
    )
    
    optimizer.run()
    
    optimization_time = time.time() - start_time
    best_params = optimizer.output_dict['variable']
    best_rmse = -optimizer.output_dict['function']
    
    print("\n" + "="*40)
    print(" 优化结果 ".center(40, '='))
    print(f" 最低RMSE: {best_rmse:.4f}")
    print(f" 总耗时: {optimization_time:.2f}秒")
    print("-"*40)
    print(" 优化参数 ".center(40, '-'))
    
    param_info = [
        ("溶氧设定值(%)", best_params[0], initial_params[0]),
        ("温度(℃)", best_params[1], initial_params[1]),
        ("搅拌速率(rpm)", best_params[2], initial_params[2]),
        ("O2流量(L/h)", best_params[3], initial_params[3]),
        ("空气流量(L/h)", best_params[4], initial_params[4]),
        ("补料泵1(mL/h)", best_params[5], initial_params[5]),
        ("补料泵2(mL/h)", best_params[6], initial_params[6]),
        ("补料泵3(mL/h)", best_params[7], initial_params[7]),
        ("补料泵4(mL/h)", best_params[8], initial_params[8]),
        ("诱导剂浓度(mM)", best_params[9], initial_params[9])
    ]
    
    for name, opt_val, init_val in param_info:
        try:
            change = ((opt_val - init_val)/abs(init_val))*100
            change_str = f"{change:+.1f}%"
        except ZeroDivisionError:
            change_str = "N/A (初始值为0)"
        print(f" {name:<15} 初始值: {init_val:.4f} → 优化值: {opt_val:.4f} ({change_str})")
    
    return best_params, best_rmse

if __name__ == "__main__":
    args = parse_args()
    try:
        optimized_params, best_rmse = optimize_inputs(args)
    except KeyboardInterrupt:
        print("\n[警告] 用户中断操作！")
        exit(0)
    except Exception as e:
        print(f"\n[严重错误] 程序异常终止: {str(e)}")
        exit(1)