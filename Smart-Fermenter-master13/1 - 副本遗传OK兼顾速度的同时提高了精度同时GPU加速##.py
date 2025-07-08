import torch
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import joblib
from dataset import FermentationData  # 假设 dataset.py 中有 FermentationData 类
from model import LSTMPredictor  # 假设 model.py 中有 LSTMPredictor 类
import argparse

# 动态设置设备
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')  # 如果有多个 GPU，使用第二个 GPU
    else:
        device = torch.device('cuda:0')  # 如果只有一个 GPU，使用第一个 GPU
else:
    device = torch.device('cpu')

# 加载训练好的模型
def load_model(model_path, input_dim, hidden_dim, output_dim, n_layers):
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)
    checkpoint = torch.load(model_path, map_location=device)  # 确保加载到正确的设备
    model.load_state_dict(checkpoint['weights'])
    
    # 将模型移动到指定的 GPU
    model.to(device)
    model.eval()
    return model

# 定义目标函数
def objective_function(X, model, scaler):
    X = np.array(X).reshape(1, -1)
    if X.shape[1] != 10:
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    X = np.hstack([X, np.zeros((X.shape[0], 1))])  # 添加虚拟特征
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # 移动到指定 GPU
    
    with torch.no_grad():
        h = model.init_hidden(1)
        h = tuple([e.data.to(device) for e in h])
        prediction, _ = model(X_tensor, h)
    
    return -prediction.item()

# 遗传算法优化
def optimize_inputs(model_path, scaler_path, input_dim, hidden_dim, output_dim, n_layers, initial_inputs):
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 动态调整输入特征的边界
    varbound = np.array([
        [max(0, initial_inputs[0] - 10), initial_inputs[0] + 10],  # m_ls_opt_do ±10
        [max(20, initial_inputs[1] - 1), min(40, initial_inputs[1] + 1)],  # m_temp ±1
        [max(0, initial_inputs[2] - 100), initial_inputs[2] + 100],  # m_stirrer ±100
        [max(0, initial_inputs[3] - 100), initial_inputs[3] + 100],  # dm_o2 ±100
        [max(0, initial_inputs[4] - 100), initial_inputs[4] + 100],  # dm_air ±100
        [max(0, initial_inputs[5] - 1), initial_inputs[5] + 1],  # dm_spump1 ±1
        [max(0, initial_inputs[6] - 1), initial_inputs[6] + 1],  # dm_spump2 ±1
        [max(0, initial_inputs[7] - 1), initial_inputs[7] + 1],  # dm_spump3 ±1
        [max(0, initial_inputs[8] - 5), initial_inputs[8] + 5],  # dm_spump4 ±5
        [max(0, initial_inputs[9] - 0.1), min(1, initial_inputs[9] + 0.1)]  # induction ±0.1
    ])

    # 设置遗传算法参数
    algorithm_param = {
        'max_num_iteration': 500,  # 增加总迭代次数
        'population_size': 100,    # 增加种群大小
        'mutation_probability': 0.2,  # 增加变异概率
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50  # 增加无改进迭代次数
    }

    # 初始化遗传算法
    model_ga = ga(
        function=lambda X: objective_function(X, model, scaler),
        dimension=len(varbound),
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        progress_bar=True
    )

    # 运行遗传算法
    model_ga.run()

    # 输出最优解
    best_solution = model_ga.output_dict['variable']
    best_value = -model_ga.output_dict['function']
    print("Optimal Inputs:", best_solution)
    print("Maximum Predicted Output (od_600):", best_value)

    return best_solution, best_value

# 使用优化后的输入特征进行预测
def predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, hidden_dim, output_dim, n_layers):
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    X = np.array(optimized_inputs).reshape(1, -1)
    if X.shape[1] != 10:
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    X = np.hstack([X, np.zeros((X.shape[0], 1))])  # 添加虚拟特征
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # 移动到指定 GPU
    
    with torch.no_grad():
        h = model.init_hidden(1)
        h = tuple([e.data.to(device) for e in h])
        prediction, _ = model(X_tensor, h)
    
    print(f"Predicted Output (od_600) for Optimized Inputs: {prediction.item()}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Optimize and predict using LSTM model.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model', type=str, default='lstm', help='Model type (default: lstm).')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension of the LSTM (default: 16).')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (default: 2).')
    parser.add_argument('--Timestamp', type=str, required=True, help='Timestamp to extract initial inputs from the dataset.')
    return parser.parse_args()

# 从数据集中提取指定时间戳的输入特征
def extract_initial_inputs(dataset_path, timestamp):
    df = pd.read_excel(dataset_path)
    print("Dataset columns:", df.columns.tolist())
    if 'Timestamp' not in df.columns:
        raise KeyError(f"Column 'Timestamp' not found in the dataset. Available columns: {df.columns.tolist()}")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S')
    row = df[df['Timestamp'] == timestamp]
    if row.empty:
        raise ValueError(f"No data found for timestamp: {timestamp}")
    feature_columns = ['m_ls_opt_do', 'm_temp', 'm_stirrer', 'dm_o2', 'dm_air', 
                       'dm_spump1', 'dm_spump2', 'dm_spump3', 'dm_spump4', 'induction']
    initial_inputs = row[feature_columns].values[0]
    return initial_inputs

# 运行优化和预测
if __name__ == "__main__":
    args = parse_args()
    model_path = args.weights
    scaler_path = f"{args.dataset}/scaler_new_data.save"
    input_dim = 11
    output_dim = 1
    timestamp = pd.to_datetime(args.Timestamp, format='%Y-%m-%dT%H:%M:%S')
    dataset_path = f"{args.dataset}/data.xlsx"
    initial_inputs = extract_initial_inputs(dataset_path, timestamp)
    optimized_inputs, best_value = optimize_inputs(model_path, scaler_path, input_dim, args.hidden_dim, output_dim, args.num_layers, initial_inputs)
    predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, args.hidden_dim, output_dim, args.num_layers)