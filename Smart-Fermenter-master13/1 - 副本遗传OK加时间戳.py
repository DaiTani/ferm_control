import torch
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import joblib
from dataset import FermentationData  # 假设 dataset.py 中有 FermentationData 类
from model import LSTMPredictor  # 假设 model.py 中有 LSTMPredictor 类
import argparse
from datetime import datetime

# 设置设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
def load_model(model_path, input_dim, hidden_dim, output_dim, n_layers):
    # 初始化模型
    model = LSTMPredictor(
        input_dim=input_dim,  # 输入特征的数量
        hidden_dim=hidden_dim,  # 与训练时一致
        output_dim=output_dim,  # 输出维度
        n_layers=n_layers  # 与训练时一致
    )
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)  # 映射到当前设备
    model.load_state_dict(checkpoint['weights'])
    model.to(device)  # 将模型移动到当前设备
    model.eval()  # 设置为评估模式
    return model

# 定义目标函数
def objective_function(X, model, scaler):
    """
    目标函数：根据输入特征预测输出，并返回预测值。
    """
    # 将输入转换为合适的形状
    X = np.array(X).reshape(1, -1)
    if X.shape[1] != 10:  # 检查特征数量
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    
    # 添加一个虚拟特征（全零列），扩展输入维度为 11
    X = np.hstack([X, np.zeros((X.shape[0], 1))])
    
    X_scaled = scaler.transform(X)  # 标准化输入
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为 PyTorch 张量并移动到 GPU

    # 使用模型进行预测
    with torch.no_grad():
        h = model.init_hidden(1)  # 初始化隐藏状态
        h = tuple([e.data.to(device) for e in h])  # 将隐藏状态移动到 GPU
        prediction, _ = model(X_tensor, h)

    return -prediction.item()  # 返回负值，因为遗传算法默认是最小化目标函数

# 遗传算法优化
def optimize_inputs(model_path, scaler_path, input_dim, hidden_dim, output_dim, n_layers, initial_inputs):
    # 加载模型和标准化器
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 定义输入特征的边界
    varbound = np.array([
        [0, 14],  # m_ls_opt_do
        [20, 40],  # m_temp
        [0, 1200],  # m_stirrer
        [0, 100],  # dm_o2
        [0, 100],  # dm_air
        [0, 100],  # dm_spump1
        [0, 100],  # dm_spump2
        [0, 100],  # dm_spump3
        [0, 100],  # dm_spump4
        [0, 1]  # induction (二进制变量)
    ])

    # 设置遗传算法参数
    algorithm_param = {
        'max_num_iteration': 500,  # 最大迭代次数
        'population_size': 100,  # 种群大小
        'mutation_probability': 0.1,  # 变异概率
        'elit_ratio': 0.01,  # 精英比例
        'crossover_probability': 0.5,  # 交叉概率
        'parents_portion': 0.3,  # 父代比例
        'crossover_type': 'uniform',  # 交叉类型
        'max_iteration_without_improv': None  # 无改进时的最大迭代次数
    }

    # 初始化遗传算法
    model_ga = ga(
        function=lambda X: objective_function(X, model, scaler),
        dimension=len(varbound),  # 输入特征的维度
        variable_type='real',  # 变量类型
        variable_boundaries=varbound,  # 变量边界
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        progress_bar=True
    )

    # 运行遗传算法
    model_ga.run()

    # 输出最优解
    best_solution = model_ga.output_dict['variable']
    best_value = -model_ga.output_dict['function']  # 因为目标函数返回的是负值

    print("Optimal Inputs:")
    print(f"m_ls_opt_do: {best_solution[0]}")
    print(f"m_temp: {best_solution[1]}")
    print(f"m_stirrer: {best_solution[2]}")
    print(f"dm_o2: {best_solution[3]}")
    print(f"dm_air: {best_solution[4]}")
    print(f"dm_spump1: {best_solution[5]}")
    print(f"dm_spump2: {best_solution[6]}")
    print(f"dm_spump3: {best_solution[7]}")
    print(f"dm_spump4: {best_solution[8]}")
    print(f"induction: {best_solution[9]}")
    print(f"Maximum Predicted Output (od_600): {best_value}")

    return best_solution, best_value

# 使用优化后的输入特征进行预测
def predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, hidden_dim, output_dim, n_layers):
    # 加载模型和标准化器
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 将优化后的输入特征转换为合适的形状
    X = np.array(optimized_inputs).reshape(1, -1)
    if X.shape[1] != 10:  # 检查特征数量
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    
    # 添加一个虚拟特征（全零列），扩展输入维度为 11
    X = np.hstack([X, np.zeros((X.shape[0], 1))])
    
    X_scaled = scaler.transform(X)  # 标准化输入
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # 转换为 PyTorch 张量并移动到 GPU

    # 使用模型进行预测
    with torch.no_grad():
        h = model.init_hidden(1)  # 初始化隐藏状态
        h = tuple([e.data.to(device) for e in h])  # 将隐藏状态移动到 GPU
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
    # 加载数据集
    df = pd.read_excel(dataset_path)  # 加载 Excel 文件
    
    # 打印数据集列名，用于调试
    print("Dataset columns:", df.columns.tolist())
    
    # 检查时间戳列是否存在
    if 'Timestamp' not in df.columns:  # 替换为实际的时间戳列名
        raise KeyError(f"Column 'Timestamp' not found in the dataset. Available columns: {df.columns.tolist()}")
    
    # 将时间戳列转换为 datetime 类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%dT%H:%M:%S')  # 替换为实际的时间戳格式
    
    # 查找指定时间戳的行
    row = df[df['Timestamp'] == timestamp]  # 替换为实际的时间戳列名
    
    if row.empty:
        raise ValueError(f"No data found for timestamp: {timestamp}")
    
    # 提取输入特征
    feature_columns = ['m_ls_opt_do', 'm_temp', 'm_stirrer', 'dm_o2', 'dm_air', 
                       'dm_spump1', 'dm_spump2', 'dm_spump3', 'dm_spump4', 'induction']
    initial_inputs = row[feature_columns].values[0]
    
    return initial_inputs

# 运行优化和预测
if __name__ == "__main__":
    args = parse_args()

    # 模型和标准化器路径
    model_path = args.weights
    scaler_path = f"{args.dataset}/scaler_new_data.save"  # 假设标准化器已保存

    # 输入特征的维度
    input_dim = 11  # 输入特征的数量（模型权重需要 11 个特征）
    output_dim = 1  # 输出维度

    # 从数据集中提取指定时间戳的输入特征
    timestamp = pd.to_datetime(args.Timestamp, format='%Y-%m-%dT%H:%M:%S')  # 替换为实际的时间戳格式
    dataset_path = f"{args.dataset}/data.xlsx"  # 测试数据集路径
    initial_inputs = extract_initial_inputs(dataset_path, timestamp)

    # 运行遗传算法优化
    optimized_inputs, best_value = optimize_inputs(model_path, scaler_path, input_dim, args.hidden_dim, output_dim, args.num_layers, initial_inputs)

    # 使用优化后的输入特征进行预测
    predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, args.hidden_dim, output_dim, args.num_layers)