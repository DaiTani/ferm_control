import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
from dataset import *
import pdb
import warnings
from model import *
import random
import utils
import math
import pandas as pd
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
import joblib
from datetime import datetime

# 动态设置设备
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 使用第一个 GPU
else:
    device = torch.device('cpu')

# 加载训练好的模型
def load_model(model_path, input_dim, hidden_dim, output_dim, n_layers):
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)
    checkpoint = torch.load(model_path, map_location=device)  # 确保加载到正确的设备
    model.load_state_dict(checkpoint['weights'])
    model.to(device)
    model.eval()
    return model

# 定义目标函数
def objective_function(X, model, scaler, test_dataset):
    """
    目标函数：根据输入特征计算 OD600 的预测值，并返回误差（RMSE 或 REFY）。
    :param X: 输入特征（1D 数组，长度为 10）
    :param model: 训练好的 LSTMPredictor 模型
    :param scaler: 标准化器
    :param test_dataset: 测试数据集对象
    :return: 误差值（RMSE 或 REFY）
    """
    # 将输入特征转换为模型所需的格式
    X = np.array(X).reshape(1, -1)
    if X.shape[1] != 10:
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    X = np.hstack([X, np.zeros((X.shape[0], 1))])  # 添加虚拟特征
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        h = model.init_hidden(1)
        h = tuple([e.data.to(device) for e in h])
        prediction, _ = model(X_tensor, h)

    # 将预测值和真实值存储
    preds = prediction.view(-1).cpu().numpy()
    labels = test_dataset.get_labels()  # 假设 test_dataset 有一个方法可以获取真实标签

    # 计算 RMSE
    mse = np.square(np.subtract(preds, labels)).mean()
    rmse = math.sqrt(mse)

    # 计算 REFY（相对最终产量误差）
    refy = abs(preds[-1] - labels[-1]) / labels[-1] * 100

    # 返回误差值（可以根据需要选择 RMSE 或 REFY）
    return rmse  # 或者 return refy

# 遗传算法优化
def optimize_inputs(model_path, scaler_path, input_dim, hidden_dim, output_dim, n_layers, initial_inputs, test_dataset):
    """
    使用遗传算法优化输入特征。
    :param model_path: 模型权重文件路径
    :param scaler_path: 标准化器文件路径
    :param input_dim: 输入特征维度
    :param hidden_dim: LSTM 隐藏层维度
    :param output_dim: 输出维度
    :param n_layers: LSTM 层数
    :param initial_inputs: 初始输入特征
    :param test_dataset: 测试数据集对象
    :return: 最优输入特征和对应的误差值
    """
    # 加载模型和标准化器
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 动态调整输入特征的边界
    varbound = np.array([
        [max(0, initial_inputs[0] - 10), initial_inputs[0] + 10],  # m_ls_opt_do ±10
        [max(0, initial_inputs[1] - 1), initial_inputs[1] + 1],    # m_temp ±1
        [max(0, initial_inputs[2] - 100), initial_inputs[2] + 100],  # m_stirrer ±100
        [max(0, initial_inputs[3] - 100), initial_inputs[3] + 100],  # dm_o2 ±100
        [max(0, initial_inputs[4] - 100), initial_inputs[4] + 100],  # dm_air ±100
        [max(0, initial_inputs[5] - 1), initial_inputs[5] + 1],      # dm_spump1 ±1
        [max(0, initial_inputs[6] - 1), initial_inputs[6] + 1],      # dm_spump2 ±1
        [max(0, initial_inputs[7] - 1), initial_inputs[7] + 1],      # dm_spump3 ±1
        [max(0, initial_inputs[8] - 5), initial_inputs[8] + 5],      # dm_spump4 ±5
        [max(0, initial_inputs[9] - 0.1), initial_inputs[9] + 0.1]   # induction ±0.1
    ])

    # 设置遗传算法参数
    algorithm_param = {
        'max_num_iteration': 500,  # 最大迭代次数
        'population_size': 100,    # 种群大小
        'mutation_probability': 0.2,  # 变异概率
        'elit_ratio': 0.01,        # 精英比例
        'crossover_probability': 0.5,  # 交叉概率
        'parents_portion': 0.3,    # 父代比例
        'crossover_type': 'uniform',  # 交叉类型
        'max_iteration_without_improv': 50  # 无改进时的最大迭代次数
    }

    # 初始化遗传算法
    model_ga = ga(
        function=lambda X: objective_function(X, model, scaler, test_dataset),
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
    best_value = model_ga.output_dict['function']
    print("Optimal Inputs:", best_solution)
    print("Minimum Error (RMSE or REFY):", best_value)

    return best_solution, best_value

# 使用优化后的输入特征进行预测
def predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, hidden_dim, output_dim, n_layers):
    """
    使用优化后的输入特征进行预测。
    :param model_path: 模型权重文件路径
    :param scaler_path: 标准化器文件路径
    :param optimized_inputs: 优化后的输入特征
    :param input_dim: 输入特征维度
    :param hidden_dim: LSTM 隐藏层维度
    :param output_dim: 输出维度
    :param n_layers: LSTM 层数
    """
    # 加载模型和标准化器
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 将优化后的输入特征转换为模型所需的格式
    X = np.array(optimized_inputs).reshape(1, -1)
    if X.shape[1] != 10:
        raise ValueError(f"Expected 10 features, but got {X.shape[1]} features.")
    X = np.hstack([X, np.zeros((X.shape[0], 1))])  # 添加虚拟特征
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        h = model.init_hidden(1)
        h = tuple([e.data.to(device) for e in h])
        prediction, _ = model(X_tensor, h)
    
    # 输出预测结果
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
    """
    从数据集中提取指定时间戳的输入特征。
    支持多种时间戳格式：
    - 2021-07-14T16:02:43
    - 14-07-2021 16:02:43
    - 2022-11-16 09:10:22
    - 16-11-2022 09:10:22
    """
    # 读取数据集
    df = pd.read_excel(dataset_path)
    
    # 确保数据集中包含 'Timestamp' 列
    if 'Timestamp' not in df.columns:
        raise KeyError(f"Column 'Timestamp' not found in the dataset. Available columns: {df.columns.tolist()}")
    
    # 尝试解析时间戳
    try:
        # 尝试解析第一种格式：2021-07-14T16:02:43
        timestamp = pd.to_datetime(timestamp, format='%Y-%m-%dT%H:%M:%S')
    except ValueError:
        try:
            # 尝试解析第二种格式：14-07-2021 16:02:43
            timestamp = pd.to_datetime(timestamp, format='%d-%m-%Y %H:%M:%S')
        except ValueError:
            try:
                # 尝试解析第三种格式：2022-11-16 09:10:22
                timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # 尝试解析第四种格式：16-11-2022 09:10:22
                    timestamp = pd.to_datetime(timestamp, format='%d-%m-%Y %H:%M:%S')
                except ValueError:
                    raise ValueError(f"Timestamp format not recognized: {timestamp}")
    
    # 将数据集中的 'Timestamp' 列转换为 datetime 类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # 查找指定时间戳对应的行
    row = df[df['Timestamp'] == timestamp]
    if row.empty:
        raise ValueError(f"No data found for timestamp: {timestamp}")
    
    # 提取输入特征
    feature_columns = [
        'm_ls_opt_do', 'm_temp', 'm_stirrer', 'dm_o2', 'dm_air', 
        'dm_spump1', 'dm_spump2', 'dm_spump3', 'dm_spump4', 'induction'
    ]
    initial_inputs = row[feature_columns].values[0]
    
    return initial_inputs

# 主函数
if __name__ == "__main__":
    args = parse_args()
    model_path = args.weights
    scaler_path = f"{args.dataset}/scaler_new_data.save"
    input_dim = 11
    output_dim = 1
    timestamp = args.Timestamp
    dataset_path = f"{args.dataset}/data.xlsx"
    initial_inputs = extract_initial_inputs(dataset_path, timestamp)

    # 加载测试数据集
    test_dataset = FermentationData(work_dir=args.dataset, train_mode=False, y_var=["od_600"])

    # 优化输入特征
    optimized_inputs, best_value = optimize_inputs(
        model_path, scaler_path, input_dim, args.hidden_dim, output_dim, args.num_layers, initial_inputs, test_dataset
    )

    # 使用优化后的输入特征进行预测
    predict_with_optimized_inputs(model_path, scaler_path, optimized_inputs, input_dim, args.hidden_dim, output_dim, args.num_layers)