import torch
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import joblib
from dataset import FermentationData  # 假设 dataset.py 中有 FermentationData 类
from model import LSTMPredictor  # 假设 model.py 中有 LSTMPredictor 类
import argparse
import os
import random
import warnings
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示进度条
import time

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
def objective_function(X, model, scaler, dataset_path):
    start_time = time.time()
    # 设置随机种子
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # 加载测试数据集（仅加载一次）
    if not hasattr(objective_function, 'test_dataset'):
        objective_function.test_dataset = FermentationData(work_dir=dataset_path, train_mode=False, y_var=["od_600"])
        objective_function.test_loader = DataLoader(dataset=objective_function.test_dataset, batch_size=1, num_workers=2, shuffle=False)

    test_dataset = objective_function.test_dataset
    test_loader = objective_function.test_loader

    # 测试函数
    def test(model, testloader):
        model.eval()
        loss = 0
        err = 0

        # 初始化存储预测值和标签的数组
        preds = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        labels = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        n_overlap = np.zeros(len(test_dataset) + test_dataset.ws - 1)

        N = 10  # 滑动平均的窗口大小
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(testloader):
                batch_size = input.size(0)

                # 初始化隐藏状态
                h = model.init_hidden(batch_size)
                h = tuple([e.data.to(device) for e in h])

                input, label = input.to(device), label.to(device)

                # 模型预测
                output, h = model(input.float(), h)

                # 对输出进行滑动平均处理
                y = output.view(-1).cpu().numpy()
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")

                # 将预测值和标签累加到重叠区域
                preds[batch_idx : (batch_idx + test_dataset.ws)] += y_smooth
                labels[batch_idx : (batch_idx + test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx : (batch_idx + test_dataset.ws)] += 1.0

                # 计算损失和误差
                loss += torch.nn.functional.mse_loss(output, label.float())
                err += torch.sqrt(torch.nn.functional.mse_loss(output, label.float())).item()

        # 计算平均损失和误差
        loss = loss / len(test_dataset)
        err = err / len(test_dataset)

        # 对重叠区域的预测值和标签取平均
        preds /= n_overlap
        labels /= n_overlap

        return err, preds, labels

    # 运行测试
    err, preds, labels = test(model, test_loader)

    # 计算 RMSE 和 REFY
    mse = np.square(np.subtract(preds, labels)).mean()
    rmse = math.sqrt(mse)
    refy = abs(preds[-1] - labels[-1]) / labels[-1] * 100

    # 打印执行时间
    end_time = time.time()
    print(f"Objective function execution time: {end_time - start_time} seconds")

    # 返回负的 RMSE（因为遗传算法默认是最小化目标函数）
    return -rmse

# 遗传算法优化
def optimize_inputs(model_path, scaler_path, input_dim, hidden_dim, output_dim, n_layers, initial_inputs, dataset_path):
    model = load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
    scaler = joblib.load(scaler_path)

    # 动态调整输入特征的边界，确保所有特征为非负
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
        'max_num_iteration': 50,  # 增加总迭代次数
        'population_size': 25,    # 增加种群大小
        'mutation_probability': 0.2,  # 增加变异概率
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50  # 增加无改进迭代次数
    }

    # 自定义回调函数，用于输出每一步的结果
    def on_generation(ga_instance):
        current_iteration = ga_instance.generations_completed
        best_solution = ga_instance.best_variable
        best_value = -ga_instance.best_function  # 因为目标函数返回的是负的 RMSE
        print(f"Iteration {current_iteration}: Best RMSE = {best_value}, Best Solution = {best_solution}")

    # 初始化遗传算法
    model_ga = ga(
        function=lambda X: objective_function(X, model, scaler, dataset_path),
        dimension=len(varbound),
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        progress_bar=True,
        function_timeout=1000
    )

    # 添加回调函数
    model_ga.on_generation = on_generation

    # 运行遗传算法
    model_ga.run()

    # 输出最优解
    best_solution = model_ga.output_dict['variable']
    best_value = -model_ga.output_dict['function']
    print("Optimal Inputs:", best_solution)
    print("Minimum RMSE:", best_value)

    return best_solution, best_value

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
            timestamp = pd.to_datetime(timestamp, format='%d-%m-%Y %H:%M:%S', dayfirst=True)
        except ValueError:
            try:
                # 尝试解析第三种格式：2022-11-16 09:10:22
                timestamp = pd.to_datetime(timestamp, format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # 尝试解析第四种格式：16-11-2022 09:10:22
                    timestamp = pd.to_datetime(timestamp, format='%d-%m-%Y %H:%M:%S', dayfirst=True)
                except ValueError:
                    raise ValueError(f"Timestamp format not recognized: {timestamp}")
    
    # 将数据集中的 'Timestamp' 列转换为 datetime 类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
    
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

# 运行优化和预测
if __name__ == "__main__":
    args = parse_args()
    model_path = args.weights
    scaler_path = f"{args.dataset}/scaler_new_data.save"
    input_dim = 11
    output_dim = 1
    dataset_path = f"{args.dataset}/data.xlsx"
    
    # 提取初始输入特征
    try:
        initial_inputs = extract_initial_inputs(dataset_path, args.Timestamp)
    except Exception as e:
        print(f"Error extracting initial inputs: {e}")
        exit(1)
    
    # 运行优化和预测
    optimized_inputs, best_value = optimize_inputs(
        model_path, scaler_path, input_dim, args.hidden_dim, output_dim, args.num_layers, initial_inputs, args.dataset
    )
    print(f"Optimized Inputs: {optimized_inputs}")
    print(f"Best RMSE: {best_value}")