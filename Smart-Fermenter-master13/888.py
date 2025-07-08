import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import os
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import List, Tuple, Dict
import warnings
from torch import nn
import json

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- 工具函数 ----------------------
def format_timestamp(minutes: float) -> str:
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    seconds = int((minutes - int(minutes)) * 60)
    return f"{hours}小时{remaining_minutes}分{seconds}秒"

# ---------------------- LSTM模型 ----------------------
class CompatibleLSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.act(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: nn.Module, scaler: object, dataset: pd.DataFrame, timestamp: float, y_mean: float, y_std: float):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.y_mean = y_mean
        self.y_std = y_std
        self.window_size = 20
        self.dataset = dataset
        self.timestamp = timestamp

        # 参数的物理约束参考（优化后）
        self.param_constraints = [
            (0, 0.0, None),  # 空气流量
            (1, 0.0, None),  # 溶氧
            (2, 3.0, 8.0),   # PH
            (3, 0.0, None),  # 搅拌
            (4, 0.0, None),  # 温度
            (5, 0.0, None),  # 氧气流量
            (6, 0.0, None),  # 补料泵1
            (7, 0.0, None),  # 补料泵2
            (8, 0.0, None),  # 补料泵3
            (9, 0.0, None),  # 补料泵4
            (10, 0.0, 1.0)   # 诱导剂浓度（确保不越界）
        ]

        initial_state = self.reset()[0]
        initial_params = initial_state[:11].copy()
        epsilon = 1e-6

        # 调整参数优化范围（避免诱导剂浓度突变）
        param_bounds = [
            (max(epsilon, initial_params[0] - 50.0), initial_params[0] + 200.0),
            (max(epsilon, initial_params[1] - 10.0), initial_params[1] + 50.0),
            (max(3.0, initial_params[2] - 0.5), min(8.0, initial_params[2] + 0.5)),
            (max(epsilon, initial_params[3] - 50.0), initial_params[3] + 200.0),
            (max(initial_params[4] - 1.0, 0.0), initial_params[4] + 1.0),
            (max(epsilon, initial_params[5] - 50.0), initial_params[5] + 200.0),
            (max(epsilon, initial_params[6] - 5.0), initial_params[6] + 5.0),
            (max(epsilon, initial_params[7] - 5.0), initial_params[7] + 5.0),
            (max(epsilon, initial_params[8] - 5.0), initial_params[8] + 5.0),
            (max(epsilon, initial_params[9] - 5.0), initial_params[9] + 5.0),
            (max(0.1, initial_params[10] - 0.1), min(0.9, initial_params[10] + 0.1))  # 限制诱导剂浓度调整幅度
        ]

        self.param_bounds = [(min(low, high), max(low, high)) for low, high in param_bounds]

        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in self.param_bounds], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, **kwargs):
        target_time = self.timestamp
        df_filtered = self.dataset[self.dataset['Timestamp'] <= target_time]

        if df_filtered.empty:
            raise ValueError(f"No data available before timestamp {target_time}")

        initial_state = df_filtered.iloc[-1][1:12].values.astype(np.float32)
        self.history = df_filtered.iloc[:, 1:12].values[-self.window_size:]
        self.hidden = self.model.init_hidden()
        self.step_count = 0

        return initial_state, {}

    def step(self, action: np.ndarray):
        # 硬约束修正
        for idx, lower, upper in self.param_constraints:
            if upper is not None:
                action[idx] = np.clip(action[idx], lower, upper)
            else:
                action[idx] = np.maximum(action[idx], lower)

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.history = np.append(self.history[1:], [clipped_action], axis=0)

        input_seq = self.history[-self.window_size:]
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)

        pred_denorm = (pred.item() * self.y_std) + self.y_mean

        # 奖励函数优化：增加OD600权重，减少惩罚系数
        prev_action = self.history[-2]
        param_change = np.linalg.norm(clipped_action - prev_action)
        penalty = 0.01 * param_change  # 原0.05改为0.01
        reward = pred_denorm * 10 - penalty  # 放大OD600影响

        done = self.step_count >= 72
        self.step_count += 1

        return clipped_action, reward, done, False, {}

    def predict_od600(self, steps: int = 2) -> List[float]:
        preds = []
        current_state = self.history[-1].copy()
        hidden = self.model.init_hidden(batch_size=1)

        for _ in range(steps):
            input_seq = np.array([current_state] * self.window_size)
            scaled_seq = self.scaler.transform(input_seq)
            tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, hidden = self.model(tensor_input, hidden)

            pred_denorm = (pred.item() * self.y_std) + self.y_mean
            preds.append(pred_denorm)
            current_state = np.roll(current_state, -1)
            current_state[-1] = pred_denorm

        return preds

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.scaler = joblib.load(scaler_path)
        
        norm_file_path = os.path.join(dataset_path, "norm_file.json")
        with open(norm_file_path, 'r') as f:
            norm_data = json.load(f)
        self.y_mean = norm_data['mean']
        self.y_std = norm_data['std']

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))
        df['Timestamp'] = df['Timestamp'].astype(float)
        # 保存列名并转换为numpy避免特征名警告
        self.feature_columns = df.columns[1:12].tolist()
        return df

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "=" * 40)
    print(" 生物过程优化系统 v8.5 (修正版)")
    print("=" * 40)

    data_loader = SafeDataLoader(
        args.dataset,
        os.path.join(args.dataset, "scaler_new_data.save")
    )

    dataset = data_loader.load_data()

    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['weights'], strict=True)
    print("✅ 模型加载成功")

    env = FermentationEnv(model, data_loader.scaler, dataset, float(args.Timestamp), data_loader.y_mean, data_loader.y_std)
    monitored_env = Monitor(env)

    # 创建独立的评估环境
    eval_env = Monitor(FermentationEnv(model, data_loader.scaler, dataset, float(args.Timestamp), data_loader.y_mean, data_loader.y_std))

    initial_obs, _ = env.reset()
    initial_params = initial_obs[:11]
    initial_od600_preds = env.predict_od600()
    print("✅ 环境加载成功")
    # 优化PPO超参数
    model_ppo = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    print("\n[状态] 开始强化学习训练...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=EvalCallback(eval_env, best_model_save_path=f"./models_{int(time.time())}/", n_eval_episodes=5),
        progress_bar=True
    )
    print(f"训练完成，耗时 {time.time()-start_time:.1f}秒")

    obs, _ = env.reset()
    optimized_action, _ = model_ppo.predict(obs, deterministic=True)

    env.history[-1] = optimized_action
    optimized_od600_preds = env.predict_od600()

    initial_od600_avg = np.mean(initial_od600_preds[:20])
    optimized_od600_avg = np.mean(optimized_od600_preds[:20])

    print(f"\n10步内优化前的 OD600 预测值的平均值：{initial_od600_avg:.4f}")
    print(f"10步内优化后的 OD600 预测值的平均值：{optimized_od600_avg:.4f}")

    if optimized_od600_avg == initial_od600_avg:
        optimized_action = initial_params

    param_names = [
        "空气流量(L/h)", "溶氧(%)", "PH", "搅拌(rpm)", "温度(℃)", "O2流量(L/h)",
        "补料泵1", "补料泵2", "补料泵3", "补料泵4", "诱导剂浓度"
    ]

    print("\n{:<15} {:<15} {:<15} {:<10}".format("参数", "原始值", "优化值", "变化率"))
    print("-" * 60)
    for i in range(11):
        orig = initial_params[i]
        optim = optimized_action[i]
        change = (optim - orig) / orig * 100 if orig != 0 else np.nan
        print(f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {change:+.1f}%" if not np.isnan(change) else
              f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {'N/A':<10}")
        
    print("\n优化前的 OD600 预测值（前 20 个时间步）：", initial_od600_preds[:20])
    print("优化后的 OD600 预测值（前 20 个时间步）：", optimized_od600_preds[:20])

    # 修复特征名警告：使用numpy数据保存到Excel
    original_data = dataset.copy()
    normalized_data = pd.DataFrame(
        data_loader.scaler.transform(dataset.iloc[:, 1:12].values),
        columns=data_loader.feature_columns
    )
    denormalized_data = pd.DataFrame(
        data_loader.scaler.inverse_transform(normalized_data),
        columns=data_loader.feature_columns
    )

    with pd.ExcelWriter(os.path.join(args.dataset, "transform.xlsx")) as writer:
        original_data.to_excel(writer, sheet_name='Original Data', index=False)
        normalized_data.to_excel(writer, sheet_name='Normalized Data', index=False)
        denormalized_data.to_excel(writer, sheet_name='Denormalized Data', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument('--weights', type=str, required=True, help="模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=2, help="LSTM层数")
    parser.add_argument('--Timestamp', type=str, required=True, help="初始时间戳（格式：小时:分钟）")

    args = parser.parse_args()
    main(args)