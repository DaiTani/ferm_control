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
from typing import List, Tuple, Dict
import warnings
from torch import nn

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

    def __init__(self, model: nn.Module, scaler: object, dataset: pd.DataFrame, timestamp: float):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.window_size = 20
        self.dataset = dataset
        self.timestamp = timestamp

        # 参数的物理约束参考
        self.param_constraints = [
            (0, 3.0, 8.0),  # PH
            (1, 0.0, None),  # 溶氧
            (2, 0.0, None),  # 温度
            (5, 0.0, None),  # 空气流量
            (6, 0.0, None),  # 补料泵1
            (7, 0.0, None),  # 补料泵2
            (8, 0.0, None),  # 补料泵3
            (9, 0.0, None),  # 补料泵4
            (10, 0.0, 1.0)  # 诱导剂浓度
        ]

        initial_state = self.reset()[0]
        initial_params = initial_state[:11].copy()
        epsilon = 1e-6
        # 参数的优化范围参考
        param_bounds = [
            (max(3.0, initial_params[0] - 0.5), min(8.0, initial_params[0] + 0.5)),  # PH
            (max(epsilon, initial_params[1] - 1.0), initial_params[1] + 1.0),  # 溶氧
            (max(0.0, initial_params[2] - 1.0), initial_params[2] + 1.0),  # 温度
            (max(epsilon, initial_params[3] - 10.0), initial_params[3] + 10.0),  # 搅拌
            (max(epsilon, initial_params[4] - 1.0), initial_params[4] + 1.0),  # O2流量
            (max(epsilon, initial_params[5] - 1.0), initial_params[5] + 1.0),  # 空气流量
            (max(epsilon, initial_params[6] - 1.0), initial_params[6] + 1.0),  # 补料泵1
            (max(epsilon, initial_params[7] - 1.0), initial_params[7] + 1.0),  # 补料泵2
            (max(epsilon, initial_params[8] - 1.0), initial_params[8] + 1.0),  # 补料泵3
            (max(epsilon, initial_params[9] - 0.5), initial_params[9] + 0.5),  # 补料泵4
            (epsilon, 1.0 - epsilon)  # 诱导剂浓度
        ]

        # 确保下限小于等于上限
        self.param_bounds = [(min(low, high), max(low, high)) for low, high in param_bounds]

        # 初始化动作和观测空间
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in self.param_bounds], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),  # 11个参数
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, **kwargs):
        # 加载当前时间戳之前的数据
        target_time = self.timestamp
        df_filtered = self.dataset[self.dataset['Timestamp'] <= target_time]

        # 使用过滤后的数据作为历史数据
        initial_state = df_filtered.iloc[-1][1:12].values.astype(np.float32)
        self.history = df_filtered.iloc[:, 1:12].values[-self.window_size:]
        self.hidden = self.model.init_hidden()
        self.step_count = 0

        # 返回观测值和空的信息字典
        return initial_state, {}

    def step(self, action: np.ndarray):
        # 应用物理约束
        for idx, lower, upper in self.param_constraints:
            if upper is not None:
                action[idx] = np.clip(action[idx], lower, upper)
            else:
                action[idx] = np.maximum(action[idx], lower)

        # 将动作剪切至优化范围
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.history = np.append(self.history[1:], [clipped_action], axis=0)

        # 构建输入序列
        input_seq = self.history[-self.window_size:]

        # 数据预处理
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

        # 使用模型预测
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)

        reward = pred.item()
        done = self.step_count >= 1
        self.step_count += 1

        return clipped_action, reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        preds = []
        current_state = self.history[-1].copy()
        hidden = self.model.init_hidden(batch_size=1)

        for _ in range(steps):
            # 预测下一步
            input_seq = np.array([current_state] * self.window_size)
            scaled_seq = self.scaler.transform(input_seq)
            tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, hidden = self.model(tensor_input, hidden)

            preds.append(pred.item())
            current_state = np.roll(current_state, -1)
            current_state[-1] = pred.item()

        return preds

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))

        # 数据验证
        df['Timestamp'] = df['Timestamp'].astype(float)
        return df

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "=" * 40)
    print(" 生物过程优化系统 v8.5")
    print("=" * 40)

    # 数据加载
    data_loader = SafeDataLoader(
        args.dataset,
        args.Timestamp,
        os.path.join(args.dataset, "scaler_new_data.save")
    )

    dataset = data_loader.load_data()

    # 加载LSTM模型
    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['weights'], strict=True)
    print("✅ 模型加载成功")

    # 创建训练环境
    env = FermentationEnv(model, data_loader.scaler, dataset, float(args.Timestamp))

    # PPO算法配置
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    # 开始训练
    print("\n[状态] 开始强化学习训练...")
    model_ppo.learn(total_timesteps=10000)

    # 打印训练结果
    obs, _ = env.reset()
    action, _ = model_ppo.predict(obs, deterministic=True)

    print("优化后的参数：", action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument('--weights', type=str, required=True, help="模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=2, help="LSTM层数")
    parser.add_argument('--Timestamp', type=str, required=True, help="初始时间戳（格式：小时:分钟）")

    args = parser.parse_args()
    main(args)