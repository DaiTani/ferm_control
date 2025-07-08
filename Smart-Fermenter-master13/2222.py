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
    pass  # function body is omitted

# ---------------------- LSTM模型 ----------------------
class CompatibleLSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super(CompatibleLSTMPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.pre_fc(x)
        x = self.relu(x)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.relu(lstm_out)
        out = self.fc(lstm_out)
        return out, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    def __init__(self, model: nn.Module, scaler: object, dataset: pd.DataFrame, timestamp: float):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.timestamp = timestamp
        # 定义动作空间和观察空间，这里假设动作空间是11维连续空间，观察空间也是11维连续空间
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        # 初始化状态
        self.history = []

    def reset(self, seed=None, **kwargs):
        # 加载当前时间戳之前的数据
        # 假设这里进行了一些数据加载和处理
        initial_observation = np.random.rand(11)  # 这里使用随机值作为示例，实际需要根据数据加载情况计算
        info = {}  # 可以根据需要提供额外的信息
        self.history = []
        return initial_observation, info

    def step(self, action: np.ndarray):
        # 应用物理约束
        self.history.append(action)
        # 这里简单返回一个随机奖励和终止标志，实际需要根据具体逻辑计算
        reward = np.random.rand()
        done = False
        truncated = False
        info = {}
        next_observation = np.random.rand(11)  # 这里使用随机值作为示例，实际需要根据数据计算
        return next_observation, reward, done, truncated, info

    def predict_od600(self, steps: int = 72) -> List[float]:
        # 这里简单返回一个随机预测值列表，实际需要根据模型进行预测
        return [np.random.rand() for _ in range(steps)]

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler_path = scaler_path

    def load_data(self) -> pd.DataFrame:
        # 这里简单返回一个空DataFrame，实际需要根据数据加载逻辑实现
        return pd.DataFrame()

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

    # 记录优化前的参数
    initial_obs, _ = env.reset()
    initial_params = initial_obs[:11]
    # 预测优化前的 OD600
    initial_od600_preds = env.predict_od600()

    # PPO算法配置，调整学习率和增加训练步数
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # 调整学习率
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    # 训练过程
    print("\n[状态] 开始强化学习训练...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=EvalCallback(env, best_model_save_path=f"./models_{int(time.time())}/", n_eval_episodes=3),
        progress_bar=True
    )
    print(f"训练完成，耗时 {time.time()-start_time:.1f}秒")

    # 打印训练结果
    obs, _ = env.reset()
    optimized_action, _ = model_ppo.predict(obs, deterministic=True)

    # 预测优化后的 OD600
    # 先将环境状态更新为优化后的参数
    env.history[-1] = optimized_action
    optimized_od600_preds = env.predict_od600()

    # 计算10步内优化前后的 OD600 预测值的平均值
    initial_od600_avg = np.mean(initial_od600_preds[:20])
    optimized_od600_avg = np.mean(optimized_od600_preds[:20])

    print(f"\n10步内优化前的 OD600 预测值的平均值：{initial_od600_avg:.4f}")
    print(f"10步内优化后的 OD600 预测值的平均值：{optimized_od600_avg:.4f}")

    # 如果10步内优化后的 OD600 预测值的平均值小于等于优化前的，则使用初始参数
    if optimized_od600_avg == initial_od600_avg:
        optimized_action = initial_params

    # 参数对比显示
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

    print("\n优化前的 OD600 预测值（前 10 个时间步）：", initial_od600_preds[:20])
    print("优化后的 OD600 预测值（前 10 个时间步）：", optimized_od600_preds[:20])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument("--dataset", type=str, default="data", help="数据集路径")
    parser.add_argument("--Timestamp", type=str, default="0", help="时间戳")
    parser.add_argument("--weights", type=str, default="weights.pth", help="模型权重文件路径")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数")
    args = parser.parse_args()
    main(args)