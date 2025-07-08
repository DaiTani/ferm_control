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

        # 新参数顺序索引映射 [原索引]
        self.new_param_order = [2, 1, 4, 3, 5, 0, 6, 7, 8, 9, 10]

        # 参数的物理约束参考（按新顺序）
        self.param_constraints = [
            (0, 3.0, 8.0),   # m_ph
            (1, 0.0, None), # m_ls_opt_do
            (2, 0.0, 38.0), # m_temp
            (3, 0.0, None), # m_stirrer
            (4, 0.0, None), # dm_o2
            (5, 0.0, None), # dm_air
            (6, 0.0, None), # dm_spump1
            (7, 0.0, None), # dm_spump2
            (8, 0.0, None), # dm_spump3
            (9, 0.0, None), # dm_spump4
            (10, 0.0, 1.0)  # induction
        ]

        initial_state = self._reorder_params(self.reset()[0])  # 初始状态按新顺序排列
        initial_params = initial_state.copy()
        epsilon = 1e-6

        # 参数的优化范围参考（按新顺序）
        param_bounds = [
            (max(3.0, initial_params[0] - 1.0), min(8.0, initial_params[0] + 1.0)),  # m_ph
            (max(epsilon, initial_params[1] - 2.0), initial_params[1] + 2.0),       # m_ls_opt_do
            (max(0.0, initial_params[2] - 2.0), initial_params[2] + 2.0),           # m_temp
            (max(epsilon, initial_params[3] - 20.0), initial_params[3] + 20.0),     # m_stirrer
            (max(epsilon, initial_params[4] - 2.0), initial_params[4] + 2.0),       # dm_o2
            (max(epsilon, initial_params[5] - 2.0), initial_params[5] + 2.0),       # dm_air
            (max(epsilon, initial_params[6] - 2.0), initial_params[6] + 2.0),       # dm_spump1
            (max(epsilon, initial_params[7] - 2.0), initial_params[7] + 2.0),       # dm_spump2
            (max(epsilon, initial_params[8] - 2.0), initial_params[8] + 2.0),       # dm_spump3
            (max(epsilon, initial_params[9] - 1.0), initial_params[9] + 1.0),       # dm_spump4
            (epsilon, 1.0 - epsilon)                                                # induction
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

    def _reorder_params(self, params: np.ndarray) -> np.ndarray:
        """将原始数据参数按新顺序重新排列"""
        return params[self.new_param_order]

    def reset(self, seed=None, **kwargs):
        target_time = self.timestamp
        df_filtered = self.dataset[self.dataset['Timestamp'] <= target_time]

        raw_state = df_filtered.iloc[-1][1:12].values.astype(np.float32)
        initial_state = self._reorder_params(raw_state)  # 按新顺序排列初始状态
        self.history = self._reorder_params(df_filtered.iloc[:, 1:12].values[-self.window_size:])
        self.hidden = self.model.init_hidden()
        self.step_count = 0

        return initial_state, {}

    def step(self, action: np.ndarray):
        # 应用物理约束
        for idx, lower, upper in self.param_constraints:
            if upper is not None:
                action[idx] = np.clip(action[idx], lower, upper)
            else:
                action[idx] = np.maximum(action[idx], lower)

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.history = np.append(self.history[1:], [clipped_action], axis=0)

        # 转换为原始顺序进行模型预测
        input_seq = np.array([self._revert_order(x) for x in self.history[-self.window_size:]])
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)

        pred_denorm = (pred.item() * self.y_std) + self.y_mean

        prev_action = self.history[-2]
        param_change = np.linalg.norm(clipped_action - prev_action)
        penalty = 0.05 * param_change

        reward = pred_denorm - penalty
        done = self.step_count >= 72
        self.step_count += 1

        return clipped_action, reward, done, False, {}

    def _revert_order(self, params: np.ndarray) -> np.ndarray:
        """将参数还原为原始顺序供模型使用"""
        reverted = np.zeros_like(params)
        for new_idx, old_idx in enumerate(self.new_param_order):
            reverted[old_idx] = params[new_idx]
        return reverted

    def predict_od600(self, steps: int = 72) -> List[float]:
        preds = []
        current_state = self.history[-1].copy()
        hidden = self.model.init_hidden(batch_size=1)

        for _ in range(steps):
            input_seq = np.array([self._revert_order(current_state)] * self.window_size)
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
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)
        
        # 加载标签归一化参数
        norm_file_path = os.path.join(dataset_path, "norm_file.json")
        with open(norm_file_path, 'r') as f:
            norm_data = json.load(f)
        self.y_mean = norm_data['mean']
        self.y_std = norm_data['std']

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))
        df['Timestamp'] = df['Timestamp'].astype(float)
        return df

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "=" * 40)
    print(" 生物过程优化系统 v8.5")
    print("=" * 40)

    data_loader = SafeDataLoader(
        args.dataset,
        args.Timestamp,
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

    initial_obs, _ = env.reset()
    initial_params = initial_obs.copy()
    initial_od600_preds = env.predict_od600()

    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    print("\n[状态] 开始强化学习训练...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=EvalCallback(env, best_model_save_path=f"./models_{int(time.time())}/", n_eval_episodes=3),
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
        "m_ph", "m_ls_opt_do", "m_temp", "m_stirrer", "dm_o2",
        "dm_air", "dm_spump1", "dm_spump2", "dm_spump3", "dm_spump4", "induction"
    ]

    print("\n{:<15} {:<15} {:<15} {:<10}".format("参数", "原始值", "优化值", "变化率"))
    print("-" * 60)
    for i in range(11):
        orig = initial_params[i]
        optim = optimized_action[i]
        change = (optim - orig) / orig * 100 if orig != 0 else np.nan
        print(f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {change:+.1f}%" if not np.isnan(change) else
              f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {'N/A':<10}")

    # 输出转换数据（保持原始顺序）
    original_data = dataset.copy()
    normalized_data = pd.DataFrame(data_loader.scaler.transform(dataset.iloc[:, 1:12]), columns=dataset.columns[1:12])
    denormalized_data = pd.DataFrame(data_loader.scaler.inverse_transform(normalized_data), columns=dataset.columns[1:12])

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