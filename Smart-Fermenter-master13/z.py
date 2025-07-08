import torch
from torch.utils.data import Dataset
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 假设的 utils 模块，需要根据实际情况实现
class utils:
    @staticmethod
    def load_data(work_dir, fermentation_number, data_file, x_cols, y_cols):
        file_path = os.path.join(work_dir, f"fermentation_{fermentation_number}", data_file)
        df = pd.read_excel(file_path)
        x = df[x_cols].values
        y = df[y_cols].values
        return x, y

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

    def __init__(self, model: nn.Module, scaler: object, dataset: pd.DataFrame, timestamp: float, max_steps=100):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.window_size = 20
        self.dataset = dataset
        self.timestamp = timestamp
        self.max_steps = max_steps

        # 参数的物理约束参考
        self.param_constraints = [
            (0, 0.0, None),  # 空气流量
            (1, 0.0, None),  # 溶氧
            (2, 3.0, 8.0),  # PH
            (3, 0.0, None),  # 搅拌
            (4, 0.0, None),  # 温度
            (5, 0.0, None),  # 氧气流量
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
            (max(epsilon, initial_params[0] - 1.0), initial_params[0] + 1.0),  # 空气流量
            (max(epsilon, initial_params[1] - 1.0), initial_params[1] + 1.0),  # 溶氧
            (max(3.0, initial_params[2] - 0.5), min(8.0, initial_params[2] + 0.5)),  # PH
            (max(epsilon, initial_params[3] - 10.0), initial_params[3] + 10.0),  # 搅拌
            (max(0.0, initial_params[4] - 1.0), initial_params[4] + 1.0),  # 温度
            (max(epsilon, initial_params[5] - 1.0), initial_params[5] + 1.0),  # O2流量
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
        self.prev_od600 = None

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

        # 计算参数变化惩罚项
        prev_action = self.history[-2]
        param_change = np.linalg.norm(clipped_action - prev_action)
        penalty = 0.1 * param_change  # 惩罚系数可以调整

        # 改进后的奖励函数
        current_od600 = pred.item()
        if self.prev_od600 is not None:
            od600_growth = current_od600 - self.prev_od600
        else:
            od600_growth = 0
        self.prev_od600 = current_od600

        reward = od600_growth - penalty

        done = self.step_count >= self.max_steps
        self.step_count += 1

        return clipped_action, reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        preds = []
        current_history = self.history.copy()
        hidden = self.model.init_hidden(batch_size=1)

        for _ in range(steps):
            # 构建输入序列
            input_seq = current_history[-self.window_size:]
            # 数据预处理
            scaled_seq = self.scaler.transform(input_seq)
            tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, hidden = self.model(tensor_input, hidden)

            preds.append(pred.item())
            new_state = current_history[-1].copy()
            new_state = np.roll(new_state, -1)
            new_state[-1] = pred.item()
            current_history = np.append(current_history[1:], [new_state], axis=0)

        return preds

# ---------------------- 数据加载器 ----------------------
class FermentationData(Dataset):
    def __init__(
        self, work_dir="./Data", train_mode=True, y_var=["od_600"], ws=20, stride=5, timestamp=None
    ):
        self.work_dir = work_dir
        self.train = train_mode
        self.timestamp = timestamp  # 新增：支持时间戳筛选
        print("Loading dataset...")  # 添加打印语句
        # lists of number of fermentations for training and testing
        # Data
        self.train_fermentations = [
            8,  # 0.540000
            11,  # 3.670000
            12,  # 3.840000
            #### 14,  # 4.500000
            16,  # 2.050000
            ### 17,  # 17.000000
            ### 19,  # 14.500000
            ### 20,  # 14.800000
            22,  # 0.500000
            23,  # 0.570000
            24,  # 0.530000
            25,  # 0.554000
            26,  # 0.532000
            27,  # 0.598000
            # 28,  # 0.674000
        ]
        self.test_fermentations = [28]

        # variables with cumulative values
        self.cumulative_var = [
            "dm_o2",
            "dm_air",
            "dm_spump1",
            "dm_spump2",
            "dm_spump3",
            "dm_spump4",
        ]
        # variables with binary values
        self.binary_var = ["induction"]

        # input variables
        self.x_var = [
            "m_ph",
            "m_ls_opt_do",
            "m_temp",
            "m_stirrer",
            "dm_o2",
            "dm_air",
            "dm_spump1",
            "dm_spump2",
            "dm_spump3",
            "dm_spump4",
            "induction",
        ]
        # output variable
        self.y_var = y_var

        # Using fermentation 16 for computing normalisation parameters
        self.fermentation_norm_number = 22  # 16
        self.X_norm, _ = self.load_data(
            fermentation_number=self.fermentation_norm_number
        )
        self.X_norm = self.X_norm[0]
        self.X_norm = self.cumulative2snapshot(self.X_norm)

        # Loading data
        self.X, self.Y = self.load_data()

        if self.train:
            self.ws, self.stride = (ws, stride)
        else:
            self.ws, self.stride = (ws, 1)  # Stride for test is set to 1

        # Preprocessing data
        self.X = self.preprocess_data(
            self.X, norm_mode="z-score", ws=self.ws, stride=self.stride
        )
        self.Y = self.preprocess_labels(
            self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride
        )

        # Shuffling for training
        if self.train:
            np.random.seed(1234)
            idx = np.random.permutation(len(self.X))

            self.X = self.X[idx]
            self.Y = self.Y[idx]

    def load_data(self, fermentation_number=None):
        # Load data for train/test fermentations
        X = []
        Y = []

        # Loading single fermentation data
        if fermentation_number is not None:
            data = utils.load_data(
                work_dir=self.work_dir,
                fermentation_number=fermentation_number,
                data_file="data.xlsx",
                x_cols=self.x_var,
                y_cols=self.y_var,
            )
            X.append(data[0])
            Y.append(data[1])

            return np.array(X), np.array(Y)

        # Loading train/test fermentations data
        if self.train:
            fermentations = self.train_fermentations
        else:
            fermentations = self.test_fermentations

        for fn in fermentations:
            data = utils.load_data(
                work_dir=self.work_dir,
                fermentation_number=fn,
                data_file="data.xlsx",
                x_cols=self.x_var,
                y_cols=self.y_var,
            )
            x, y = data
            df = pd.DataFrame(x, columns=self.x_var)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
            if self.timestamp is not None:
                df = df[df['Timestamp'] <= self.timestamp]
                y = y[df.index]
            x = df[self.x_var].values
            if len(x) > 0:
                X.append(x)
                Y.append(y)

        return np.array(X), np.array(Y)

    def cumulative2snapshot(self, X):
        # 假设这里是将累积值转换为快照值的逻辑，需要根据实际情况实现
        return X

    def preprocess_data(self, X, norm_mode, ws, stride):
        # 假设这里是数据预处理的逻辑，需要根据实际情况实现
        return X

    def preprocess_labels(self, Y, norm_mode, ws, stride):
        # 假设这里是标签预处理的逻辑，需要根据实际情况实现
        return Y

# 训练LSTM模型
def train_lstm_model(model, dataset, scaler, hidden_dim, num_layers, epochs=100, lr=0.001):
    # 准备数据
    data = dataset.iloc[:, 1:12].values
    scaled_data = scaler.transform(data)
    X = []
    y = []
    window_size = 20
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size, -1])
    X = np.array(X)
    y = np.array(y)

    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).unsqueeze(1).to(device)

    dataset = TensorDataset(X, y)
    kf = KFold(n_splits=5, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"Fold {fold + 1}")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                hidden = model.init_hidden(inputs.size(0))
                outputs, _ = model(inputs, hidden)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    hidden = model.init_hidden(inputs.size(0))
                    outputs, _ = model(inputs, hidden)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    return model
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

    # 训练LSTM模型
    model = train_lstm_model(model, dataset, data_loader.scaler, args.hidden_dim, args.num_layers)

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
    best_model_dir = f"./models_{int(time.time())}"
    eval_env = Monitor(FermentationEnv(model, data_loader.scaler, dataset, float(args.Timestamp)))
    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_dir, n_eval_episodes=3)
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=eval_callback,
        progress_bar=True
    )
    print(f"训练完成，耗时 {time.time() - start_time:.1f}秒")

    # 加载最佳模型
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        best_model = PPO.load(best_model_path)
        print("✅ 最佳模型加载成功")
    else:
        print("未找到最佳模型，使用训练结束的模型进行预测。")
        best_model = model_ppo

    # 打印训练结果
    obs, _ = env.reset()
    optimized_action, _ = best_model.predict(obs, deterministic=True)

    # 预测优化后的 OD600
    # 先将环境状态更新为优化后的参数
    env.history[-1] = optimized_action
    optimized_od600_preds = env.predict_od600()

    # 计算10步内优化前后的 OD600 预测值的平均值
    initial_od600_avg = np.mean(initial_od600_preds[:10])
    optimized_od600_avg = np.mean(optimized_od600_preds[:10])

    print(f"\n10步内优化前的 OD600 预测值的平均值：{initial_od600_avg:.4f}")
    print(f"10步内优化后的 OD600 预测值的平均值：{optimized_od600_avg:.4f}")

    # 如果10步内优化后的 OD600 预测值的平均值小于等于优化前的，则使用初始参数
    if optimized_od600_avg <= initial_od600_avg:
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

    print("\n优化前的 OD600 预测值（前 10 个时间步）：", initial_od600_preds[:10])
    print("优化后的 OD600 预测值（前 10 个时间步）：", optimized_od600_preds[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument('--weights', type=str, required=True, help="模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=2, help="LSTM层数")
    parser.add_argument('--Timestamp', type=str, required=True, help="初始时间戳（格式：小时:分钟）")

    args = parser.parse_args()
    main(args)