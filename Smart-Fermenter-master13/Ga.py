import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasetC import FermentationData  # 假设数据集类在 datasetB.py 中
from model import LSTMPredictor, RNNpredictor  # 假设模型类在 model.py 中
import utils
from tqdm import tqdm  # 导入 tqdm

class ParameterOptimizer:
    def __init__(self, param_names, param_bounds, model, test_dataset, x_mean, x_std, y_mean, y_std):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.n_params = len(param_names)
        self.model = model
        self.test_dataset = test_dataset
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.best_od600 = float('-inf')
        self.saved_preds = None
        self.saved_labels = None
        self.saved_iter = None  # 新增：记录 iter 值
        self.saved_predsB = None  # 新增：记录 iter 值

    # 打印数据集完成加载后最后一行的数据
        #last_row_X = self.test_dataset.X[-1]
        #last_row_Y = self.test_dataset.Y[-1]
        #print("X 数据的最后一部分:", last_row_X[-1])
        #print("Y 数据的最后一部分:", last_row_X[-1])
    # 对最后一行数据进行逆归一化
        #last_row_X_denorm = last_row_X * self.x_std + self.x_mean
        #last_row_Y_denorm = last_row_Y * self.y_std + self.y_mean

        #print("数据集完成加载后 X 的最后一行数据: ", last_row_X_denorm)
        #print("数据集完成加载后 Y 的最后一行数据: ", last_row_Y_denorm)
    # 输出 X 和 Y 数据的最后一部分
        #print("X 数据的最后一部分:", last_row_X_denorm[-1])
        #print("Y 数据的最后一部分:", last_row_Y_denorm[-1])




        
        # 遗传算法参数
        self.pop_size = 50
        self.n_generations = 100
        self.mutation_rate = 0.1

        # 新增属性：最佳OD600
        self.best_od600 = float('-inf')		
    def update_specific_window(self, new_value):
        """
        更新滑动窗口中受影响的部分
        """
        if self.saved_windows is not None:
            # 更新最后一行的最后一个特征位置
            self.saved_windows[self.last_row_index, -1] = new_value
        else:
            raise ValueError("滑动窗口数据未初始化，请先调用 preprocess_data 方法。")    
    def optimize(self, initial_params, initial_score):
        # 初始化种群
        population = self._initialize_population(initial_params)
        self.best_params = initial_params        
        for gen in range(self.n_generations):
            # 评估适应度
            fitness = np.array([self._evaluate(ind, initial_score) for ind in population])
            
            # 选择
            selected = self._select(population, fitness)
            
            # 交叉和变异
            population = self._crossover_mutate(selected)
                # 使用 tqdm 显示进度条
        for gen in tqdm(range(self.n_generations), desc="Generations", unit="gen"):
            fitness = np.array([self._evaluate(ind, initial_score) for ind in population])
            selected = self._select(population, fitness)
            population = self._crossover_mutate(selected)
            #print(f"当前 population[np.argmax(fitness)] 的值: {population[np.argmax(fitness)]}")            
        return self.best_params
        
    def _initialize_population(self, base_params):
        population = []
        for _ in range(self.pop_size):
            ind = base_params.copy()
            for i in range(self.n_params):
                ind[i] += np.random.normal(0, self.param_bounds[i][1]*0.1)
                ind[i] = np.clip(ind[i], *self.param_bounds[i])
            population.append(ind)
        return np.array(population)

    def _evaluate(self, individual, baseline):
        #print(f"当前 individual 的值: {individual}")
        
        # 归一化优化后的参数
        optimized_params_norm = self._normalize_individual(individual)

        # 更新测试数据集的最后一行的最后一个特征位置
        last_row_X = self.test_dataset.X[-1]
        last_row_Y = self.test_dataset.Y[-1]
        self.test_dataset.update_specific_window(optimized_params_norm)

        # 打印最后一个滑动窗口的数据内容
        #if self.test_dataset.saved_windows is not None:
        #    print(f"更新后的最后一个滑动窗口数据: {111111111111}")

        if self.saved_preds is not None:
            # 获取受影响的窗口索引
            affected_window_index = self.test_dataset.last_row_index

            # 重新初始化 DataLoader
            test_loader = DataLoader(
                dataset=self.test_dataset, batch_size=1, num_workers=0, shuffle=False
            )

            # 初始化隐藏状态
            h = self.model.init_hidden(1)

            # 仅对受影响的窗口及其后续窗口进行预测
            self.model.eval()
            with torch.no_grad():
                for idx in range(affected_window_index, len(self.test_dataset)):
                    input, label = test_loader.dataset[idx]
                    input, label = input.cuda(), label.cuda()

                    # 重新初始化隐藏状态
                    h = self.model.init_hidden(1)
                    h = tuple([e.data for e in h])

                    output, h = self.model(input.float().unsqueeze(0), h)  # 使用隐藏状态
                    y = output.view(-1).cpu().numpy()

                    # 确保 y 是一个标量值
                    y_scalar = y[-1]
                    self.saved_preds[idx] = y_scalar
                    self.saved_labels[idx] = label.cpu().numpy()[-1]
                    self.saved_predsB = self.saved_preds[idx]
                    preds_denorm = self.saved_predsB * self.y_std + self.y_mean
                    current_od600 = preds_denorm
                    #print(f"output3: {current_od600}")                    
        else:
            # 第一次运行时，完整预测所有窗口
            test_loader = DataLoader(
                dataset=self.test_dataset, batch_size=1, num_workers=0, shuffle=False
            )
            err, preds, labels, batch_to_window_map, iter_values = self.test(0, self.model, test_loader)
            self.saved_preds = preds
            self.saved_labels = labels
            self.saved_iter = iter_values  # 记录 iter 值

            # 保存批次到窗口的映射
            self.batch_to_window_map = batch_to_window_map

        # 逆归一化预测结果
            preds_denorm = self.saved_preds * self.y_std + self.y_mean
            current_od600 = preds_denorm[-1]

        # 更新最佳 OD600
        if current_od600 > self.best_od600:
            self.best_od600 = current_od600
            self.best_params = individual

        print(f"当前预测OD600: {current_od600:.4f} | 历史最佳OD600: {self.best_od600:.4f}")
        #print(f"当前 best_params 的值: {self.best_params}")

        return current_od600

    def _select(self, population, fitness):
        if len(population) <= 2:
            return population
        return population[np.argsort(fitness)[-int(self.pop_size*0.5):]]

    def _crossover_mutate(self, selected):
        new_pop = []
        while len(new_pop) < self.pop_size:
            indices = np.random.permutation(len(selected))[:2]  # 随机选择两个个体的索引
            parents = selected[indices]
            child = (parents[0] + parents[1])/2
            for i in range(self.n_params):
                if np.random.rand() < self.mutation_rate:
                    child[i] += np.random.normal(0, self.param_bounds[i][1]*0.2)
                    child[i] = np.clip(child[i], *self.param_bounds[i])
            new_pop.append(child)
        return np.array(new_pop)

    def test(self, epoch, model, testloader):
        model.eval()
        loss = 0
        err = 0
        iter = 0
        h = model.init_hidden(1)  # batch_size 为 1
        preds = np.zeros(len(self.test_dataset) + self.test_dataset.ws - 1)
        labels = np.zeros(len(self.test_dataset) + self.test_dataset.ws - 1)
        n_overlap = np.zeros(len(self.test_dataset) + self.test_dataset.ws - 1)
        N = 10
        iter_values = []  # 新增：记录 iter 值

        # 新增：记录批次编号与滑动窗口索引的映射
        batch_to_window_map = {}

        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(testloader):
                iter += 1
                iter_values.append(iter)  # 记录 iter 值
                #print(f"iter: {iter}")
                batch_size = input.size(0)
                #print(f"batch_size: {batch_size}")
                h = model.init_hidden(batch_size)
                #print(f"hhhh3数据: {h}")
                h = tuple([e.data for e in h])
                #print(f"hhhh4数据: {h}")				
                input, label = input.cuda(), label.cuda()
                #print(f"input: {input}")
                #print(f"label: {label}")
                output, h = model(input.float(), h)
                y = output.view(-1).cpu().numpy()
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")

                # 存储预测和标签的累加值
                preds[batch_idx : (batch_idx + self.test_dataset.ws)] += y_smooth
                labels[batch_idx : (batch_idx + self.test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx : (batch_idx + self.test_dataset.ws)] += 1.0
                #print(f"label: {batch_idx}")

                # 记录批次编号与滑动窗口索引的映射
                batch_to_window_map[batch_idx] = list(range(batch_idx, batch_idx + self.test_dataset.ws))

                # 计算损失
                loss += nn.MSELoss()(output, label.float())
                err += torch.sqrt(nn.MSELoss()(output, label.float())).item()

        loss = loss / len(self.test_dataset)
        err = err / len(self.test_dataset)

        # 计算平均预测和标签
        preds /= n_overlap
        labels /= n_overlap
        return err, preds, labels, batch_to_window_map, iter_values  # 返回 iter 值

    def mse(self, output, target):
        return torch.mean((output - target) ** 2)
    def _normalize_individual(self, individual):
        """
        归一化 individual 数据
        """
        return (individual - self.x_mean) / self.x_std	
    def _cumulative2snapshot(self, data):
    # Trasform cumulative data to snapshot data

        tmp_data = np.insert(data, 0, data[0])[:-1]

        return data - tmp_data
    