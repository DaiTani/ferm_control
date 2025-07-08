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
import json  # 新增导入 json 库

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--batch_size", default=256, type=int, help="test batchsize")
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--seed", default=123)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--model", default="lstm", type=str)
    parser.add_argument("--pred_steps", default=1, type=int, help="Number of prediction steps")

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Predict with overlapped sequences
    def test(epoch, model, testloader):
        model.eval()

        loss = 0
        err = 0

        iter = 0

        # Initialise model hidden state
        h = model.init_hidden(args.batch_size)

        # Initialise vectors to store predictions, labels
        preds = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        labels = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        n_overlap = np.zeros(len(test_dataset) + test_dataset.ws - 1)

        N = 10
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(testloader):
                iter += 1

                batch_size = input.size(0)

                h = model.init_hidden(batch_size)
                h = tuple([e.data for e in h])

                input, label = input.cuda(), label.cuda()

                output, h = model(input.float(), h)

                y = output.view(-1).cpu().numpy()
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")

                # Store predictions and labels summing over the overlapping
                preds[
                    batch_idx : (batch_idx + test_dataset.ws)
                ] += y_smooth
                labels[batch_idx : (batch_idx + test_dataset.ws)] += (
                    label.view(-1).cpu().numpy()
                )
                n_overlap[batch_idx : (batch_idx + test_dataset.ws)] += 1.0

                loss += mse(output, label.float())
                err += torch.sqrt(mse(output, label.float())).item()

        loss = loss / len(test_dataset)
        err = err / len(test_dataset)

        # Compute the average dividing for the number of overlaps
        preds /= n_overlap
        labels /= n_overlap

        return err, preds, labels

    # Setting data
    test_dataset = FermentationData(
        work_dir=args.dataset, train_mode=False, y_var=["od_600"]
    )

    # 打印数据集完成加载后最后一行的数据
    last_row_X = test_dataset.X[-1]
    last_row_Y = test_dataset.Y[-1]
    print("数据集完成加载后 X 的最后一行数据: ", last_row_X[-1])
    print("数据集完成加载后 Y 的最后一行数据: ", last_row_Y[-1])
	# 输出 X 和 Y 数据的最后一部分
    #print("X 数据的最后一部分:", last_row_X[-1][-1])
    #print("Y 数据的最后一部分:", last_row_Y[-1][-1])
	
    print("Loading testing-set!")
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, num_workers=2, shuffle=False
    )

    # Setting model
    if args.model == "lstm":
        model = LSTMPredictor(
            input_dim=test_dataset.get_num_features(),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        )
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=test_dataset.get_num_features(),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        )

    model.cuda()

    weights = (
        os.path.join("model_weights", "od600_prediction", args.model, "weights_best.tar")
        if args.weights == ""
        else args.weights
    )
    model = utils.load_weights(model, weights)

    mse = nn.MSELoss()

    # 加载归一化参数
    norm_file_path = os.path.join(args.dataset, "norm_file.json")
    with open(norm_file_path, 'r') as f:
        norm_data = json.load(f)
    y_mean = norm_data['mean']
    y_std = norm_data['std']

    # Testing
    print("\nTesting")
    err, preds, labels = test(0, model, test_loader)

    # 逆归一化预测结果和标签
    preds_denorm = preds * y_std + y_mean
    labels_denorm = labels * y_std + y_mean

    preds_denorm = preds_denorm.reshape(-1, 1)
    labels_denorm = labels_denorm.reshape(-1, 1)

    preds_denorm = preds_denorm[50:]
    labels_denorm = labels_denorm[50:]

    mse = np.square(np.subtract(preds_denorm, labels_denorm)).mean()
    rmse = math.sqrt(mse)

    # Relative Error on Final Yield
    refy = abs(preds_denorm[-1] - labels_denorm[-1]) / labels_denorm[-1] * 100

    # 读取 data.xlsx 文件的第一列数据
    data_path = os.path.join(args.dataset, "28", "data.xlsx")
    df = pd.read_excel(data_path)
    first_column = df.iloc[:, 0].values
    first_column = first_column[50:]  # 与 preds 和 labels 对齐

    # 将 rmse 和 refy 扩展为与 first_column 长度相同的数组
    rmse_array = np.full_like(first_column, rmse)
    refy_array = np.full_like(first_column, refy)

    # 将第一列数据添加到 results.npz 的最左侧
    timestamp_data = np.column_stack((first_column, preds_denorm, labels_denorm, rmse_array, refy_array))

    # 保存到 results.npz
    np.savez(
        weights.split("/weights")[0] + "/results.npz",
        Timestamp=timestamp_data,
        preds=preds_denorm,
        labels=labels_denorm,
        rmse=rmse,
        refy=refy,
    )
    print("Saved: ", weights.split("/weights")[0] + "/results.npz")

    # 绘制曲线
    utils.plot_od600_curve(
        preds_denorm, labels_denorm, weights[:-17], rmse, refy
    )

    print("\nRMSE Error OD600: ", rmse)
    print(
        "\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1]))
    )

    # 绘制多步预测曲线
    plt.figure(figsize=(10, 6))
    plt.title(f"Multi-step Prediction (Prediction Steps: {args.pred_steps})")
    plt.plot(labels_denorm, label="True Values", color="black", linewidth=2)
    for step in range(args.pred_steps):
        plt.plot(preds_denorm[:, step], label=f"Predicted Step {step + 1}")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("OD600")
    plt.savefig(weights.split("/weights")[0] + "/multi_step_prediction.png")
    plt.show()

if __name__ == '__main__':
    main()