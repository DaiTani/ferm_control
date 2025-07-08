import numpy as np

data = np.load('logs/LSTM_Data5_Batch-28/results.npz', allow_pickle=True)
print(data.files)  # 查看文件中的键
print(data['combined_data'])  # 查看 combined_data 数据