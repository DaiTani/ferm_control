from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
from torch.multiprocessing import freeze_support  # 导入 freeze_support

# 数据预处理
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  # 如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True,  # 如果为true，则从 Internet 下载数据集并将其放在根目录中。如果已下载数据集，则不会再次下载。
    transform=transform
)

def main():
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(0, mp.cpu_count(), 2):  
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            shuffle=True, 
            num_workers=num_workers, 
            batch_size=64, 
            pin_memory=True
        )
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

if __name__ == '__main__':
    freeze_support()  # 在 Windows 上需要调用这个函数
    main()