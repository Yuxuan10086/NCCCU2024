import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
import run
local_path = os.path.dirname(os.path.abspath(__file__))

def main():
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda")
    # 设置数据的预处理步骤，包括缩放、裁剪、转换为张量以及归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])
    print("step1")
    # 加载训练数据集
    train_dataset = datasets.ImageFolder(root=r'C:\Users\24253\Desktop\NCCCU2024\train', transform=transform)
    print("step2")
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=12, pin_memory=True) # 本地访问可以开多点进程
    print("load...")
    model = torch.load(local_path + "\\model.pth", map_location=torch.device("cuda"), weights_only=False)
    print("load finished")


    # 冻结模型参数
    for name, param in model.named_parameters():
    # 根据ResNet50的结构，layer2、layer3、layer4分别对应于不同的残差块组
        if "layer1" in name or "layer2" in name or "layer3" in name:
            param.requires_grad = False

    # 替换模型的最后一层
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # 替换全连接层，以匹配新的类别数

    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器，仅优化最后一层的参数

    num_epochs = 5  # 训练的轮数
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} start...')
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):  # 遍历数据加载器中的所有批次
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播，计算输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累积损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')  # 打印每个epoch的损失
        torch.save(model, local_path + "\\model.pth")
        run.main()
        torch.save(model, local_path + "\\model" + str(epoch) + ".pth")

    # torch.save(model.state_dict(), local_path + "\\model.pth")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time/3600:.2f} 小时")

