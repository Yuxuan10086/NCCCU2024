import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
local_path = os.path.dirname(os.path.abspath(__file__))

def main():
    torch.backends.cudnn.enabled = False

    # 设置数据的预处理步骤，包括缩放、裁剪、转换为张量以及归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])

    # 加载训练数据集，这里假设数据集存放在'train'文件夹下，并且按照您的文件结构组织
    train_dataset = datasets.ImageFolder(root='C:\\Users\\24253\\Desktop\\NCCCU2024\\train', transform=transform)

    # 创建数据加载器，用于在训练时批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # 加载预训练的ResNet50模型
    # model = models.resnet50(pretrained=True)
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(local_path + "\\model.pth"))


    # 冻结模型的所有参数，这样在训练过程中它们不会被更新
    for param in model.parameters():
        param.requires_grad = False

    # 替换模型的最后一层，以适应新的分类任务
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # 替换全连接层，以匹配新的类别数

    for param in model.fc.parameters():
        param.requires_grad = True  # 只解冻最后一层的参数

    # 将模型移动到GPU上进行训练，如果GPU不可用则使用CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器，仅优化最后一层的参数

    # 训练模型
    num_epochs = 3  # 训练的轮数
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, labels in train_loader:  # 遍历数据加载器中的所有批次
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU

            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播，计算输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数

            running_loss += loss.item()  # 累积损失
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')  # 打印每个epoch的损失

    torch.save(model, local_path + "\\model.pth")
if __name__ == '__main__':
    main()