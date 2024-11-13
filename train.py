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
from PIL import Image
from torch.utils.data import Dataset
local_path = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NewDataset(Dataset):
    def __init__(self, root_dir, transform=None, start=0, step=100):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0
        self.classes = os.listdir(root_dir)
        self.start = start
        self.step = step
        self.samples = self._get_samples()
    def _get_samples(self):
        samples = []
        for class_name in self.classes:
            if self._load_rule(class_name):  # 使用提供的函数决定是否加载该类别的数据
                class_dir = os.path.join(self.root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.num_classes))
                self.num_classes += 1
        return samples
    def __getitem__(self, index):
        img_path, class_idx = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx
    def _load_rule(self, name):
        return int(name.split('_')[1]) >= self.start and int(name.split('_')[1]) < self.start + self.step
    def __len__(self):
        return len(self.samples)

def main():
    data_root = r'C:\Users\24253\Desktop\train'
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda")
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])

    model = torch.load(local_path + "\\model.pth", map_location=device, weights_only=False)
    # 冻结模型参数
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器
    num_epochs = 5  # 训练的轮数
    step = 300 # 每次训练载入的类别数
    start = 35
    exc = 0
    res = ['img_name,label'] 
    for name in os.listdir(data_root):
        if int(name.split('_')[1]) < start:
            exc += 1
    data_epoch_num = int((len(os.listdir(data_root)) - exc)/step + 0.6)
    print(f"共{data_epoch_num}批数据")
    # run.main()
    for data_epoch in range(data_epoch_num):
        # 加载数据
        train_dataset = NewDataset(data_root, transform, start+step*data_epoch, step)
        # train_dataset = datasets.ImageFolder(data_root, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=12, pin_memory=True)

        # num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数
        # model.fc = nn.Linear(num_ftrs, step)  # 替换全连接层，以匹配新的类别数
        num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数
        model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        model.to(device)

        for name, param in model.named_parameters():
            if not "fc" in name and not "layer4" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(f"\n开始训练第{data_epoch+1}批 class{start+step*data_epoch}-{start+step*(data_epoch+1)} 共{train_dataset.num_classes}类")
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1} start...')
            for name, param in model.named_parameters():
                if "layer3." + str(3-epoch) in name:
                    param.requires_grad = True
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
            res.append("model_" + str(data_epoch) + "_" + str(epoch) + ',' + run.main())
            torch.save(model, local_path + "\\model_" + str(data_epoch+1) + "_" + str(epoch) + ".pth")
    
    with open(os.path.join(local_path, "train_res.csv"), 'w') as f:
        f.write('\n'.join(res))

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"训练时间: {execution_time/3600:.2f} 小时")

