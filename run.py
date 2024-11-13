import time
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys

start_time = time.time()


def extract_features(image_path, model, device = torch.device("cuda")):
    # 设置数据的预处理步骤，与训练时相同
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
    ])

    # 加载图像并进行预处理
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加批次维度

    # 将图像移动到指定设备
    image = image.to(device)

    # 设置模型为评估模式
    model.eval()

    # 禁用梯度计算
    with torch.no_grad():
        # 前向传播，获取平均池化层的输出
        features = model.conv1(image)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)

        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features4 = model.layer4(features)

        # 平均池化层的输出
        # 1. 对 layer4 输出进行全局平均池化
        features4_avg = F.adaptive_avg_pool2d(features4, (1, 1))  # 输出 (batch_size, channels, 1, 1)
        features4_avg = features4_avg.view(features4_avg.size(0), -1)  # 压平为 (batch_size, channels)

        # 2. 获取平均池化层的输出（通常会被用作最终特征）
        pooled_features = model.avgpool(features4)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # 压平为 (batch_size, features_dim)

        # 3. 拼接池化层输出与全局平均池化后的 layer4 输出
        final_features = torch.cat((pooled_features, features4_avg), dim=1)  # 按通道维度拼接

    # 将最终特征转为 numpy 数组并返回
    return final_features.squeeze(0).cpu().numpy().reshape(1, -1)


def main(to_pred_dir = 0, result_save_path = 0):
    device = torch.device("cuda")
    local_path = os.path.dirname(os.path.abspath(__file__))
    if to_pred_dir == 0:
        to_pred_dir = local_path
        result_save_path = os.path.join(local_path, "res.csv")
    # 1.加载模型
    model = torch.load(os.path.join(local_path, "model.pth"), map_location=device, weights_only=False)
    model.to(device)

    
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py) # 当前文件夹路径
    dirpath = os.path.abspath(to_pred_dir)
    filepath = os.path.join(dirpath, 'testA') # 测试集A文件夹路径
    task_lst = [item for item in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, item)) and "task" in item]  

    res = ['img_name,label']  # 初始化结果文件，定义表头
    right = 0
    total = 0
    for task_name in task_lst:  # 循环task文件夹l
        # 2.提取支持集特征 构建knn分类器
        y_train = np.array([])  # 标签
        x_train = np.empty((0, 4096)) #((0, model.fc.in_features + model.layer4[-1].conv2.out_channels))
        support_path = os.path.join(filepath, task_name, 'support')  # 支持集路径（文件夹名即为标签）
        for support in [item for item in os.listdir(support_path) if os.path.isdir(os.path.join(support_path, item))]:
            for img in [item for item in os.listdir(os.path.join(support_path, support)) if ".png" in item]:
                # print(os.path.join(support_path, support, img))
                y_train = np.append(y_train, support)
                features = extract_features(os.path.join(support_path, support, img), model)
                # print(features.shape)
                # print(x_train.shape)
                x_train = np.vstack((x_train, features.copy()))
        # print(y_train)
        # print(x_train)
        knn = KNeighborsClassifier(n_neighbors = 5)  
        knn.fit(x_train, y_train)  

        query_path = os.path.join(filepath, task_name, 'query')  # 查询集路径（无标签，待预测图片）

        # 3.预测
        test_img_lst = [name for name in os.listdir(query_path) if name.endswith('.png')]
        total += len(test_img_lst)
        for pathi in test_img_lst:
            name_img = os.path.join(query_path, pathi)
            pred_class = knn.predict(extract_features(name_img, model))  # 这里指定了一个值代替预测值，选手需要根据自己模型进行实际的预测
            # print(pathi)
            print(pred_class)
            right += (pathi.split('_')[1] == pred_class[0])
            res.append(pathi + ',' + pred_class[0])

    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))
    
    print(f"正确率：{right/total*100:.2f}%")
    return str(right/total*100)

# if __name__ == "__main__":
#     main()
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"代码执行时间: {execution_time:.2f} 秒")

# python run.py "C:\Users\24253\Desktop\NCCCU2024" "C:\Users\24253\Desktop\NCCCU2024\res.csv"

if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)