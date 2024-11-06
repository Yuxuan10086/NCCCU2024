import torch
import torchvision.models as models
import os
def turn_to_full():
    local_path = os.path.dirname(os.path.abspath(__file__))
    model = models.resnet152(pretrained=False)
    model.load_state_dict(torch.load(local_path + "\\model.pth", weights_only = False))
    torch.save(model, local_path + "\\model.pth")

import random

def delete_random_png_files(folder_path, percentage):
    """
    删除指定文件夹下指定百分比的PNG文件。

    参数:
    folder_path (str): 要删除PNG文件的文件夹路径。
    percentage (float): 要删除的PNG文件的百分比 (0.0 到 1.0)。
    """
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print("提供的路径不存在")
        return

    # 获取文件夹中所有PNG文件的列表
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # 计算要删除的文件数量
    num_files_to_delete = int(len(png_files) * percentage)
    
    # 如果没有PNG文件或不需要删除任何文件，则退出函数
    if num_files_to_delete == 0:
        print("没有PNG文件需要删除或删除百分比为零。")
        return
    
    # 随机选择要删除的文件
    files_to_delete = random.sample(png_files, num_files_to_delete)
    
    # 删除选定的文件
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"已删除文件：{file_path}")

def remove_first_five_chars_from_folders(folder_path):
    """
    将指定文件夹下所有文件夹的名字的前五个字符去掉。

    参数:
    folder_path (str): 要修改的文件夹路径。
    """
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print("提供的路径不存在")
        return

    # 检查路径是否为文件夹
    if not os.path.isdir(folder_path):
        print("提供的路径不是一个文件夹")
        return

    # 获取文件夹下所有项的列表
    items = os.listdir(folder_path)
    
    # 遍历所有项
    for item in items:
        item_path = os.path.join(folder_path, item)
        
        # 检查项是否为文件夹
        if os.path.isdir(item_path):
            # 获取文件夹的新名字
            new_folder_name = item[5:] if len(item) > 5 else ''
            
            # 如果新名字不为空，重命名文件夹
            if new_folder_name:
                new_folder_path = os.path.join(folder_path, new_folder_name)
                os.rename(item_path, new_folder_path)
                print(f"文件夹 '{item}' 已重命名为 '{new_folder_name}'")
            else:
                print(f"文件夹 '{item}' 名字太短，无法去掉前五个字符。")

delete_random_png_files(r"C:\Users\24253\Desktop\NCCCU2024\testA\task2\query", 0.5)
delete_random_png_files(r"C:\Users\24253\Desktop\NCCCU2024\testA\task1\query", 0.5)