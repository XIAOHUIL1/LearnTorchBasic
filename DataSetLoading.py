import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data():
    """ 加载数据集 """
    # 训练数据集的加载
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    # 测试数据集的加载
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    """
    root:存储训练/测试数据的路径
    train指定训练或测试数据集，当布尔值为True则为训练集，当布尔值为False则为测试集
    download=True从互联网下载数据（如果无法在本地获得）
    transform指定特征转换方式，target_transform指定标签转换方式
    """
    return train_data, test_data


train_data, test_data = load_data()
print(train_data)


def show_data(train_data):
    label_map = {
        0: "T_Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    # 从训练集种随机抽出九个样本
    for i in range(1, cols * rows +1):
    # 设置索引，索引取值为0到训练集的长度
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        """
        torch.randint(low=0， high， size， generator=None， out=None， dtype=None， layout=torch.strided， device=None， requires_grad=False） → Tensor
        用于取随机整数，返回值为张量
        low：int类型，表明要从分布中提取的最低整数
        high：int类型，表明要从分布中提取的最高整数1
        size：元组类型，表明输出张量的形状
        dtype：返回值张量的数据类型
        device：返回张量所需的设备
        requires_grad：布尔类型，表明是否要对返回的张量自动求导。
        例如：
        torch.randint(3, 5, (3，））
        tensor([4, 3, 4])
        意味着生成一个一维3元素向量，其中向量的元素取值从3-5取
        """
    # 取出对应样本的图片和标签
    img, label = train_data[sample_idx]
    # 依次画在事先指定的九宫格图上
    figure.add_subplot(rows, cols, i)
    # 设置对应图片的标题
    plt.title(label_map[label])
    # 关闭坐标轴
    plt.axis("off")
    # 展示图片
    plt.imshow(img.squeeze(), cmap="gray")
    # 释放画布
    plt.show()

train_data, test_data = load_data()
show_data(train_data)