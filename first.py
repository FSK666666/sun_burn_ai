import torchvision
import torchvision.transforms as transforms

# 定义数据预处理（转换为 Tensor 并归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载训练集
trainset = torchvision.datasets.CIFAR10(
    root='./data',      # 数据保存路径
    train=True,         # 是否为训练集
    download=True,      # 如果本地没有则下载
    transform=transform
)

# 下载测试集
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,        # 是否为测试集
    download=True,
    transform=transform
)

print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")