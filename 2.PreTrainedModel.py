'''
    加载预训练模型，冻结层
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152  # ResNet系列


# 定义训练函数，需要
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()

        # 得到预测的结果pred
        pred = model(X)

        # 计算预测的误差
        # print(pred,y)
        loss = loss_fn(pred, y)

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练100次，输出一次当前信息
        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    print("size = ", size)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("correct = ", correct)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    batch_size = 4
    #
    # # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)
    #
    train_dataloader = DataLoader(dataset=train_data, num_workers=1, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=1, pin_memory=True, batch_size=batch_size)
    #
    # # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    '''
            修改ResNet18模型的最后一层
    '''
    pretrain_model = resnet18(pretrained=False)  # 加载ResNet
    # pretrain_model = resnet34(pretrained=False)
    print(pretrain_model)
    print('OK')
    # pretrain_model = resnet101(pretrained=False)  # 加载ResNet
    num_ftrs = pretrain_model.fc.in_features    # 获取全连接层的输入
    pretrain_model.fc = nn.Linear(num_ftrs, 2)  # 全连接层改为不同的输出
    print(pretrain_model)
    #
    # 预先训练好的参数， 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    pretrained_dict = torch.load('./resnet18_pretrain.pth')
    # pretrained_dict = torch.load('./resnet34_pretrain.pth')
    # print(pretrained_dict)

    # y = k x + b
    #
    # #
    # # 弹出fc层的参数
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')
    print(pretrained_dict)
    # #
    # # # 自己的模型参数变量，在开始时里面参数处于初始状态，所以很多0和1
    model_dict = pretrain_model.state_dict()
    print(model_dict)
    #
    # # # 去除一些不需要的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    # # # 模型参数列表进行参数更新，加载参数
    model_dict.update(pretrained_dict)
    print(model_dict)
    # #
    # # 改进过的预训练模型结构，加载刚刚的模型参数列表
    pretrain_model.load_state_dict(model_dict)
    # #
    print(pretrain_model)
    #
    # '''
    #     冻结部分层
    # '''
    # 将满足条件的参数的 requires_grad 属性设置为False
    for name, value in pretrain_model.named_parameters():
        if (name != 'fc.weight') and (name != 'fc.bias'):
            value.requires_grad = False
    # #
    # # filter 函数将模型中属性 requires_grad = True 的参数选出来
    params_conv = filter(lambda p: p.requires_grad, pretrain_model.parameters())    # 要更新的参数在parms_conv当中
    # # #
    # #
    # # # pretrain_model = resnet18(pretrained=False, num_classes=5)
    # #
    model = pretrain_model.to(device)
    # #
    # #
    # #
    # #
    # #
    # # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()
    # #
    # # # # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 初始学习率
    # #
    # '''   控制优化器只更新需要更新的层  '''
    optimizer = torch.optim.SGD(params_conv, lr=1e-3)  # 初始学习率
    # #
    # # 一共训练5次
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")

    # 保存训练好的模型
    torch.save(model.state_dict(), "model_resnet34.pth")
    print("Saved PyTorch Model Success!")