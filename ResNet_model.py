import torch
from torch import nn
from torchsummary import summary


# 定义残差块类
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1,use_1conv=False):
        """
        注意点：输入通道不等于输出通道时，必须使用1x1卷积核。二者相等时，用不用都是一样的。
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param stride: 步幅，通常取值为1或2，取1表示宽高不变，取2便是宽高折半
        :param use_1conv: 是否用到1x1卷积核（该参数其实可以通过"输入通道是否等于输出通道"来获得，此处为了方便，人工输入即可）
        """

        # 父类初始化
        super(Residual, self).__init__()

        self.ReLU = nn.ReLU()  # 定义激活函数
        self.cov1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1,stride=stride)  # 卷积层：3x3大小，填充1
        self.cov2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,padding=1)  # 卷积层：3x3大小，填充1
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)  # 批归一化层
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)  # 批归一化层

        if use_1conv:
            self.cov3 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride)     #卷积层：1x1大小，填充0，步幅1或2
        else:
            self.cov3 = None

    def forward(self,x):

        y=self.cov1(x)
        y=self.bn1(y)
        y=self.ReLU(y)
        y=self.cov2(y)
        y=self.bn2(y)

        if self.cov3:
            x=self.cov3(x)

        y=self.ReLU(y+x)

        return y


class ResNet18(nn.Module):
    def __init__(self,num_classes:int):
        # 父类初始化
        super(ResNet18, self).__init__()

        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(in_channels=64,out_channels=64),   # 第一个残差块
            Residual(in_channels=64, out_channels=64),  # 第二个残差块
            Residual(in_channels=64, out_channels=128,stride=2,use_1conv=True),  # 第三个残差块
            Residual(in_channels=128, out_channels=128),  # 第四个残差块
            Residual(in_channels=128, out_channels=256,stride=2,use_1conv=True),  # 第五个残差块
            Residual(in_channels=256, out_channels=256),  # 第六个残差块
            Residual(in_channels=256, out_channels=512,stride=2,use_1conv=True),  # 第七个残差块
            Residual(in_channels=512, out_channels=512),  # 第八个残差块
        )

        self.b3=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),  # 添加Dropout
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5 / 2),    # 添加Dropout
            nn.Linear(in_features=256, out_features=num_classes)    # 再添加一层全连接层
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 卷积层初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # BatchNorm层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 全连接层初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self,x):
        x=self.b1(x)
        x=self.b2(x)
        y = self.b3(x)

        return y

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=ResNet18(num_classes=10).to(device)

    print(summary(model,(3,224,224)))
