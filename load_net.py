import torch 
from torch import nn


class Conv2dNormActivation(nn.Module):
    def __init__(self, **args):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=args['in_channels'],
            out_channels=args['out_channels'],
            kernel_size=(args['kernel_size'],args['kernel_size']),
            stride=args['stride'],
            padding=args['kernel_size']//2,
            groups=args['groups'] if 'groups' in args.keys() else 1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(args['out_channels'], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, input_image):
        x = input_image
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class DepthWiseConv(nn.Module):
    def __init__(self,in_channels: int, out_channels: int,  kernel_size: int = 3, stride=1):
        super().__init__()

        self.conv = Conv2dNormActivation(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=stride, groups=in_channels)
        
    def forward(self, x):
        return self.conv(x)


class ResedualBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expension_factor: int, stride=1, isSE: bool= True):
        super().__init__()

        self.isSE = isSE
        expension_factor = expension_factor if self.isSE else 1
        exp_channels = expension_factor*in_channels
        self.conv1 = Conv2dNormActivation(in_channels=in_channels, out_channels=exp_channels, kernel_size=1, stride=1)
        self.conv2 = Conv2dNormActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3,stride=stride, groups=exp_channels)
        self.conv3 = nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if self.isSE:
            self.block = nn.Sequential(self.conv1, self.conv2, self.conv3, self.bn)
        else:
            self.block = nn.Sequential(self.conv2, self.conv3, self.bn)
        

    def forward(self, x):
        res = x
        x = self.block(x)
        x += res

        return x

#mobile net 
class MobileNetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.normact = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=(2,2))
        self.normact2 = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=(2,2))
        self.RBNBlock = ResedualBottleNeck(32,16,1, isSE=False)
        
    def forward(self, image: torch.Tensor):
        x = image
        x = self.normact(x)
        x = self.RBNBloack(x)

        return x


if __name__=='__main__':

    a = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=(1,1))
    print(a)