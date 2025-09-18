import torch 
from torch import nn

class ConvNormAct(nn.Module):
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


## Without (No) Expenad-Squeeze (ES) BottleNeck (BN)
class NoESBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expension_factor: int, stride=1, isES: bool= True):
        super().__init__()

        self.res = in_channels == out_channels and stride == 1
        expension_factor = expension_factor
        exp_channels = expension_factor*in_channels
        self.conv2 = ConvNormAct(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3,stride=stride, groups=exp_channels)
        self.conv3 = nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        return x + res if self.res else x


class ResedualBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expension_factor: int, stride=1, isES: bool= True):
        super().__init__()
		
        self.res = in_channels == out_channels and stride == 1

        expension_factor = expension_factor if isES else 1
        exp_channels = expension_factor*in_channels

        l = []

        if isES: l.append(ConvNormAct(in_channels=in_channels, out_channels=exp_channels, kernel_size=1, stride=1))
        l.append(ConvNormAct(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3,stride=stride, groups=exp_channels))
        l.append(nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False))
        l.append(nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True))

        self.RBNBlock = nn.Sequential(*l)

    def forward(self, x):
        res = x
        x = self.RBNBlock(x)
        return x + res if self.res else x


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x

#mobile net 
class BurrahMobileNet(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.bottlenek_config=[# t, c,   n, s
                                [1, 16,  1, 1],
                                [6, 24,  2, 2],
                                [6, 32,  3, 2],
                                [6, 64,  4, 2],
                                [6, 96,  3, 1],
                                [6, 160, 3, 2],
                                [6, 320, 1, 1],]
        
        self.layers = []
        
        self.normact = ConvNormAct(in_channels=3, out_channels=32, kernel_size=3, stride=(2,2))
        # self.NOESBlock = NoESBN(32, 16, 1) 

        in_channels = 32
        for t, c, n, s in self.bottlenek_config:
            isES = False if t==1 else True
            for i in range(n):
                stride = s if i==0 else 1
                self.layers.append(
                    ResedualBottleNeck(in_channels=in_channels, out_channels=c, expension_factor=t, stride=stride, isES = isES)
                ) 
                in_channels = c

        self.layers.append(ConvNormAct(in_channels=320, out_channels=1280, kernel_size=1, stride=(1,1)))
        self.BottleNeckBlocks = nn.Sequential(*self.layers)
        self.classifier = Classifier(1280, 1000, 0.24)

    def forward(self, image: torch.Tensor):
        x = image
        x = self.normact(x)
        x = self.BottleNeckBlocks(x)
        x = self.classifier(x)

        return x

    def load_state_dict(self, state_dict, strict = True, assign = False):
        state_dict = dict(zip(self.state_dict().keys(), state_dict.values()))
        return super().load_state_dict(state_dict, strict, assign)


if __name__=='__main__':


    from PIL import Image
    from utils import preprocess_image
    from labels import labels
    import matplotlib.pyplot as plt
    model = BurrahModel()
    
    model.load_state_dict(torch.load("mobilenet_v2-b0353104.pth"))
    # model.load_state_dict(new_dict)
    model.eval()
    
    image = Image.open('hen.jpeg')
    # image = Image.open('dog.jpg')
    image = preprocess_image(image).float()
    output = torch.softmax(model(image), dim=1)
    print(output.shape)
    label = torch.argmax(output).item()
    label = labels[label]
    print(label)
    print(image.shape)
    
