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

class NoSEBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expension_factor: int, stride=1, isSE: bool= True):
        super().__init__()

        expension_factor = expension_factor
        exp_channels = expension_factor*in_channels
        self.conv2 = Conv2dNormActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3,stride=stride, groups=exp_channels)
        self.conv3 = nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        # x += res

        return x

class ResedualBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expension_factor: int, stride=1, isSE: bool= True):
        super().__init__()

        expension_factor = expension_factor
        exp_channels = expension_factor*in_channels
        self.conv1 = Conv2dNormActivation(in_channels=in_channels, out_channels=exp_channels, kernel_size=1, stride=1)
        self.conv2 = Conv2dNormActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=3,stride=stride, groups=exp_channels)
        self.conv3 = nn.Conv2d(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        # x += res

        return x

class BottleNeckBigBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expension_factor, number):
        super().__init__()
        self.l = [ResedualBottleNeck(in_channels, in_channels, expension_factor, stride=2)]
        for i in range(number-2):
            self.l.append(ResedualBottleNeck(in_channels, in_channels, expension_factor))
        self.l.append(ResedualBottleNeck(in_channels, out_channels, expension_factor))
        self.block = nn.Sequential(*self.l)

    def forward(self, x):
        return self.block(x)


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x

#mobile net 
class MobileNetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.normact = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=(2,2))
        self.NOSEBlock = NoSEBN(32, 16, 1) 
        self.BottleneckBlocks =  nn.Sequential(ResedualBottleNeck(in_channels=16, out_channels=24,expension_factor=6, stride=1),
        BottleNeckBigBlock(in_channels=24, out_channels=32,expension_factor=6, number=2),
        BottleNeckBigBlock(in_channels=32, out_channels=64,expension_factor=6, number=3),
        BottleNeckBigBlock(in_channels=64, out_channels=96,expension_factor=6, number=4),
        BottleNeckBigBlock(in_channels=96, out_channels=160,expension_factor=6, number=3),
        BottleNeckBigBlock(in_channels=160, out_channels=320,expension_factor=6, number=3),
        Conv2dNormActivation(in_channels=320, out_channels=1280, kernel_size=1, stride=(1,1)))
        self.classifier = Classifier(1280, 1000, 0.25)

    def forward(self, image: torch.Tensor):
        x = image
        x = self.normact(x)
        x = self.NOSEBlock(x)
        x = self.BottleneckBlocks(x)
        x = self.classifier(x)

        return x


if __name__=='__main__':

    def load_state_dict(model, path: str):
        my_model = model.state_dict()
        their_model = torch.load(path)
        # print("our\n----")
        # print(MobileNetV2())
        # print("thier\n----")
        # print(mobilenet_v2())


        for my, their in zip(my_model.items(), their_model.items()):
            my_key, my_value = my
            their_key, their_value = their
            my_value = their_value
            
    from PIL import Image
    from utils import preprocess_image
    import matplotlib.pyplot as plt
    model = MobileNetV2()
    
    load_state_dict(model, path="mobilenet_v2-b0353104.pth") 
    model.eval()
    
    image = Image.open('hen.jpeg')
    image = preprocess_image(image).float()
    print(f"{image.dtype=}")
    output = model(image)
    print(output.shape)
    label = torch.argmax(output)
    print(label,output[0][label] )
    print(image.shape)
    