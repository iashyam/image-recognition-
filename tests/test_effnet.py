import tests
import torch
from torchvision.models import mobilenet_v2


if __name__ == "__main__":

    model = mobilenet_v2()
    print(model)