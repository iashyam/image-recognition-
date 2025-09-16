import unittest
import torch
from torchvision.models import mobilenet_v2
from load_net import MobileNetV2

class test_mobile_net(unittest.TestCase):
        
    def test_mobilnet(self):
        my_model_dict = [x.shape for x in MobileNetV2().state_dict().values()]
        model_dict = [x.shape for x in mobilenet_v2().state_dict().values()]
        
        for x, y in zip(my_model_dict, model_dict):
            print(x, y)
            self.assertEqual(x, y)

if __name__ == "__main__":
    unittest.main()