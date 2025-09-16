import unittest
import torch
from torchvision.models import mobilenet_v2
from load_net import MobileNetV2

           
class test_mobile_net(unittest.TestCase):
        
    def test_mobilnet(self):
         my_model = MobileNetV2().state_dict()
         their_model = mobilenet_v2().state_dict()
         print("our\n----")
         print(MobileNetV2())
         print("thier\n----")
         print(mobilenet_v2())


         for my, their in zip(my_model.items(), their_model.items()):
            my_key, my_value = my
            their_key, their_value = their
            print(f"{their_key:<49}    {str(their_value.shape):<30} | {str(my_value.shape):<30}  {my_key}")
            self.assertEqual(my_value.shape, their_value.shape)


if __name__ == "__main__":
    unittest.main()