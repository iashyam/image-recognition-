import unittest
from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from load_net import MobileNetV2

torch.manual_seed(43)

def load_state_dict(model):
    my_model = model.state_dict()
    their_model = mobilenet_v2().state_dict()
    # print("our\n----")
    # print(MobileNetV2())
    # print("thier\n----")
    # print(mobilenet_v2())


    for my, their in zip(my_model.items(), their_model.items()):
        my_key, my_value = my
        their_key, their_value = their
        my_value = their_value
        
class test_mobile_net(unittest.TestCase):
        
    def test_mobilnet(self):
         my_model = MobileNetV2().state_dict()
         their_model = mobilenet_v2().state_dict()
        #  print("our\n----")
        #  print(MobileNetV2())
        #  print("thier\n----")
        #  print(mobilenet_v2())


         for my, their in zip(my_model.items(), their_model.items()):
            my_key, my_value = my
            their_key, their_value = their
            print(f"{their_key:<49}    {str(their_value.shape):<30} | {str(my_value.shape):<30}  {my_key}")
            self.assertEqual(my_value.shape, their_value.shape)
        
         self.assertEqual(len(my_model.items()), len(their_model.items()))

    def test_prediction(self):
        their_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        their_model.eval()
        my_model = MobileNetV2()
        load_state_dict(my_model)
        my_model.eval()
        image = Image.open('hen.jpeg')
        from utils import preprocess_image
        image = preprocess_image(image)
        self.assertAlmostEqual(their_model(image), my_model(image))

if __name__ == "__main__":
    unittest.main()