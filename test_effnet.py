import unittest
from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from mobilenet_v2 import MobileNetV2

torch.manual_seed(43)

def load_state_dict(model, that_model):
    my_model = model.state_dict()
    their_model = that_model.state_dict()

    # print("our\n----")
    # print(MobileNetV2())
    # print("thier\n----")
    # print(mobilenet_v2())

    combined_dict = {}
    for my, their in zip(my_model.items(), their_model.items()):
        my_key, my_value = my
        their_key, their_value = their
        combined_dict[my_key] = their_value

    return combined_dict 

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
            # print(f"{their_key:<49}    {str(their_value.shape):<30} | {str(my_value.shape):<30}  {my_key}")
            self.assertEqual(my_value.shape, their_value.shape)
        
         self.assertEqual(len(my_model.items()), len(their_model.items()))

    def test_prediction(self):
        their_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        their_model.eval()
        my_model = MobileNetV2()
        state_dict = load_state_dict(my_model, their_model)
        my_model.load_state_dict(state_dict)
        my_model.eval()
        image = Image.open('hen.jpeg')
        from utils import preprocess_image
        image = preprocess_image(image)
        print(torch.argmax(their_model(image)))
        print(torch.argmax(my_model(image)))
        # self.assertEqual(their_model.state_dict().values(), my_model.state_dict().values())
        # self.assertTrue(torch.allclose(their_model(image), my_model(image), atol=1e-3))

if __name__ == "__main__":
    unittest.main()