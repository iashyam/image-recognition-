import unittest
from PIL import Image
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from mobilenet_v2 import BurrahModel
from labels import labels
torch.manual_seed(43)


def predict(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    max_prob, max_index = torch.max(probabilities, dim=0)
    label = labels[max_index.item()]
    return label, max_prob.item()

class test_mobile_net(unittest.TestCase):
        
    def test_architecture(self):
    
         my_model = BurrahModel().state_dict()
         their_model = mobilenet_v2().state_dict()

         for my, their in zip(my_model.items(), their_model.items()):
            my_key, my_value = my
            their_key, their_value = their
            # print(f"{their_key:<49}    {str(their_value.shape):<30} | {str(my_value.shape):<30}  {my_key}")
            self.assertEqual(my_value.shape, their_value.shape)
        
         self.assertEqual(len(my_model.items()), len(their_model.items()))

    def test_prediction(self):
        
        their_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        their_model.eval()
        my_model = BurrahModel()
        my_model.load_state_dict(their_model.state_dict())
        my_model.eval()

        # image = Image.open('hen.jpeg')
        image = Image.open('dog.jpg')
        from utils import preprocess_image
        image = preprocess_image(image)

        their_label, their_prob = predict(their_model, image)
        my_label, my_prob = predict(my_model, image)
        print(f"Their model's output: {their_label} with probability {their_prob:.4f}")
        print(f"My model's output    : {my_label} with probability {my_prob:.4f}")

        for a, b in zip(their_model.state_dict().values(), my_model.state_dict().values()):
            self.assertTrue(torch.allclose(a, b))
        
        self.assertTrue(torch.allclose(their_model(image), my_model(image), atol=1e-3))

        
if __name__ == "__main__":
    unittest.main()