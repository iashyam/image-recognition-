import requests
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO
import torch
from labels import labels
## loading and prepareing efficient net
eff_net_transforms = EfficientNet_B2_Weights.DEFAULT.transforms()
network = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
network.eval()
#importing labels for correct labeling
# lables_ = "https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/refs/heads/master/examples/simple/labels_map.txt"
# response = requests.get(lables_)
# eff_net_LABELS = eval(response.text.split("\n")[0])
# eff_net_LABELS = {int(k):v for k, v in eff_net_LABELS.items()}

eff_net_LABELS = labels
def recoganize_image(image: Image.Image):
	image_transfomed = eff_net_transforms(image).unsqueeze(0)
	with torch.no_grad():
		pred = network(image_transfomed)
		pred_softmax = torch.nn.functional.softmax(pred, dim=1).argmax()
	label = eff_net_LABELS[pred_softmax.item()]
	print(label)
	return label


#plot a random image to see the dataset
def image_from_a_link(link: str):
    response = requests.get(link)
    image = Image.open(BytesIO(response.content))
    return image 
	
if __name__ == "__main__":
	link = "https://imgs.search.brave.com/uiwpUrnRL20-jsinT_ii5Ri0U_AHw8R8KINqJVEtNL0/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wMjQv/NTcwLzczOS9zbWFs/bC9wb3J0cmFpdC1v/Zi1hLWhhcHB5LXdl/bHNoLWNvcmdpLWRv/Zy1pbi1hdXR1bW4t/Zm9yZXN0LWFpLWdl/bmVyYXRlZC1mcmVl/LXBob3RvLmpwZw"

	link = input("Enter link: ")
	def recoganize_image_from_link(link: str):
		image = image_from_a_link(link)
		predicted = recoganize_image(image)
		plt.title(f"Predicted: {predicted} ")
		plt.imshow(image)
		plt.axis('off')
		plt.show()

	recoganize_image_from_link(link)
