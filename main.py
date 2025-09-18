import requests
import torch
from utils import preprocess_image
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO
from mobilenet_v2 import BurrahModel

try:
	from labels import labels
except ModuleNotFoundError:
	print("Lables not found downloading now")
	lables_link = "https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/refs/heads/master/examples/simple/labels_map.txt"
	response = requests.get(lables_link)
	eff_net_LABELS = eval(response.text.split("\n")[0])
	labels = {int(k):v for k, v in eff_net_LABELS.items()}


model = BurrahModel()
model.eval()
state_dict_path = "mobilenet_v2-b0353104.pth"
if not state_dict_path in os.listdir():
	#TODO: implement loading from internet
	pass

model.load_state_dict(torch.load(state_dict_path))

def recoganize_image(image: Image.Image):
	image_transfomed = preprocess_image(image).float()
	with torch.no_grad():
		pred = model(image_transfomed)
		pred_softmax = torch.nn.functional.softmax(pred, dim=1).argmax()
	label = labels[pred_softmax.item()]
	return label

def plot_image(image: Image.Image):
	
	label = recoganize_image(image)
	plt.title(f"Predicted: {label} ")
	plt.imshow(image)
	plt.axis('off')
	plt.show()
	return label

def recoganize_image_from_a_link(link: str, plot: bool = True):
	response = requests.get(link)
	image = Image.open(BytesIO(response.content))
	label = recoganize_image(image)
	if plot:
		return plot_image(image)
	return label
	
if __name__ == "__main__":
	link = "https://images.pexels.com/photos/20787/pexels-photo.jpg"

	# label = recoganize_image_from_a_link(link)
	print(plot_image(Image.open("hen.jpeg")))

	# recoganize_image_from_link(link)
