from PIL import Image
import numpy as np
import torch

def preprocess_image(img, target_size=224):
    
    # Resize so the shorter side is 256 (standard for B0)
    scale = 256.0 / min(img.size)
    new_size = tuple([int(x * scale) for x in img.size])
    img = img.resize(new_size, Image.BILINEAR)
    
    # Center crop
    width, height = img.size
    left = (width - target_size) / 2
    top = (height - target_size) / 2
    right = (width + target_size) / 2
    bottom = (height + target_size) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    
    # HWC to CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # Convert to tensor
    img_tensor = torch.tensor(img_np).unsqueeze(0)
    return img_tensor.float()

