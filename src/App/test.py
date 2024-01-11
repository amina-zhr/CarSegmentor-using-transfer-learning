import pickle
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seg import mask2img

def get_mask(img, radio, radio2):
    image = np.array(Image.open(img).convert("RGB"))
    transform = A.Compose(
            [
                A.Resize(height=160, width=240),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
    )
    transformed_img = transform(image=image)
    transformed_img = transformed_img['image'].unsqueeze(0)
    if (radio == "Pre-trained on ImgaeNet"):
        with open("UnetVGG16.pickle", "rb") as f :
            model = pickle.load(f)
    elif (radio == "Not pre-trained"):
        with open("Unet.pickle", "rb") as f :
            model = pickle.load(f)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(transformed_img))
        preds = (preds > 0.5).float()

    preds_image = preds.squeeze().cpu().numpy()

    preds_image = Image.fromarray(preds_image * 255)
    preds_image = preds_image.convert('RGB')
    if radio2 == "Mask":
        return preds_image.resize(Image.open(img).size)
    elif radio2 in ["Cropped", "Highlighted"]:
        return mask2img(Image.open(img), preds_image, radio2)