import pickle
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import cv2

def mask_function(image):

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
    with open("model_pickle_vgg13.pickle", "rb") as f :
        model = pickle.load(f)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(transformed_img))
        preds = (preds > 0.5).float()

    mask_image= preds.squeeze().cpu().numpy()
    return mask_image * 255

def segmentation_function(image):
    mask = mask_function(image).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    image = image.astype(np.uint8)
    return Image.fromarray(cv2.bitwise_and(image, image, mask=mask))

