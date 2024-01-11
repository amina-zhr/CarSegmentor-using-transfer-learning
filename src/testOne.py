from PIL import Image
import numpy as np
from test_func import segmentation_function

img_path = 'path/image'
image = np.array(Image.open(img_path).convert("RGB"))
segmentation_function(image).show()
