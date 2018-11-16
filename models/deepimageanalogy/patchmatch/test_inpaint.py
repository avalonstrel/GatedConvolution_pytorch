from .PatchMatchCuda import PatchMatch
from PIL import Image
import numpy as np

def save_image(name, array):
    img = Image.fromarray(array.astype(np.uint8))
    img.save(name)

img_a = np.array(Image.open('bike_a.png').convert("RGB"))
img_b = np.array(Image.open('bike_b.png').convert("RGB"))
masks = np.zeros(shape=(img_a.shape[0], img_a.shape[1])).astype(np.int32) + 1
masks[img_a.shape[0]/4:img_a.shape[0]/2, img_a.shape[1]/4:img_a.shape[1]/2]
img_b = img_b*masks

pm = PatchMatch(img_a, img_a, img_b, img_b, 5)
pm.propagate()
recon_b = pm.reconstruct_avg(img_b)

save_image("recon_b.png", recon_b)
