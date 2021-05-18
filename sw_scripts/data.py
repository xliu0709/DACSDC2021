from PIL import Image
import numpy as np

img = Image.open("boat6_0.jpg")
img = img.convert("RGB")
# img = cv2.resize(img, (320, 160))

img = np.asarray(img)
print(img.shape)

img.tofile("boat6_0.bin")
