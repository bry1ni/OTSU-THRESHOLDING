import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ------ IMAGE NET --------#
img1Path = r"/Users/rayanpicso/Desktop/MIV/TAI/OTSU/net.png"

imgNet = cv2.imread(img1Path)
gray = cv2.cvtColor(imgNet, cv2.COLOR_BGR2GRAY)

opt1, netOtsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# ------ IMAGE BRUITÉE --------#
img2Path = r"/Users/rayanpicso/Desktop/MIV/TAI/OTSU/bruit.png"

imgBruit = cv2.imread(img2Path)
gray = cv2.cvtColor(imgBruit, cv2.COLOR_BGR2GRAY)

opt2, bruitOtsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# ------ DISPLAY --------#

plt.figure(figsize=(12, 8))
plt.subplot(1, 4, 1)
plt.title('Image originale net')
plt.imshow(imgNet, cmap='Blues')

plt.subplot(1, 4, 2)
plt.title('Seuil Otsu')
plt.imshow(netOtsu, cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Image originale bruitée')
plt.imshow(imgBruit, cmap='Blues')

plt.subplot(1, 4, 4)
plt.title('Seuil Otsu')
plt.imshow(bruitOtsu, cmap='gray')

plt.show()

print(opt1)
print(opt2)
