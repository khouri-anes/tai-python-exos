import cv2
import numpy as np
import matplotlib.pyplot as plt

im1 = cv2.imread('Images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('Images/pepper.bmp')


hist1 = cv2.calcHist([im1], [0], None, [256], [0, 256])


hist2_B = cv2.calcHist([im2], [0], None, [256], [0, 256])
hist2_G = cv2.calcHist([im2], [1], None, [256], [0, 256])
hist2_R = cv2.calcHist([im2], [2], None, [256], [0, 256])


plt.figure(figsize=(12,6))


plt.subplot(1, 2, 1)
plt.title("Histogram of Grayscale Image")
plt.plot(hist1, color='black')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')

plt.subplot(1, 2, 2)
plt.title("Histograms of Colour Image")
plt.plot(hist2_B, color='blue', label='Blue Channel')
plt.plot(hist2_G, color='green', label='Green Channel')
plt.plot(hist2_R, color='red', label='Red Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.legend()

plt.tight_layout()
plt.show()


def my_histogram(image, channel=None):

    if channel is not None:
        image = image[:, :, channel]
    hist = np.zeros(256, dtype=int)
    for value in image.ravel():
        hist[value] += 1
    return hist



histp1 = my_histogram(im1)
histp2_B = my_histogram(im2, 0)
histp2_G = my_histogram(im2, 1)
histp2_R = my_histogram(im2, 2)


print("Compare histograms (OpenCV vs Manual):")
print("Grayscale match:", np.allclose(hist1.flatten(), histp1))
print("Blue match:", np.allclose(hist2_B.flatten(), histp2_B))
print("Green match:", np.allclose(hist2_G.flatten(), histp2_G))
print("Red match:", np.allclose(hist2_R.flatten(), histp2_R))

plt.plot(histp1)
plt.title("Manual Histogram")
plt.xlabel("Intensity (0–255)")
plt.ylabel("Pixel Count")
plt.show()

