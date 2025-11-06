import cv2
import numpy as np


def sp_noise(image, prob):

    output = image.copy()


    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:
            black = [0, 0, 0]
            white = [255, 255, 255]
        else:
            black = 0
            white = 255

    probs = np.random.random(output.shape[:2])


    output[probs < (prob / 2)] = black


    output[probs > 1 - (prob / 2)] = white

    return output



im1 = cv2.imread('Images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)


if im1 is None:
    print("Erreur: Impossible de charger l'image")
    exit()


print("Application du bruit Gaussien...")
gaussien_noise = np.zeros(im1.shape, np.float32)
cv2.randn(gaussien_noise, 0, 10)

noisy_img_float = cv2.add(im1.astype(np.float32), gaussien_noise)
im2 = np.clip(noisy_img_float, 0, 255).astype(np.uint8)


cv2.imshow("Image originale (im1)", im1)
cv2.imshow("Image avec bruit Gaussien (im2)", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Images/cameraman_bruit_gauss_sig0_001.bmp', im2)
print("Image avec bruit Gaussien sauvegardée")


print("\nApplication du bruit sel et poivre...")
im3 = sp_noise(im1, 0.1)


cv2.imshow("Image originale (im1)", im1)
cv2.imshow("Image avec bruit sel et poivre (im3)", im3)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Images/cameraman_bruit_sel_poivre_p_10.bmp', im3)
print("Image avec bruit sel et poivre sauvegardée")


print("\nComparaison des trois images...")
cv2.imshow("Image originale (im1)", im1)
cv2.imshow("Image avec bruit Gaussien (im2)", im2)
cv2.imshow("Image avec bruit sel et poivre (im3)", im3)
cv2.waitKey(0)
cv2.destroyAllWindows()




