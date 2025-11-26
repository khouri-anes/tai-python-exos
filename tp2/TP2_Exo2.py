import cv2
import numpy as np
import time

def filtreMoyenne(im,r):
    im = im.astype(np.float32)
    hm=2*r+1
    taille=hm*hm
    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(r,h-r):
        for j in range(r,w-r):
            s=0.0
            for k in range(i-r,i+r+1):
                for m in range(j-r,j+r+1):
                    s=s+im[k,m]
            out[i,j]=s/taille


    return np.clip(out, 0, 255).astype(np.uint8)


def filtremedian2(im, r):
    im = im.astype(np.float32)  # prevent overflow
    hm = 2 * r + 1
    taille = hm * hm
    h, w = im.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(r, h - r):
        for j in range(r, w - r):
            s = []
            for k in range(i - r, i + r + 1):
                for m in range(j - r, j + r + 1):
                    s.append(im[k, m])
            out[i, j] = np.median(s)

    # clip and return
    return np.clip(out, 0, 255).astype(np.uint8)

def calculate_psnr(im1, im2):
    mse = np.mean((im1.astype(np.float32) - im2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    R = 255.0
    psnr = 10 * np.log10((R * R) / mse)
    return psnr













im1 = cv2.imread('Images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)

im2 = cv2.imread('Images/cameraman_bruit_gauss_sig0_001.bmp', cv2.IMREAD_GRAYSCALE)

im3 = cv2.imread('Images/cameraman_bruit_sel_poivre_p_10.bmp', cv2.IMREAD_GRAYSCALE)

imf2=cv2.blur(im2,(3,3))

imfp2=filtreMoyenne(im2,3)



imf3 = cv2.medianBlur(im3, 3)
imfm2 = cv2.medianBlur(im2, 3)


imf3_custom = filtremedian2(im3, 1)
imfm2_custom = filtremedian2(im2, 1)

#
# cv2.imshow(" (im1)", im2)
# cv2.imshow("(imf2)", imf2)
# cv2.imshow("(imfp2)", imfp2)


cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Original (im1)", im1)
cv2.imshow("Gaussian filtre (im2)", im2)
cv2.imshow("im3", im3)
cv2.imshow("Median  imf3", imf3)
cv2.imshow("Median Custom imf3_custom", imf3_custom)
cv2.imshow("Median OpenCV imfm2", imfm2)
cv2.imshow("Median Custom imfm2_custom", imfm2_custom)

cv2.waitKey(0)
cv2.destroyAllWindows()


psnr_im1_im2 = cv2.PSNR(im1, im2)
psnr_im1_imf2 = cv2.PSNR(im1, imf2)


imf3 = cv2.medianBlur(im3, 3)
imfm2 = cv2.medianBlur(im2, 3)

psnr_im1_im3 = cv2.PSNR(im1, im3)
psnr_im1_imf3 = cv2.PSNR(im1, imf3)
psnr_im1_imfm2 = cv2.PSNR(im1, imfm2)

print("----- PSNR with OpenCV -----")
print("PSNR(im1, im2) =", psnr_im1_im2)
print("PSNR(im1, imf2) =", psnr_im1_imf2)
print("PSNR(im1, im3) =", psnr_im1_im3)
print("PSNR(im1, imf3) =", psnr_im1_imf3)
print("PSNR(im1, imfm2) =", psnr_im1_imfm2)


print("\n----- PSNR with custom function -----")
print("PSNR(im1, im2) =", calculate_psnr(im1, im2))
print("PSNR(im1, imf2) =", calculate_psnr(im1, imf2))
print("PSNR(im1, im3) =", calculate_psnr(im1, im3))
print("PSNR(im1, imf3) =", calculate_psnr(im1, imf3))
print("PSNR(im1, imfm2) =", calculate_psnr(im1, imfm2))







results = []


for k in [3, 5, 7]:

    start = time.time()
    f2 = cv2.blur(im2, (k, k))
    t = time.time() - start
    results.append(("Moyenne", f"{k}x{k}", "im2", cv2.PSNR(im1, f2), t))


    start = time.time()
    f3 = cv2.blur(im3, (k, k))
    t = time.time() - start
    results.append(("Moyenne", f"{k}x{k}", "im3", cv2.PSNR(im1, f3), t))



for k in [3, 5]:
    for sigma in [2, 1.5, 1, 0.5]:

        start = time.time()
        f2 = cv2.GaussianBlur(im2, (k, k), sigma)
        t = time.time() - start
        results.append(("Gaussian", f"{k}x{k}, σ={sigma}", "im2", cv2.PSNR(im1, f2), t))

        start = time.time()
        f3 = cv2.GaussianBlur(im3, (k, k), sigma)
        t = time.time() - start
        results.append(("Gaussian", f"{k}x{k}, σ={sigma}", "im3", cv2.PSNR(im1, f3), t))



for k in [3, 5, 7]:

    start = time.time()
    f2 = cv2.medianBlur(im2, k)
    t = time.time() - start
    results.append(("Median", f"{k}x{k}", "im2", cv2.PSNR(im1, f2), t))

    start = time.time()
    f3 = cv2.medianBlur(im3, k)
    t = time.time() - start
    results.append(("Median", f"{k}x{k}", "im3", cv2.PSNR(im1, f3), t))



start = time.time()
f2 = cv2.bilateralFilter(im2, 7, 40, 40)
t = time.time() - start
results.append(("Bilateral", "d=7", "im2", cv2.PSNR(im1, f2), t))

start = time.time()
f3 = cv2.bilateralFilter(im3, 7, 40, 40)
t = time.time() - start
results.append(("Bilateral", "d=7", "im3", cv2.PSNR(im1, f3), t))



print("\n---- RESULTS ----")
print("Filter\t\tParams\t\tImage\tPSNR(dB)\tTime(s)")
for r in results:
    print(f"{r[0]:10} {r[1]:15} {r[2]:5} {r[3]:8.3f}   {r[4]:.6f}")
