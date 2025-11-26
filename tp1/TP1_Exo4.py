import cv2


im1 = cv2.imread("Images/pepper.bmp")


if im1 is None:
    print("Erreur : impossible de lire l'image 'Images/pepper.jpg'")
    exit()


h, w = im1.shape[:2]
print(f"Taille de l'image : hauteur = {h}, largeur = {w}")


im2 = cv2.resize(im1, (w // 2, h // 2))


y_start, y_end = 30, 150
x_start, x_end = 200, 400
im3 = im1[y_start:y_end, x_start:x_end]


im4 = im1.copy()

cv2.rectangle(im4, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

cv2.putText(im4, "Mon rectangle", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


im5 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
# im5=cv2.warpAffine(im1, (200, 200), (200, 200))

im6 = cv2.GaussianBlur(im1, (15, 15), 0)


cv2.imshow("Image originale (im1)", im1)
cv2.imshow("Image redimensionnée (im2)", im2)
cv2.imshow("Image cropée (im3)", im3)
cv2.imshow("Image avec rectangle et texte (im4)", im4)
cv2.imshow("Image tournée (im5)", im5)
cv2.imshow("Image floutée (im6)", im6)

cv2.waitKey(0)
cv2.destroyAllWindows()
