# อ่านรูปภาพมาทำงาน
import cv2

img = cv2.imread("image/cat.jpg")
print(type(img.ndim))
print(img)