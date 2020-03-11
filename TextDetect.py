import pytesseract
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
"""
cap1 = cv2.VideoCapture(0)
while True:
    ret, frame = cap1.read()
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
cap1.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    print(frame)
else:
    ret = False

img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.title('Image Camera-1')
plt.xticks([])
plt.yticks([])
plt.show()
name=input("Enter Photo Name:")
cv2.imwrite(name+'.jpg',img1)
cap.release()
"""
img = Image.open("Galib.jpg")
text = pytesseract.image_to_string(img)
print(text)





