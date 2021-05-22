import numpy as np
import matplotlib.pyplot as plt

mnist = np.load('mnist.npz')

x_train = mnist['x_train']
y_train = mnist['y_train']
x_test = mnist['x_test']
y_test = mnist['y_test']

plt.imshow(x_train[38], cmap='gray')
plt.show()

import cv2
img = cv2.imread('dog.jpg')
'''
cv2.namedWindow('test', cv2.WINDOW_NORMAL) # 크기 변경 가능
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #컬러 변환
'''
dst = img[:,:,::-1] #BGR/RGB 변환
plt.imshow(dst)

from PIL import Image
img = Image.open('dog.jpg')
img = np.array(img)

