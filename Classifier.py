import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps

x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())

classes=['0','1','2','3','4','5','6','7','8','9']
numberOfClasses=10

XTrain,XTest,YTrain,YTest=train_test_split(x,y,train_size=7500,test_size=2500,random_state=9)
XTrainScaled=XTrain/255
XTestScaled=XTest/255

classifier=LogisticRegression(solver='saga',multi_class='multinomial')
classifier.fit(XTrainScaled,YTrain)

ypredict=classifier.predict(XTestScaled)
accuracy=accuracy_score(YTest,ypredict)
print(accuracy)

def get_prediction(image):
        image_pil = Image.open(image)

        image_pil_gray = image_pil.convert('L')

        image_pil_gray_resize = image_pil_gray.resize((28,28), Image.ANTIALIAS)

        pixel_filter = 20

        min_pixel = np.percentile(image_pil_gray_resize, pixel_filter)

        image_pil_gray_resize_scaled = np.clip(image_pil_gray_resize - min_pixel, 0, 255)

        max_pixel = np.max(image_pil_gray_resize)

        image_pil_gray_resize_scaled = np.asarray(image_pil_gray_resize_scaled) / max_pixel

        test_sample = np.array(image_pil_gray_resize_scaled).reshape(1,784)

        test_prediction = classifier.predict(test_sample)

        return test_prediction[0] 
