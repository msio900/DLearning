import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras



loaded_model = keras.models.load_model("p121.h5")


result = loaded_model.predict(test_generator)

# 테스트 데이터 예측

import matplotlib.pyplot as plt
import cv2  # pip install opencv-python

image = cv2.imread(test_df['image'][7])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image);

desc = zip(class_col,list(result[7]))
desc_list = list(desc);
type = desc_list[0:6];
color = desc_list[6:11];
type = sorted(type, key=lambda z:z[1], reverse=True)
color = sorted(color, key=lambda z:z[1], reverse=True)

print(type[0][0],type[0][1])
print(color[0][0], color[0][1])