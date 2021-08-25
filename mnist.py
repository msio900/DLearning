import pandas as pd;
import numpy as np;
import tensorflow as tf;
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 텐서플로를 사용하면서 에러를 잡아주기 위한...

# Data Loading
from tensorflow.keras.datasets.mnist import load_data;
(x_train, y_train),(x_test, y_test) = load_data(path='mnist.npz');

print(x_train.shape, y_train.shape)
print(x_train)
print('-'*120)
print(y_train)

# Show image
import matplotlib.pyplot as plt;
img = x_train[7, :];
print(img);
label = y_train[7];
plt.figure();
plt.imshow(img);
plt.title('%d %d' % (7, label), fontsize=15);
plt.show();
