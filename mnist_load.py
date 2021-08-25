import pandas as pd;
import numpy as np;
import tensorflow as tf;
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 텐서플로를 사용하면서 에러를 잡아주기 위한...

loaded_model = keras.models.load_model("mnist.h5")
print('Completed Load ...')

from tensorflow.keras.datasets.mnist import load_data;

(x_train, y_train),(x_test, y_test) = load_data(path='mnist.npz');

x_test = x_test.reshape(x_test.shape[0], 28* 28);
x_test_scaler = MinMaxScaler().fit_transform(x_test);

result = loaded_model.predict(x_test_scaler);

print(str(np.argmax(result[0], axis=-1)))