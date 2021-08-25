import tensorflow as tf;
import numpy as np;
print(tf.__version__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


x = [-2,3,20,0,-5,7,-23,-8, 32, -15]
y = [-1, 4, 21, 1, -4, 8, -22, -7, 33, -14]

x_train = np.array(x).reshape(-1,1);
y_train = np.array(y)

print(x_train);
print(y_train);
print(x_train.shape);
print(y_train.shape);

from tensorflow.keras import Sequential;
from tensorflow.keras.layers import Dense; # dense는 밀도를?

model = Sequential();
model.add(Dense(units=1, activation='linear', input_dim=1));
model.compile(optimizer='adam', loss='mse', metrics=['mae']);
model.fit(x_train, y_train, epochs=3000, verbose=0);

x_data = [8, 9, 10, -1, 4, -15];
x_data = np.array(x_data).reshape(-1,1);

result = model.predict(x_data);
print(result);