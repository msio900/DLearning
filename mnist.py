import pickle

import pandas as pd;
import numpy as np;
import tensorflow as tf;
import os

from matplotlib import pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 텐서플로를 사용하면서 에러를 잡아주기 위한...

# Data Loading
from tensorflow.keras.datasets.mnist import load_data;
(x_train, y_train),(x_test, y_test) = load_data(path='mnist.npz');

print(x_test.shape, y_test.shape);
print(x_train.shape, y_train.shape);
print(x_train);
print('-'*120);
print(y_train);

# Show image
# import matplotlib.pyplot as plt;
# img = x_train[7, :];
# print(img);
# label = y_train[7];
# plt.figure();
# plt.imshow(img);
# plt.title('%d %d' % (7, label), fontsize=15);
# plt.show();

# 훈련 / 검증 데이터로 분리
# 데이터 정규화
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import MinMaxScaler;

x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train,
                                                    test_size=0.3,
                                                    random_state=777
                                                    );
print('훈련 데이터셋 :',x_train.shape, y_train.shape)
print('검증 데이터셋 :', x_vali.shape, y_vali.shape)

# x_train과 x_test모두 정규화를 시켜줘야함.

x_train = x_train.reshape(x_train.shape[0], 28 * 28)
print('x_train차원 축소 :', x_train.shape)
x_vali = x_vali.reshape(x_vali.shape[0], 28 * 28)
print('x_vali차원 축소 :', x_vali.shape)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)
print('x_test차원 축소 :', x_test.shape)



# 정규화!! (min max scaler는 3차원을 가능하지 않음.)
x_train_scaler = MinMaxScaler().fit_transform(x_train)
x_vali_scaler = MinMaxScaler().fit_transform(x_vali)
x_test_scaler = MinMaxScaler().fit_transform(x_test)

print('정규화된 형태',x_train_scaler[0, :])

# 데이터가 시리얼 형태로 들어가 있음.
# 데이터를 범주형으로 변환
print(y_train.shape)
print('원본 데이터:',y_train[1])
y_train_cate = to_categorical(y_train) # 원핫인코딩과 유사한 개념
print(y_train.shape)
print('to_categorical 이후:',y_train[1])
y_vali_cate = to_categorical(y_vali)
y_test_cate = to_categorical(y_test)

# 모델 구성 - 신경망 구성
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;



model = Sequential();
# 784개 들어와서 10개가 나감.
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # 다중분류라서...softmax

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc']);
model.fit(x_train_scaler, y_train_cate,
          epochs=30,
          batch_size=128, # 데이터를 128개 씩 나누어 메모리에 올리고 처리
          validation_data=(x_vali_scaler, y_vali_cate),
          verbose=1);
print(model.evaluate(x_test_scaler, y_test_cate))

# with open("mnist.model", "wb") as w:
#     pickle.dump(model, w);

model.save("mnist.h5")


#
# result = model.predict(x_test_scaler);
# print(result.shape)
# print(result[0])
#
# plt.imshow(x_test[0].reshape(28,28))
# plt.title(str(result[0]))
# plt.show()