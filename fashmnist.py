import matplotlib.pyplot as plt
from tensorflow.keras.datasets.fashion_mnist import load_data;
(x_train, y_train), (x_test, y_test) = load_data();
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

class_name = ['Top','Trouser','Pullover','Dress','Coat','Sendal','Shirt','Sneaker','Bag','Boot'];


plt.imshow(x_train[10])
plt.title(class_name[y_train[10]]);
plt.show()