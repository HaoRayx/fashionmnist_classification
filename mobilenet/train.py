import numpy as np
import mnist_reader
from tqdm import tqdm
from scipy import misc
import tensorflow as tf
from skimage.transform import resize
import time

np.random.seed(2017)

tf.random.set_seed(2017)

X_train, y_train = mnist_reader.load_mnist('../fashion-mnist-master/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../fashion-mnist-master/data/fashion', kind='t10k')

height,width = 56,56

from keras.applications.mobilenet import MobileNet
from keras.layers import Input,Dense,Dropout,Lambda
from keras.models import Model
from keras import backend as K

input_image = Input(shape=(height,width))
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(input_image)
base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')
output = Dropout(0.5)(base_model.output)
predict = Dense(10, activation='softmax')(output)

model = Model(inputs=input_image, outputs=predict)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

X_train = X_train.reshape((-1, 28, 28))
X_train_resized = np.array([resize(x, (height, width)).astype(float) for x in tqdm(X_train)])
X_train_resized /= 255.0


X_test = X_test.reshape((-1, 28, 28))
X_test_resized = np.array([resize(x, (height, width)).astype(float) for x in tqdm(X_test)])

X_test_resized /= 255.0

model.fit(X_train_resized, y_train, batch_size=64, epochs=10, validation_data=(X_test_resized, y_test))

valstart = time.time()
model.predict(X_train_resized)
valend = time.time()
print("%.6f seconds",(valend-valstart)/60000)
