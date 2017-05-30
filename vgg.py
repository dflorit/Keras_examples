import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Flatten, Input, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.optimizers import SGD
import numpy as np

num_samples = 200
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train[0:num_samples]
X_train = X_train.astype('float32')
X_train /= 255

y_train = y_train[0:num_samples]
y_train = keras.utils.to_categorical(y_train, 10)


print X_train.shape

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

input_vgg16_conv = model_vgg16_conv.input
output_vgg16_conv = model_vgg16_conv.output

x = GlobalAveragePooling2D()(output_vgg16_conv)
x = Dense(1024, activation ='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=input_vgg16_conv, outputs=predictions)

for layer in model_vgg16_conv.layers:
	layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)

for layer in model.layers[:172]:
	layer.trainable = False
for layer in model.layers[172:]:
	layer.trainable = True


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)


