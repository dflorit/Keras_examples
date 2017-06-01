import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, Flatten, Input, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

num_samples = 200
n_epochs = 1

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
model.fit(X_train, y_train, epochs= n_epochs, batch_size=32)

#Getting the names of each layer in the VGG Model:
for i, layer in enumerate(model_vgg16_conv.layers):
	print(i, layer.name)

#Getting the names of each layer in the new model. these contain the VGG layers and the extra ones on top:
print("++++++++++++++++++++++++++++++++++++")
for i, layer in enumerate(model.layers):
	print(i, layer.name)

#Freezing layers up to layer 10
for layer in model.layers[:10]:
	layer.trainable = False
for layer in model.layers[10:]:
	layer.trainable = True

#Re-compiling and training model:
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= n_epochs, batch_size=32)

#This function gets the output of the layer you want. Right now it's getting the layer right before the last one (dense_1)
get_feature = K.function([model.layers[0].input], [model.layers[-2].output])
feature = get_feature([X_train])
print feature

