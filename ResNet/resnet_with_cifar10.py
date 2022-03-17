# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import os
import time

from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization
from keras.layers import Add
from keras.callbacks import ReduceLROnPlateau

"""Tensorboard Setting"""

root_logdir = os.path.join(os.curdir, "logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

"""Data load"""

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Labeling
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Slicing
validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

# Show image
plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')

# Resizing
def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 224x224
    image = tf.image.resize(image, (224,224))
    return image, label

# preprocessing & batching
train_ds = (train_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))

"""Model building"""

def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='relu'):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    
    return x
    
def plain18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    
    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3))
    
    conv3_1 = conv2d_bn(conv2_3, 128, (3, 3), strides=2) # (28, 28, 128)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3))
    
    conv4_1 = conv2d_bn(conv3_2, 256, (3, 3), strides=2) # (14, 14, 256)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3))
    
    conv5_1 = conv2d_bn(conv4_2, 512, (3, 3), strides=2) # (7, 7, 512)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3))
    

    gap = GlobalAveragePooling2D()(conv5_2)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='Plain18')
        
    return model

def ResNet18(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    
    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = conv2d_bn(conv2_1, 64, (3, 3))
    conv2_3 = conv2d_bn(conv2_2, 64, (3, 3), activation=None) # (56, 56, 64)
    
    shortcut_1 = Add()([conv2_3, conv2_1])
    shortcut_1 = Activation(activation='relu')(shortcut_1) # (56, 56, 64)

    
    conv3_1 = conv2d_bn(shortcut_1, 128, (3, 3), strides=2)
    conv3_2 = conv2d_bn(conv3_1, 128, (3, 3)) # (28, 28, 128)
    
    shortcut_2 = conv2d_bn(shortcut_1, 128, (1, 1), strides=2, activation=None) # (56, 56, 64) -> (28, 28, 128)
    shortcut_2 = Add()([conv3_2, shortcut_2])
    shortcut_2 = Activation(activation='relu')(shortcut_2) # (28, 28, 128)

    
    conv4_1 = conv2d_bn(shortcut_2, 256, (3, 3), strides=2)
    conv4_2 = conv2d_bn(conv4_1, 256, (3, 3)) # (14, 14, 256)
    
    shortcut_3 = conv2d_bn(shortcut_2, 256, (1, 1), strides=2, activation=None) # (28, 28, 128) -> (14, 14, 256)
    shortcut_3 = Add()([conv4_2, shortcut_3])
    shortcut_3 = Activation(activation='relu')(shortcut_3) # (14, 14, 256)
    
    
    conv5_1 = conv2d_bn(shortcut_3, 512, (3, 3), strides=2)
    conv5_2 = conv2d_bn(conv5_1, 512, (3, 3)) # (7, 7, 512)
    
    shortcut_4 = conv2d_bn(shortcut_3, 512, (1, 1), strides=2, activation=None) # (14, 14, 256) -> (7, 7, 512)
    shortcut_4 = Add()([conv5_2, shortcut_4])
    shortcut_4 = Activation(activation='relu')(shortcut_4) # (7, 7, 512)
    

    gap = GlobalAveragePooling2D()(shortcut_4)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='ResNet18')
        
    return model

def bottleneck_identity(input_tensor, filter_sizes):
    filter_1, filter_2, filter_3 = filter_sizes
    
    x = conv2d_bn(input_tensor, filter_1, (1, 1))
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)
    
    shortcut = Add()([input_tensor, x])
    shortcut = Activation(activation='relu')(shortcut)
    
    return shortcut

def bottleneck_projection(input_tensor, filter_sizes, strides=2):
    filter_1, filter_2, filter_3 = filter_sizes
    
    x = conv2d_bn(input_tensor, filter_1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_2, (3, 3))
    x = conv2d_bn(x, filter_3, (1, 1), activation=None)
    
    projected_input = conv2d_bn(input_tensor, filter_3, (1, 1), strides=strides, activation=None)
    shortcut = Add()([projected_input, x])
    shortcut = Activation(activation='relu')(shortcut)
    
    return shortcut

def ResNet50(model_input, classes=10):
    conv1 = conv2d_bn(model_input, 64, (7, 7), strides=2, padding='same') # (112, 112, 64)
    
    conv2_1 = MaxPooling2D((3, 3), strides=2, padding='same')(conv1) # (56, 56, 64)
    conv2_2 = bottleneck_projection(conv2_1, [64, 64, 256], strides=1)
    conv2_3 = bottleneck_identity(conv2_2, [64, 64, 256])
    conv2_4 = bottleneck_identity(conv2_3, [64, 64, 256])# (56, 56, 256)
    
    conv3_1 = bottleneck_projection(conv2_4, [128, 128, 512])
    conv3_2 = bottleneck_identity(conv3_1, [128, 128, 512])
    conv3_3 = bottleneck_identity(conv3_2, [128, 128, 512])
    conv3_4 = bottleneck_identity(conv3_3, [128, 128, 512]) # (28, 28, 512)
    
    conv4_1 = bottleneck_projection(conv3_4, [256, 256, 1024])
    conv4_2 = bottleneck_identity(conv4_1, [256, 256, 1024])
    conv4_3 = bottleneck_identity(conv4_2, [256, 256, 1024])
    conv4_4 = bottleneck_identity(conv4_3, [256, 256, 1024])
    conv4_5 = bottleneck_identity(conv4_4, [256, 256, 1024])
    conv4_6 = bottleneck_identity(conv4_5, [256, 256, 1024]) # (14, 14, 1024)
    
    conv5_1 = bottleneck_projection(conv4_6, [512, 512, 2048])
    conv5_2 = bottleneck_identity(conv5_1, [512, 512, 2048])
    conv5_3 = bottleneck_identity(conv5_2, [512, 512, 2048]) # (7, 7, 2048)

    gap = GlobalAveragePooling2D()(conv5_3)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(gap) # 'softmax'
    
    model = Model(inputs=model_input, outputs=model_output, name='ResNet50')
        
    return model

inputs = keras.Input(shape=(224,224,3), name="input")

# model = ResNet18(inputs, 10)
model = ResNet50(inputs, 10)

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.1, decay=0.0001, momentum=0.9), metrics=['accuracy'])
model.summary()

checkpoint_filepath = '{epoch:02d}-{val_accuracy:.5f}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=1,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)

model.fit(train_ds,
          epochs=25,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb, model_checkpoint_callback])

print(test_ds.take(1))
for i, (image, label) in enumerate(test_ds.take(1)):
  predictions = model.predict(image)
  for i in range(20):
    print("real : ",CLASS_NAMES[int(label[i])])
    print("predict : ", CLASS_NAMES[np.argmax(predictions[i])])