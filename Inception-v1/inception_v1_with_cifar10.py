# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import time

root_logdir = os.path.join(os.curdir, "logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

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

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    #1x1
    conv_1x1 = keras.layers.Conv2D(filters=filters_1x1, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(x)

    #3x3
    conv_3x3_reduce = keras.layers.Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(x)
    conv_3x3 = keras.layers.Conv2D(filters=filters_3x3, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu')(conv_3x3_reduce)

    #5x5
    conv_5x5_reduce = keras.layers.Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(x)
    conv_5x5 = keras.layers.Conv2D(filters=filters_5x5, kernel_size=(5, 5), strides=(1,1), padding='same', activation='relu')(conv_5x5_reduce)

    #pool_proj
    max_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = keras.layers.Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1,1), padding='same', activation='relu')(max_pool)

    output = keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output


def auxiliary(x, name=None):
    x = keras.layers.AveragePooling2D((5, 5), strides=(3,3), name='avg_pool_'+name)(x)
    x = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu', name='conv_'+name)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu', name='dense_'+name)(x)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dense(10, activation='softmax', name='output_'+name)(x)

    return x
  

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epoch_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
    return lrate

def googlenet(model_input, classes=10):
  # Layer 1
  x = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', input_shape=(227,227,3), name='conv_1_7x7/2')(model_input)
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_1')(x)

  # Layer 2
  x = keras.layers.Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', name='conv_2_1x1/1')(x)
  x = keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv_2_3x3/1')(x)
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_2')(x)

  # Layer 3
  x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
  x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_3')(x)
  
  # Layer 4
  x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
  x1 = auxiliary(x, name='aux_1')
  x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
  x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
  x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
  x2 = auxiliary(x, name='aux_2')
  x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same', name='max_pool_4')(x)

  #Layer 5
  x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
  x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b')

  x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = keras.layers.Dropout(0.4, name='drop_out')(x)
  model_output = keras.layers.Dense(classes, activation='softmax', name='softmax')(x)
  
  model = keras.models.Model(inputs=model_input, outputs=model_output, name='GoogLeNet')

  return model

inputs = keras.Input(shape=(224,224,3), name="input")

model = googlenet(inputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
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