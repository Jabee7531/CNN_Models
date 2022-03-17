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

plt.figure(figsize=(20,20))
for i, (image, label) in enumerate(train_ds.take(5)):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(image)
    plt.title(CLASS_NAMES[label.numpy()[0]])
    plt.axis('off')

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 227x227
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))
validation_ds = (validation_ds
                  .map(process_images)
                  .batch(batch_size=20, drop_remainder=True))

def AlexNet(model_input, classes=10):
  x = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3), name='Conv1')(model_input)
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool1')(x)

  x = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same', name='Conv2')(x)
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool2')(x)

  x = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='Conv3')(x)

  x = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='Conv4')(x)

  x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='Conv5')(x)
  x = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name='MaxPool3')(x)

  x = keras.layers.Flatten(name='Flatten')(x)

  x = keras.layers.Dense(4096, activation='relu', name='Dense1')(x)
  x = keras.layers.Dropout(0.5, name='DropOut1')(x)

  x = keras.layers.Dense(4096, activation='relu', name='Dense2')(x)
  x = keras.layers.Dropout(0.5, name='DropOut2')(x)

  model_output = keras.layers.Dense(classes, activation='softmax', name ='SoftMax')(x)

  model = keras.models.Model(inputs=model_input, outputs=model_output, name='AlexNet')

  return model

inputs = keras.Input(shape=(227,227,3), name="input")

model = AlexNet(inputs)

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()

checkpoint_filepath = '{epoch:02d}-{val_loss:.5f}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=1,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb, model_checkpoint_callback])

print(test_ds.take(1))
for i, (image, label) in enumerate(test_ds.take(1)):
  predictions = model.predict(image)
  for i in range(20):
    print("real : ",CLASS_NAMES[int(label[i])])
    print("predict : ", CLASS_NAMES[np.argmax(predictions[i])])