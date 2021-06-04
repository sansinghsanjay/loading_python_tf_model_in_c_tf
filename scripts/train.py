'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To train a simple CNN on MNIST dataset
'''

# packages
import os
import tensorflow as tf
import numpy as np
import pandas as pd

# constants
IMG_W = 28
IMG_H = 28
IMG_C = 3
LR = 0.01
BATCH_SIZE = 70
CLASSES = 10
EPOCHS = 2

# paths
image_data_path = "/home/sansingh/Downloads/DATASET/mnist/trainingSet/"
train_csv_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/intermediate_outputs/train_data.csv"
val_csv_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/intermediate_outputs/val_data.csv"
target_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/trained_model/"

# function to generate data and pass in batches
def data_generator(train_df):
	while(True):
		image_batch = np.ndarray((BATCH_SIZE, IMG_W, IMG_H, IMG_C))
		image_label = np.ndarray((BATCH_SIZE, CLASSES))
		n = 0
		for i in range(train_df.shape[0]):
			dir_name = str(train_df.iloc[i]['label']) + "/"
			image_batch[n] = tf.keras.preprocessing.image.load_img(image_data_path + dir_name + train_df.iloc[i]['imagename'])
			image_label[n] = tf.one_hot(train_df.iloc[i]['label'], CLASSES)
			n = n + 1
			if(n == BATCH_SIZE):
				n = 0
				yield image_batch, image_label

# loading train_csv data
train_df = pd.read_csv(train_csv_path)
print("Shape of train_df: ", train_df.shape)

# loading val_csv data
val_df = pd.read_csv(val_csv_path)
print("Shape of val_df: ", val_df.shape)

# loading val data
val_images = np.ndarray((val_df.shape[0], IMG_W, IMG_H, IMG_C))
val_labels = np.ndarray((val_df.shape[0], CLASSES))
for i in range(val_df.shape[0]):
	dir_name = str(val_df.iloc[i]['label']) + "/"
	val_images[i] = tf.keras.preprocessing.image.load_img(image_data_path + dir_name + val_df.iloc[i]['imagename'])
	val_labels[i] = tf.one_hot(val_df.iloc[i]['label'], CLASSES)

# defining CNN architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_W, IMG_H, IMG_C)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
print(model.summary())

# compile model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# take sample data
train_df = train_df.iloc[:140, :]
val_df = val_df.iloc[:140, :]

# train model
steps = train_df.shape[0] // BATCH_SIZE
generator = data_generator(train_df)
model.fit_generator(generator, epochs=EPOCHS, steps_per_epoch=steps, validation_data=(val_images, val_labels), verbose=1)
model.save(target_path + "model_v1.h5")
