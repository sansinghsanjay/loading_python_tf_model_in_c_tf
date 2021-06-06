'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To plot classwise number of images in training set
'''

# packages
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# constants
DIR_NAME = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_FILES_PER_CLASS = 3500
VAL_FILES_PER_CLASS = 200

# paths
train_source_path = "/home/sansingh/Downloads/DATASET/mnist/trainingSet/"
test_source_path = "/home/sansingh/Downloads/DATASET/mnist/testSet/"
target_path = "/home/sansingh/github_repo/python-tf-keras-mnist/intermediate_outputs/"

# getting number of files in different classes of training data
train_image_count = list()
print("Number of files in train data: ")
for dir_name in DIR_NAME:
	files = os.listdir(train_source_path + str(dir_name) + "/")
	train_image_count.append(len(files))
	print("Class ", dir_name, " : ", len(files), " Files")
print()

# making plot
fig, ax = plt.subplots()
plt.title("Classwise number of images in training set")
plt.xlabel("Class Name")
plt.ylabel("Number of images")
plt.xticks(DIR_NAME)
plt.grid()
plt.bar(DIR_NAME, train_image_count, color="blue")
for index, value in enumerate(train_image_count):
	ax.text(index, value, str(value), color='black')
plt.savefig(target_path + "classwise_images_in_training_set.png")
print("Successfully saved plot for classwise number of images in training set")
