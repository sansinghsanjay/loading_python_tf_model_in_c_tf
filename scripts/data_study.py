'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To study data - see number of images in each class, generating plot and saving it.
'''

# packages
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random
from itertools import repeat

# constants
DIR_NAME = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
TRAIN_FILES_PER_CLASS = 3500

# paths
train_source_path = "/home/sansingh/Downloads/DATASET/mnist/trainingSet/"
test_source_path = "/home/sansingh/Downloads/DATASET/mnist/testSet/"
target_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/intermediate_outputs/"

# getting number of files in different classes of training data
train_image_count = list()
print("Number of files in train data: ")
for dir_name in DIR_NAME:
	files = os.listdir(train_source_path + dir_name + "/")
	train_image_count.append(len(files))
	print("Class ", dir_name, " : ", len(files), " Files")
print()

# getting number of files in test data
test_image_count = len(os.listdir(test_source_path))
print("Number of files in test data: ", test_image_count)
print()

# taking first TRAIN_FILES_PER_CLASS for training and rest for validation
train_filenames = list()
val_filenames = list()
train_labels = list()
val_labels = list()
print("Segregating data for training and validation: ")
for dir_name in tqdm(DIR_NAME):
	files = os.listdir(train_source_path + dir_name + "/")
	random.shuffle(files)
	train_filenames.extend(files[:TRAIN_FILES_PER_CLASS])
	train_labels.extend(repeat(dir_name, TRAIN_FILES_PER_CLASS))
	val_filenames.extend(files[TRAIN_FILES_PER_CLASS:])
	val_labels.extend(repeat(dir_name, len(files[TRAIN_FILES_PER_CLASS:])))
print()

# making dataframe and saving data
train_df = pd.DataFrame(columns=['imagename', 'label'])
val_df = pd.DataFrame(columns=['imagename', 'label'])
train_df['imagename'] = train_filenames
train_df['label'] = train_labels
val_df['imagename'] = val_filenames
val_df['label'] = val_labels

# shuffling train and validation dataframes
train_df = train_df.sample(frac=1.0).reset_index(drop=True)
val_df = val_df.sample(frac=1.0).reset_index(drop=True)

# saving train and validation dataframes as csv files
train_df.to_csv(target_path + "train_data.csv", index=False)
val_df.to_csv(target_path + "val_data.csv", index=False)
print("Successfully saved train and validation data in csv files")
print()

# making plot of statistics of data
# x-axis labels
labels = ['T0', 'V0', 'T1', 'V1', 'T2', 'V2', 'T3', 'V3', 'T4', 'V4', 'T5', 'V5', 'T6', 'V6', 'T7', 'V7', 'T8', 'V8', 'T9', 'V9']
# y-axis values
values = [len(train_df[train_df['label'] == '0']), len(val_df[val_df['label'] == '0']), len(train_df[train_df['label'] == '1']), len(val_df[val_df['label'] == '1']), len(train_df[train_df['label'] == '2']), len(val_df[val_df['label'] == '2']), len(train_df[train_df['label'] == '3']), len(val_df[val_df['label'] == '3']), len(train_df[train_df['label'] == '4']), len(val_df[val_df['label'] == '4']), len(train_df[train_df['label'] == '5']), len(val_df[val_df['label'] == '5']), len(train_df[train_df['label'] == '6']), len(val_df[val_df['label'] == '6']), len(train_df[train_df['label'] == '7']), len(val_df[val_df['label'] == '7']), len(train_df[train_df['label'] == '8']), len(val_df[val_df['label'] == '8']), len(train_df[train_df['label'] == '9']), len(val_df[val_df['label'] == '9'])]
# taking subplot
fig, ax = plt.subplots()
plt.title("No. of images in train & validation data - classwise")
plt.xlabel("Class Name")
plt.ylabel("Number of images")
plt.grid()
bar_list = plt.bar(labels, values)
# coloring train and validation bars in different colors
for i in range(len(bar_list)):
	if(i % 2 == 0):
		bar_list[i].set_color('b')
	else:
		bar_list[i].set_color('r')
# making legends for colors
color_legends = {'Train': 'blue', 'Val': 'red'}
legend_labels = list(color_legends.keys())
handles = [plt.Rectangle((0,0), 1, 1, color=color_legends[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
# writing numerical value on respective bars
for index, value in enumerate(values):
	ax.text(index, value, str(value), color='black')
# saving plot
plt.savefig(target_path + "classwise_stats_train_val_test_data.png")
print("Successfully saved plot for stats of data")
