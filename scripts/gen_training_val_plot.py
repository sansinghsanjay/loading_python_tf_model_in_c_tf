'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To generate loss and accuracy plot for training and validation data
'''

# packages
import matplotlib.pyplot as plt
import pickle

# paths
model_history_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/trained_model/history_v1"
target_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/trained_model/"

# loading history - pickle file
history_ptr = open(model_history_path, 'rb')
history_data = pickle.load(history_ptr)
history_ptr.close()

# making loss plot
plt.title("Train & Val - Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
x_values = list(range(1, 6))
plt.xticks(x_values)
plt.scatter(x_values, history_data['loss'], color='blue')
plt.plot(x_values, history_data['loss'], color='blue')
plt.scatter(x_values, history_data['val_loss'], color='red')
plt.plot(x_values, history_data['val_loss'], color='red')
# making legends
color_legends = {'Train Loss': 'blue', 'Val Loss': 'red'}
legend_labels = list(color_legends.keys())
handles = [plt.Rectangle((0,0), 1, 1, color=color_legends[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
# saving loss plot
plt.savefig(target_path + "loss_plot.png")
plt.close()

# make accuracy plot
plt.title("Train & Val - Accuracy Plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
x_values = list(range(1, 6))
plt.xticks(x_values)
plt.scatter(x_values, history_data['accuracy'], color='blue')
plt.plot(x_values, history_data['accuracy'], color='blue')
plt.scatter(x_values, history_data['val_accuracy'], color='red')
plt.plot(x_values, history_data['val_accuracy'], color='red')
# making legends
color_legends = {'Train Accuracy': 'blue', 'Val Accuracy': 'red'}
legend_labels = list(color_legends.keys())
handles = [plt.Rectangle((0,0), 1, 1, color=color_legends[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
# saving loss plot
plt.savefig(target_path + "accuracy_plot.png")
plt.close()
