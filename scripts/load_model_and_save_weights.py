'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To load model and save weights so that they can be loaded into C-Tensorflow
'''

# packages
import tensorflow as tf

# paths
saved_model_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/trained_model/model_v1.h5"
target_path = "/home/sansingh/github_repo/loading_python_tf_model_in_c_tf/trained_model/model_weights/"

# loading model
model = tf.keras.models.load_model(saved_model_path)

# model summary
print(model.summary())

# saving weights of model
model.save_weights(target_path + "model_weights")
print("Successfully saved model weights")
