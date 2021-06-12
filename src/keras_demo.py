import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import tensorflow as tf
from utils import data_preprocess

from keras.callbacks import ModelCheckpoint 
from datetime import datetime

root_path = r'E:\projects\ruixinwei\2021rui\2021_ruixinwei_soundrecognition\data'
dst_path_tr = 'mfcc_train.txt'
dst_path_val = 'mfcc_val.txt'

data_processor = data_preprocess.data_preprocess(root_path)

# process training data
"""
在运行第一遍或者已经存在相关文件（.csv 与 .txt）之后，可以将前三行注释掉
"""
# filenames, timestamps, labels = data_processor.csvfile_resolution('train_labels.csv')
# data_processor.frame_resolution(filenames, timestamps, labels, sr=16000, span=160, dst_path='tr_piece.csv')
# filenames, timestamps, labels = data_processor.csvfile_resolution('tr_piece.csv')
# data_processor.feature_to_file(filenames, timestamps, labels, dst_path_tr)
train_labels, train_features = data_processor.feature_load(dst_path_tr)

# process validation data
"""
在运行第一遍或者已经存在相关文件（.csv 与 .txt）之后，可以将前三行注释掉
"""
# filenames, timestamps, labels = data_processor.csvfile_resolution('val_labels.csv')
# data_processor.frame_resolution(filenames, timestamps, labels, sr=16000, span=160, dst_path='val_piece.csv')
# filenames, timestamps, labels = data_processor.csvfile_resolution('val_piece.csv')
# data_processor.feature_to_file(filenames, timestamps, labels, dst_path_val, train=False)
val_labels, val_features = data_processor.feature_load(dst_path_val)

train_features = train_features.astype(np.float32)
val_features = val_features.astype(np.float32)
print(val_features.shape)

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
le = LabelEncoder()
val_labels = to_categorical(le.fit_transform(val_labels))
train_labels = to_categorical(le.fit_transform(train_labels))

num_rows = 40
num_columns = 1
num_channels = 1

# train_features = train_features.reshape(train_features.shape[0], num_rows, num_columns, num_channels)
# val_features = val_features.reshape(val_features.shape[0], num_rows, num_columns, num_channels)
train_features = train_features.reshape(train_features.shape[0], num_rows)
val_features = val_features.reshape(val_features.shape[0], num_rows)
input_shape = (40,)

num_labels = 4
filter_size = 2

# Construct model 
# model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=2, input_shape=(num_rows, num_channels), activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling1D())

# model.add(Dense(num_labels, activation='softmax'))

model = Sequential()
model.add(Dense(256, activation="relu", input_shape=input_shape))
model.add(Dense(128, activation="relu", input_shape=input_shape))
model.add(Dense(64, activation="relu", input_shape=input_shape))
model.add(Dense(32, activation="relu", input_shape=input_shape))
model.add(Dense(num_labels, activation='softmax'))

#################################

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(val_features, val_labels, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 36
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(train_features, train_labels, batch_size=num_batch_size, epochs=num_epochs, validation_data=(val_features, val_labels), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(train_features, train_labels, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(val_features, val_labels, verbose=0)
print("Testing Accuracy: ", score[1])