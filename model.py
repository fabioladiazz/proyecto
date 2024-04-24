import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


train_data_path = 'chest_xray/train'
filepaths =[]
labels = []
folds = os.listdir(train_data_path)

for fold in folds:
    f_path = os.path.join(train_data_path, fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path, file))
        labels.append(fold)

Fseries = pd.Series(filepaths , name='filepaths')
Lseries = pd.Series(labels, name='label')
df = pd.concat([Fseries, Lseries], axis=1)
test_data_path = 'chest_xray/test'

filepaths = []
labels = []
folds = os.listdir(test_data_path)

for fold in folds:
    f_path = os.path.join(test_data_path , fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path , file))
        labels.append(fold)

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='label')

test = pd.concat([Fseries, Lseries], axis=1)

valid_data_path = 'chest_xray/val'

filepaths= []
labels = []
folds = os.listdir(test_data_path)

for fold in folds:
    f_path = os.path.join(test_data_path , fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path , file))
        labels.append(fold)

Fseries = pd.Series(filepaths, name = 'filepaths')
Lseries = pd.Series(labels, name ='label')
valid = pd.concat([Fseries, Lseries], axis=1)

color_discrete_map = {
    df['label'].unique()[0]: '#344e41',
    df['label'].unique()[1]: '#A3B18A'
}

fig = px.histogram(
    data_frame=df,
    y=df['label'],
    color=df['label'].values,
    title='Number of images in each class of the train data',
    template='seaborn',
    color_discrete_map=color_discrete_map
)
fig.show()

fig = px.histogram(
    data_frame=test,
    y=df['label'],
    color=df['label'].values,
    title='Number of images in each class of the train data',
    template='seaborn',
    color_discrete_map=color_discrete_map
)
fig.show()

fig = px.histogram(
    data_frame=valid,
    y=df['label'],
    color=df['label'].values,
    title='Number of images in each class of the train data',
    template='seaborn',
    color_discrete_map=color_discrete_map
)
fig.show()

train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle= True, random_state= 42)
valid_df, test_df= train_test_split(dummy_df, train_size= 0.6, shuffle= True, random_state= 42)

img_size = (224 ,224)
batch_size = 16
img_shape= (img_size[0], img_size[1], 3)

def scalar(img):
    return img

tr_gen = ImageDataGenerator(preprocessing_function= scalar)
ts_gen = ImageDataGenerator(preprocessing_function= scalar)

train_gen = tr_gen.flow_from_dataframe(train_df , x_col='filepaths', y_col='label',
                                       target_size=img_size,  # Set target size to (224, 224)
                                       class_mode='categorical', color_mode='rgb',
                                       shuffle=True, batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df , x_col='filepaths', y_col='label',
                                       target_size=img_size,  # Set target size to (224, 224)
                                       class_mode='categorical', color_mode='rgb',
                                       shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df , x_col='filepaths', y_col='label',
                                      target_size=img_size,  # Set target size to (224, 224)
                                      class_mode='categorical', color_mode='rgb',
                                      shuffle=False, batch_size=batch_size)

gen_dict = train_gen.class_indices
classes = list(gen_dict.keys())
images, labels = next(train_gen)

plt.figure(figsize=(20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    image = images[i] / 255
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='#344E41', fontsize=12)
    plt.axis('off')
plt.show()

img_size = (224, 224)
img_shape = (img_size[0] , img_size[1] , 3)
num_class = len(classes)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = img_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides = 2, padding = 'same'),

    tf.keras.layers.Conv2D(64, (3, 3), strides = 1, padding = 'same',activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2) , strides = 2 , padding = 'same'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64, (3, 3), strides = 1, padding = 'same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides = 2, padding = 'same'),

    tf.keras.layers.Conv2D(128, (3, 3), strides = 1, padding = 'same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides = 2, padding = 'same'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(256, (3, 3), strides = 1, padding = 'same',activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides = 2, padding = 'same'),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(units = num_class, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x= train_gen , epochs = 15, verbose = 1, validation_data= valid_gen,validation_steps = None , shuffle = False)

plt.plot(history.history['accuracy'], label='accuracy', color='#A3B18A')
plt.plot(history.history['val_accuracy'], label='val_accuracy', color='#3A5A40')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds , axis = 1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())
cm = confusion_matrix(test_gen.classes, y_pred)

colors = ['#F6F6F3', '#3A5A40']
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=cmap_custom)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
for edge, spine in plt.gca().spines.items():
    spine.set_visible(True)
    spine.set_color('black')
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none'))
plt.grid(False)
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
print(classification_report(test_gen.classes, y_pred , target_names= classes ))

model.save('model.h5')
keras_model = load_model('model.h5')