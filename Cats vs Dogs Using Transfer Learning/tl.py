import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input

i_width=128
i_height=128
i_shape=(i_width,i_height)
i_channels=3

base_model = VGG16(weights='imagenet', 
                      include_top=False, 
                      input_shape=(i_height, i_width, 3))

filenames=os.listdir("./data/train")
categories=[]
for f_name in filenames:
    category=f_name.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)
df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})

df["category"] = df["category"].replace({0:'cat',1:'dog'})

train_df,validate_df = train_test_split(df,test_size=0.20)

train_df=train_df.reset_index(drop=True)
validate_df=validate_df.reset_index(drop=True)

total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=20

train_datagen=ImageDataGenerator(rotation_range=30,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "./data/train/",
                                                 x_col='filename',
                                                 y_col='category',
                                                 target_size=i_shape,
                                                 class_mode='binary',
                                                 batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./data/train/", 
    x_col='filename',
    y_col='category',
    target_size=i_shape,
    class_mode='binary',
    batch_size=batch_size
)

base_model.trainable = False

x=base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

epochs=10

model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size
)

