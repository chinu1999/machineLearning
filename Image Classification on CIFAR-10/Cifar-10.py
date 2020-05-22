from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

(train_x, train_y),(test_x, test_y)=cifar10.load_data()
    
train_x=train_x.astype('float32')
test_x=test_x.astype('float32')
 
train_x=train_x/255.0
test_x=test_x/255.0

train_y=np_utils.to_categorical(train_y)
test_y=np_utils.to_categorical(test_y)
 
num_classes=test_y.shape[1]

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy',
  optimizer=sgd,
  metrics=['accuracy'])

model.fit(train_x,train_y,
    validation_data=(test_x,test_y),
    epochs=20,batch_size=32)

_,acc=model.evaluate(test_x,test_y)
print(acc*100)

# model.save("model1_cifar_10epoch.h5")