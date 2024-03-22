
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')




pathHead = 'D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/data_batch_'
datadicW = unpickle(f"{pathHead}1")

for i in range(2,6):
    # print(f"{pathHead}{i}")
    datadicti = unpickle(f"{pathHead}{i}")
    d = datadicW["data"]
    l = datadicW["labels"]
    datadicW["data"] = np.concatenate((d,datadicti["data"] ))
    datadicW["labels"] = np.concatenate((l,datadicti["labels"] ))
# SET XW YW as TRAIN SET
XW = datadicW["data"]
YW = datadicW["labels"]
XW = XW.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float64")
YW = np.array(YW)
yp = np.zeros([len(YW),10])
for i in range(0,10):
    idx = np.where(YW==i)
    
    yp[idx[0],i]=1

YW = yp

datadict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]
yp2 = np.zeros([len(Y),10])
for i in range(0,10):
    idx = np.where(Y==i)
    # print(yp[idx[0],i])
    
    yp2[idx[0],i]=1
Y = yp2
# print(yp2)
labeldict = unpickle('D:/GRAM/MasterProgramme/Tampere/DATA.ML.100-2020-2021-1 Introduction to Pattern Recognition and Machine Learning (Lectures)/week2/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float64")
Y=np.array(Y)
X = np.array(X)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
model = Sequential()
model.add(Dense(5, input_shape=(32,32,3), activation='sigmoid'))
## 2nd hidden layer - YOU MAY TEST THIS
model.add(keras.layers.Conv2D(32,(5,5), strides=2, activation="relu"))
model.add(keras.layers.Flatten())
# model.add(Dense(5, input_dim=3072, activation='sigmoid'))

# Output layer
model.add(Dense(10, activation='softmax'))
# Learning rate has huge effect 
keras.optimizers.SGD(lr=0.01)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])



model.fit(XW, YW, epochs=100)
#result of testbatch is bad,maybe because of overfit
test_loss, test_acc = model.evaluate(X,  Y, verbose=2)
print('\nTest accuracy:', test_acc)


