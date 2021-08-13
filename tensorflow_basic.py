#tensorflow:deep learning library
#build model
#compile model
#train model
#evaluate model
#make predictions

import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_dataset(num_samples,test_size):
    x=np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)]) #num_samples:no.of samples in data set 
    y=np.array([[i[0] + i[1]] for i in x])

    #usually dataset is divided into:train set and test set
    #train set: for training the model
    #test set: testing the predictions made by model
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size) #training set is 30% of the entire data set
    return x_train,x_test,y_train,y_test

if __name__=="__main__":
    x_train,x_test,y_train,y_test=generate_dataset(5000,0.3) #in total 10 samples, 2 samples should belong to train set and 8 to test set

    #build model:2 neurons=>5 neurons->1 neuron
    model=tf.keras.Sequential([
            tf.keras.layers.Dense(5,input_dim=2,activation="sigmoid"), #Dense connects neurons from previous layer to current layer, this line is for the hidden layer with 5 neurons
            tf.keras.layers.Dense(1,activation="sigmoid")#output layer with 1 neuron
            
    ])

    #compile the model
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1)#gradient
    model.compile(optimizer=optimizer,loss="MSE") #loss:error function, min sqaure error b/w output and target, optimizer:gradient function

    #train the model
    model.fit(x_train,y_train,epochs=100)

    #evaluate the model
    print("Model evaluation:")
    model.evaluate(x_test,y_test,verbose=1) #evaluating model on test set

    #make predictions
    data=np.array([[0.1,0.2],[0.2,0.2]])
    predictions=model.predict(data)

    print("\nPredictions:")
    for d,p in zip(data,predictions):
        print("\n {} + {}={}".format(d[0],d[1],p[0]))

    
   

