#MLP:Multi Layer Propagation
#Backward propagation
#1)store all activations and derivatives
#2)implement back prop
#3)implement gradient descent
#4)implement training data set
import numpy as np
from random import random

class MLP:
    def __init__(self,num_inputs,num_hidden,num_outputs):

        self.num_inputs=num_inputs
        self.num_hidden=num_hidden
        self.num_outputs=num_outputs

        layers= [self.num_inputs]+ self.num_hidden+[self.num_outputs]
        #input->layer1->layer2->output

        #initiate random weights
        weights=[]
        for i in range(len(layers)-1):
            w=np.random.rand(layers[i],layers[i+1]) #creating a 2-d array with rows=layers[i] and columns=layers[i+1]
            weights.append(w)

        self.weights=weights

        activations=[]
        for i in range(len(layers)):
            a=np.zeros(layers[i])
            activations.append(a)

        self.activations=activations

        derivatives=[]
        for i in range(len(layers)-1):
            d=np.zeros((layers[i],layers[i+1]))
            derivatives.append(d)

        self.derivatives=derivatives

    def back_propagate(self,error,verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations=self.activations[i+1]
            delta=error*self._sigmoid_derivative(activations)
            delta_reshape=delta.reshape(delta.shape[0],-1).T

            
            curr_activations=self.activations[i]

            act_reshaped=curr_activations.reshape(curr_activations.shape[0],-1)

            self.derivatives[i]=np.dot(act_reshaped,delta_reshape)

            error=np.dot(delta,self.weights[i].T)

            if verbose:
                print(self.derivatives[i])

        return error

            


    def _sigmoid_derivative(self,x):
        return x*(1.0-x)
        
        


    def forward_propagate(self,inputs):

        activations=inputs

        self.activations[0]=inputs

        for i,w in enumerate(self.weights):
            #calculate net inputs for a layer
            net_inputs=np.dot(activations,w)
            
            
            #calculate the activations
            activations=self._sigmoid(net_inputs)
            self.activations[i+1]=activations

        return activations

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            weights=self.weights[i]
            
            
            derivatives=self.derivatives[i]

            
            weights+=derivatives * learning_rate

    def train(self,inputs,targets,epochs,learning_rate):
        #epochs:no.of times the data set is fed to the network, so that it can make better predictions

        for i in range(epochs):
            sum_error=0
            for inputz,target in zip(inputs,targets):
                #forward propagation
                output=self.forward_propagate(inputz)

                error=target-output

                #back_prop
                self.back_propagate(error)

                #apply gradient descent
                self.gradient_descent(learning_rate)

                #report error for each iteration
                sum_error+=self._mse(target,output) #min squared error

            print("error at {} epoch:{}".format(i,sum_error/len(inputs)))
            

    def _mse(self,target,output):
        return np.average((target-output)**2)


if __name__=="__main__":

    #create a dataset to train the model to peform sum operation on input
    inputs=np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    #array([[0.1,0.2],[0.3,0.4]])
    
    targets=np.array([[i[0] + i[1]] for i in inputs]) #array([[0.3],[0.7]]), 0.1+0.2=0.3 and 0.3+0.4=0.7

    #create an MLP
    mlp=MLP(2,[5],1)

    #train our mlp
    mlp.train(inputs,targets,50,0.1)

    #create dummy data
    inputz=np.array([0.3,0.1])
    target=np.array([0.4])

    output=mlp.forward_propagate(inputz)
    print("0.3+0.1={}".format(output))
    
    
              

            
        

        
