#MLP:Multi Layer Propagation
import numpy as np

class MLP:
    def __init__(self,num_inputs=3,num_hidden=[3,5],num_outputs=2):

        self.num_inputs=num_inputs
        self.num_hidden=num_hidden
        self.num_outputs=num_outputs

        layers= [self.num_inputs]+ self.num_hidden+[self.num_outputs]
        #input->layer1->layer2->output

        #initiate random weights
        self.weights=[]
        for i in range(len(layers)-1):
            w=np.random.rand(layers[i],layers[i+1]) #creating a 2-d array with rows=layers[i] and columns=layers[i+1]
            self.weights.append(w)


    def forward_propagate(self,inputs):

        activations=inputs

        for w in self.weights:
            #calculate net inputs for a layer
            net_inputs=np.dot(activations,w)
            
            
            #calculate the activations
            activations=self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))


if __name__=="__main__":

    #create an MLP
    mlp=MLP()

    #create some inputs
    inputs= np.random.rand(mlp.num_inputs)

    #perform forward propagation
    outputs=mlp.forward_propagate(inputs)

    #print results
    print(outputs )
              

            
        

        
