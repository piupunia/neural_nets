import numpy as np

#structure of the neural network i.e the number of neurons in each layer (Bias neurons not included)

input_layer_size=2
hidden_layer_size=2
output_layer_size=1
learn_rate=1

X = np.array(([0, 0], [1, 1], [0, 1], [1, 0]), dtype=float) #(4*2) possible input for XOR
Y = np.array(([0],[0],[1],[1]), dtype=float)                #(4*1) correct ans for each input

#randomly initialized the weights of the connections from one layer to another (including the bias neuron in each layer)

Theta1 = np.random.rand(input_layer_size+1, hidden_layer_size)  #(3*2)
Theta2 = np.random.rand(hidden_layer_size+1, output_layer_size) #(3*1)


# sigmoid function f(z)=1/(1+e^(-z))
def sigmoid(z):
    return 1/(1+np.exp(-z))


# sigmoid gradient or sigmoid prime is the derivative of the sigmoid function
def sigmoid_grad(z):
    g=sigmoid(z)
    return g*(1-g)              #element wise multiplication


#forward propagation
def forward_prop(X,Theta1,Theta2):
    a1=np.c_[np.ones(X.shape[0]),X]   #(4*3)=((4*1)+(4*2))  this is the activation matrix for neurons of the first layer and the bias unit(X=X+(column of 1))

    z2=np.dot(a1,Theta1)     #(4*2)=(4*3)*(3*2) this is the input given to the second layer i.e input for first layer times the weights of corresponding networks

    a2=sigmoid(z2)   #(4*2)=(4*2) this is the activation for the neurons of second layer

    a2=np.c_[np.ones(a2.shape[0]),a2] #(4*3)=((4*1)+(4*2)) adding the bias unit

    z3=np.dot(a2,Theta2)   #(4*1)=((4*3)*(3*1)) this is the input given to the third  layer a2*Theta2

    a3=sigmoid(z3)         #(4*1)=(4*1) this is the activation for the third layer

    h=a3                   #(4*1)=(4*1) this is our output

    return a1,a2,z2,a3,z3,h



# backpropagation
for i in range(1000):
    a1,a2,z2,a3,z3,h=forward_prop(X,Theta1,Theta2)
    d3=h-Y     #(4*1)=(4*1)-(4*1) error for the last layer


    # the error contribution of the second layer i.e (d3*Theta2').*(g'(z2)) here element wise multiplication is represented by .*
    #Theata2 is from 1 and not 0 because we do not consider the bias unit to contribute to an error
    d2=np.dot(d3,Theta2[1:].T)*sigmoid_grad(z2) #(4*2)=(4*1)*(1*2)





    Tri1=np.dot(a1.T,d2)  #(3*2)=(3*4)*(4*2)
    Tri2=np.dot(a2.T,d3)  #(3*1)=(3*4)*(4*1)
    


    D1=Tri1/X.shape[0]    #(3*2)=(3*2)
    D2=Tri2/X.shape[0]    #(3*1)=(3*1)

    # applying gradient descent updating the values of Theta using a temporary variable
    t1=Theta1-learn_rate*D1  #(3*2)=(3*2)-(3*2)
    t2=Theta2-learn_rate*D2  #(3*1)=(3*1)-(3*1)


    Theta1=t1
    Theta2=t2



a1,a2,z2,a3,z3,h_f=forward_prop(X,Theta1,Theta2)
print(h_f)





