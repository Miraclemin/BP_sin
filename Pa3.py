# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
  

x = np.linspace(-np.pi,np.pi,140).reshape(140,-1)
y = np.sin(x)

lr = 0.02     #set learning rate


def mean_square_loss(y_pre,y_true):         #define loss 
    loss = np.power(y_pre - y_true, 2).mean()*0.5
    loss_grad = (y_pre-y_true)/y_pre.shape[0]
    return loss , loss_grad           # return loss and loss_grad
    
class ReLU():                     # ReLu layer
    def __init__(self):
        pass
    def forward(self,input):
        input[input<0]=0
        return input

        
    def backward(self,input,grad_output):
        input[input<=0]=0
        input[input>0]=1
        return input*grad_output

        
        

class FC():
    def __init__(self,input_dim,output_dim):    # initilize weights
        self.W = np.random.randn(input_dim,output_dim)*1e-2
        self.b = np.zeros((1,output_dim))
                       
    def forward(self,input):
        
        result = np.dot(input,self.W) - self.b
        return result
        
        
    
    def backward(self,input,grad_out):       # backpropagation , update weights in this step
        
        delt_W = (input*grad_out)
        delt_b = -input
        dimx,dimy=self.W.shape
        self.W = self.W.reshape(-1,1)
        for i in np.arange(x.shape[0]):
          self.W -= lr * delt_W[i].reshape(-1,1)
          self.b -= lr * delt_b[i]
        self.W=self.W.reshape(dimx,dimy)
        return self.W.reshape(1,-1)


#  bulid the network      
layer1 = FC(1,80)
ac1 = ReLU()
out_layer = FC(80,1)

# count steps and save loss history
loss = 1
step = 0
l= []
while loss >= 1e-4 and step < 15000: # training
    # forward     input x , through the network and get y_pre and loss_grad
    
        net=layer1.forward(x)
        out=ac1.forward(net)
        y_pre=out_layer.forward(out)

    #backward   # backpropagation , update weights through loss_grad
    
        loss,det2 = mean_square_loss(y_pre,y)
        w2=out_layer.backward(det2,out)
        det1 = ac1.backward(out,det2 * w2)
        layer1.backward(det1,x)
        step += 1
        l.append(loss)



    
    
# after training , plot the results
plt.plot(x,y,c='r',label='true_value')
plt.plot(x,y_pre,c='b',label='predict_value')
plt.legend()
plt.savefig('1.png')
plt.figure()
plt.plot(np.arange(0,len(l)), l )
plt.title('loss history')
plt.show()


