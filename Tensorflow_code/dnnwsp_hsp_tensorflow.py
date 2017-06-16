# -*- coding: utf-8 -*-

################################################# Import #################################################

# This import statement gives Python access to all of TensorFlow's classes, methods, and symbols. 
import tensorflow as tf
# NumPy is the fundamental package for scientific computing with Python.
import numpy as np
# Linear algebra module for calculating L1 and L2 norm  
from numpy import linalg as LA
# To plot the results
import matplotlib.pyplot as plt
# To check the directory when saving the results
import os.path
# The module for file input and output
import scipy.io as sio


################################################# Parameters #################################################

from customizationGUI \
        import mode, optimizer_algorithm, momtentum, nodes, n_epochs, batch_size,\
        beginAnneal, decay_rate, lr_init, lr_min, beta_lrates, L2_reg, max_beta, tg_hspset
    

################################################# Input data #################################################


datasets = sio.loadmat('lhrhadvs_sample_data.mat')

################ lhrhadvs_sample_data.mat ##################
# train_x  = 240 volumes x 74484 voxels  
# train_y  = 240 volumes x 1 [0:LH, 1:RH, 2:AD, 3:VS]
# test_x  = 120 volumes x 74484 voxels
# test_y  = 120 volumes x 1 [0:LH, 1:RH, 2:AD, 3:VS]
############################################################


train_x = datasets['train_x']
train_y = np.zeros((np.shape(datasets['train_y'])[0],np.max(datasets['train_y'])+1))
# transform into One-hot
for i in np.arange(np.shape(datasets['train_y'])[0]):
    train_y[i][datasets['train_y'][i][0]]=1 


test_x = datasets['test_x']
test_y = np.zeros((np.shape(datasets['test_y'])[0],np.max(datasets['test_y'])+1))
# transform into One-hot
for i in np.arange(np.shape(datasets['test_y'])[0]):
    test_y[i][datasets['test_y'][i][0]]=1 


################################################# Build Model #################################################


# 'node_index' to split placeholder, for an example, given hidden_nodes=[100, 100, 100], nodes_index=[0, 100, 200, 300]
nodes_index= [int(np.sum(nodes[1:i+1])) for i in np.arange(np.shape(nodes)[0]-1)]

# Make two placeholders to fill the values later when training or testing
X=tf.placeholder(tf.float32,[None,nodes[0]])
Y=tf.placeholder(tf.float32,[None,nodes[-1]])

# Create randomly initialized weight variables 
w_init=[tf.div(tf.random_normal([nodes[i],nodes[i+1]]), tf.sqrt(float(nodes[i])/2)) for i in np.arange(np.shape(nodes)[0]-1)]
w=[tf.Variable(w_init[i], dtype=tf.float32) for i in np.arange(np.shape(nodes)[0]-1)]
# Create randomly initialized bias variables 
b=[tf.Variable(tf.random_normal([nodes[i+1]]), dtype=tf.float32) for i in np.arange(np.shape(nodes)[0]-1)]

# Build MLP model 
hidden_layers=[0.0]*(np.shape(nodes)[0]-2)
for i in np.arange(np.shape(nodes)[0]-2):
    # Input layer
    if i==0:
        hidden_layers[i]=tf.add(tf.matmul(X,w[i]),b[i])
        hidden_layers[i]=tf.nn.tanh(hidden_layers[i])
    # The other layers    
    else:     
        hidden_layers[i]=tf.add(tf.matmul(hidden_layers[i-1],w[i]),b[i])
        hidden_layers[i]=tf.nn.tanh(hidden_layers[i])
# Output layer
output_layer=tf.add(tf.matmul(hidden_layers[-1],w[-1]),b[-1])

# Logistic regression layer
logRegression_layer=tf.nn.tanh(output_layer)
                    


############################################# Variables declaration #############################################


# Make placeholders for total beta array (make a long one to concatenate every beta vector) 
def init_beta():
    if mode=='layer':
        # The size is same with the number of layers
        Beta=tf.placeholder(tf.float32,[np.shape(nodes)[0]-2])
    elif mode=='node':
        # The size is same with the number of nodes
        Beta=tf.placeholder(tf.float32,[np.sum(nodes[1:-1])])

    return Beta


# Make L1 loss term for regularization
def init_L1():
    if mode=='layer':
        # Get L1 loss term by simply multiplying beta(scalar value) and L1 norm of weight for each layer
        L1=[Beta[i]*tf.reduce_sum(abs(w[i])) for i in np.arange(np.shape(nodes)[0]-2)]
    elif mode=='node':
        # Get L1 loss term by multiplying beta(vector values as many as nodes) and L1 norm of weight for each layer
        L1=[tf.reduce_sum(tf.matmul(abs(w[i]),tf.cast(tf.diag(Beta[nodes_index[i]:nodes_index[i+1]]),tf.float32))) for i in np.arange(np.shape(nodes)[0]-2)]
        
#    L1_total=tf.reduce_sum(L1)

    return L1


# Make L2 loss term for regularization
def init_L2():
    L2=[tf.reduce_sum(tf.square(w[i])) for i in np.arange(np.shape(nodes)[0]-1)] 
    
#    L2_total=L2_reg*tf.reduce_sum(L2) 
    
    return L2


       

# Define cost term (Cost = cross entropy + L1 term + L2 term )    
def init_cost():

    # A softmax regression : it adds up the evidence of our input being in certain classes, and converts that evidence into probabilities.
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logRegression_layer, labels=Y)) \
                                     + tf.reduce_sum(L1) + L2_reg*tf.reduce_sum(L2) 
    
    return cost


# TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
def init_optimizer(Lr):
    if optimizer_algorithm=='GradientDescent':
        optimizer=tf.train.GradientDescentOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Adagrad':
        optimizer=tf.train.AdagradOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Adam':
        optimizer=tf.train.AdamOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Momentum':
        optimizer=tf.train.MomentumOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='RMSProp':
        optimizer=tf.train.RMSPropOptimizer(Lr).minimize(cost) 

    return optimizer



# initialization   
def init_otherVariables():           
    if mode=='layer': 
        beta_val = np.zeros(np.shape(nodes)[0]-2)
        beta = np.zeros(np.shape(nodes)[0]-2)
        hsp_val = np.zeros(np.shape(nodes)[0]-2)            
        plot_beta = np.zeros(np.shape(nodes)[0]-2)
        plot_hsp = np.zeros(np.shape(nodes)[0]-2)
                   
    elif mode=='node':                       
        beta_val = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]  
        beta = np.zeros(np.sum(nodes[1:-1]))
        hsp_val = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]            
        plot_beta = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]
        plot_hsp = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]
    
    # make arrays to store and plot results
    plot_lr=np.zeros(1)
    plot_cost=np.zeros(1)
    plot_train_err=np.zeros(1)
    plot_test_err=np.zeros(1)
    
    # initialize learning rate
    lr = lr_init 
    
    
    return lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err
        




# Make a placeholder to be able to update learning rate (Learning rate decaying) 
Lr=tf.placeholder(tf.float32)


Beta = init_beta()
L1 = init_L1()
L2 = init_L2()
cost = init_cost()

optimizer=init_optimizer(Lr)

 
correct_prediction=tf.equal(tf.argmax(output_layer,1),tf.argmax(Y,1))  
# calculate an average error depending on how frequent it classified correctly   
error=1-tf.reduce_mean(tf.cast(correct_prediction,tf.float32))      


lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = init_otherVariables()





if mode=='layer':
    # Weight sparsity control with Hoyer's sparsness (Layer wise)  
    def Hoyers_sparsity_control(w_,b,max_b,tg):
        
        # Get value of weight
        W=sess.run(w_)
        [nodes,dim]=W.shape  
        
        # vectorize weight matrix 
        Wvec=W.flatten()     
        sqrt_nsamps=np.sqrt(Wvec.shape[0])
        
        # Calculate L1 and L2 norm     
        L1norm=LA.norm(Wvec,1)
        L2norm=LA.norm(Wvec,2)
        
        # Calculate hoyer's sparsness
        h=(sqrt_nsamps-(L1norm/L2norm))/(sqrt_nsamps-1)
        
        # Update beta
        b-=beta_lrates*np.sign(h-tg)
        
        # Trim value
        b=0.0 if b<0.0 else b
        b=max_b if b>max_b else b
                         
        return [h,b]
    
    
elif mode=='node':   
    # Weight sparsity control with Hoyer's sparsness (Node wise)
    def Hoyers_sparsity_control(w_,b_vec,max_b,tg):
    
        # Get value of weight
        W=sess.run(w_)
        [nodes,dim]=W.shape
        sqrt_nsamps=np.sqrt(nodes)
        
        # Calculate L1 and L2 norm 
        L1norm=LA.norm(W,1,axis=0)
        L2norm=LA.norm(W,2,axis=0)
     
        # Calculate hoyer's sparsness
        h_vec = np.zeros((1,dim))
        h_vec=(sqrt_nsamps-(L1norm/L2norm))/(sqrt_nsamps-1)
        
        tg_vec = np.ones(dim)*tg
        # Update beta       
        b_vec-=beta_lrates*np.sign(h_vec-tg_vec)
        
        # Trim value
        b_vec[b_vec<0.0]=0.0
        b_vec[b_vec>max_b]=max_b
        
               
        return [h_vec,b_vec]
    



############################################# Condition check #############################################

condition=False

print()

if np.size(nodes) <3:
    print("Error : The number of total layers is not enough.")
elif (np.size(nodes)-2) != np.size(max_beta):
    print("Error : The number of hidden layers and max beta values don't match. ")
elif (np.size(nodes)-2) != np.size(tg_hspset):
    print("Error : The number of hidden layers and target sparsity values don't match.")
elif np.size(train_x,axis=0) != np.size(train_y,axis=0):
    print("Error : The sizes of input train datasets and output train datasets don't match. ")  
elif np.size(test_x,axis=0) != np.size(test_y,axis=0):
    print("Error : The sizes of input test datasets and output test datasets don't match. ")     
elif (np.any(np.array(tg_hspset)<0)) | (np.any(np.array(tg_hspset)>1)):  
    print("Error : The values of target sparsities are inappropriate.")
else:
    condition=True


################################################ Learning ################################################


if condition==True:
    

    # variables are not initialized when you call tf.Variable
    # To initialize all the variables in a TensorFlow program, you must explicitly call a special operation         
    init = tf.global_variables_initializer()              
     

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:           
    
        # run tensorflow variable initialization
        sess.run(init)
        

        # Start training 
        for epoch in np.arange(n_epochs):            
                   
            # Shuffle training data at the begining of each epoch           
            total_sample = np.size(train_x, axis=0)
            sample_ids = np.arange(total_sample)
            np.random.shuffle(sample_ids) 
            
            train_x_shuff = np.array([ train_x[i] for i in sample_ids])
            train_y_shuff = np.array([ train_y[i] for i in sample_ids])
            
            
            # Begin Annealing
            if beginAnneal == 0:
                lr = lr * 1.0
            elif epoch+1 > beginAnneal:
                lr = max( lr_min, (-decay_rate*(epoch+1) + (1+decay_rate*beginAnneal)) * lr )  
            
        
            
            
            # Calculate how many mini-batch iterations we need
            total_batch = int(np.shape(train_x)[0]/batch_size) 
            
            cost_epoch=0.0
            
            # minibatch based training  
            for batch in np.arange(total_batch):
                batch_x = train_x_shuff[batch*batch_size:(batch+1)*batch_size]
                batch_y = train_y_shuff[batch*batch_size:(batch+1)*batch_size]
                
                # Get cost and optimize the model
                cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Y:batch_y, Beta:beta})

                cost_epoch+=cost_batch/total_batch      
        
        
                # weight sparsity control    
                if mode=='layer':                   
                    for i in np.arange(np.shape(nodes)[0]-2):
                        [hsp_val[i], beta_val[i]] = Hoyers_sparsity_control(w[i], beta_val[i], max_beta[i], tg_hspset[i])   
                    beta=beta_val                      

                elif mode=='node':                             
                    for i in np.arange(np.shape(nodes)[0]-2):
                        [hsp_val[i], beta_val[i]] = Hoyers_sparsity_control(w[i], beta_val[i], max_beta[i], tg_hspset[i])   
                    # flatten beta_val (shape (3, 100) -> (300,))
                    beta=[item for sublist in beta_val for item in sublist]
               
            # get train error
            train_err_epoch=sess.run(error,{X:train_x_shuff, Y:train_y_shuff})
            plot_train_err=np.hstack([plot_train_err,[train_err_epoch]])
            
            # get test error
            test_err_epoch=sess.run(error,{X:test_x, Y:test_y})
            plot_test_err=np.hstack([plot_test_err,[test_err_epoch]])
            
            
            
            # Save the results to plot at the end
            plot_lr=np.hstack([plot_lr,[lr]])
            plot_cost=np.hstack([plot_cost,[cost_epoch]])
            
            if mode=='layer':
                plot_hsp=[np.vstack([plot_hsp[i],[hsp_val[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                plot_beta=[np.vstack([plot_beta[i],[beta[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                
            elif mode=='node':
                plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp_val[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]
                plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta_val[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]

            
            # Print cost and errors after every training epoch       
            print("< Epoch", "{:02d}".format(epoch+1),"> Cost :", "{:.3f}".format(cost_epoch)\
                                            ,"/ Train err :", "{:.3f}".format(train_err_epoch),"/ Test err :","{:.3f}".format(test_err_epoch)) 
            print("             beta :",np.mean(np.mean(plot_beta,axis=1),axis=1))
            print("             hsp :",np.mean(np.mean(plot_hsp,axis=1),axis=1))  

        # Print final accuracy on test set
        print("")
        print("* Test accuracy :", "{:.3f}".format(1-sess.run(error,{X:test_x, Y:test_y})))
            
else:
    # Don't run the session but print 'failed' if any condition is not met
    print("Failed!")  
     
    
    
################################################ Plot & save results ################################################



if condition==True:
    
    
    # make a new 'results' directory in the current directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'results')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory) 
      
    # Plot the change of learning rate
    plt.figure() 
    plt.title("Learning rate plot",fontsize=16)
    plot_lr=plot_lr[1:]
    plt.ylim(0.0, lr_init*1.2)
    plt.plot(plot_lr)
    plt.savefig(final_directory+'/learning_rate.png')
    plt.show(block=False)
    
    
    # Plot the change of cost
    plt.figure() 
    plt.title("Cost plot",fontsize=16)
    plot_cost=plot_cost[1:]
    plt.plot(plot_cost)
    plt.savefig(final_directory+'/cost.png')
    plt.show(block=False)   
    
 
  
    # Plot train & test error
    plt.figure() 
    plt.title("Train & Test error plot",fontsize=16)
    plot_train_err=plot_train_err[1:]
    plt.plot(plot_train_err)
    plt.hold
    plot_test_err=plot_test_err[1:]
    plt.plot(plot_test_err)
    plt.ylim(0.0, 1.0)
    plt.legend(['Train error', 'Test error'],loc='upper right')
    plt.savefig(final_directory+'/error.png')
    plt.show(block=False) 
    


 
    # Plot the change of beta value
    print("")       
    plt.figure() 
    for i in np.arange(np.shape(nodes)[0]-2):
        print("")
        plt.title("Beta plot \n Hidden layer %d"%(i+1),fontsize=16)
        plot_beta[i]=plot_beta[i][1:]
        plt.plot(plot_beta[i])
        plt.ylim(0.0, np.max(max_beta)*1.2)
        plt.savefig(final_directory+'/beta%d.png'%(i+1))
        plt.show(block=False)
    
    
    # Plot the change of Hoyer's sparsity
    print("")      
    plt.figure()       
    for i in np.arange(np.shape(nodes)[0]-2):
        print("")
        plt.title("Hoyer's sparsity plot \n Hidden layer %d"%(i+1),fontsize=16)
        plot_hsp[i]=plot_hsp[i][1:]
        plt.plot(plot_hsp[i])
        plt.ylim(0.0, 1.0)
        plt.savefig(final_directory+'/hsp%d.png'%(i+1))
        plt.show(block=False)

        
    # save results as .mat file
    sio.savemat(final_directory+"/result_learningrate.mat", mdict={'lr': plot_lr})
    sio.savemat(final_directory+"/result_cost.mat", mdict={'cost': plot_cost})
    sio.savemat(final_directory+"/result_train_err.mat", mdict={'trainErr': plot_train_err})
    sio.savemat(final_directory+"/result_test_err.mat", mdict={'testErr': plot_test_err})
    sio.savemat(final_directory+"/result_beta.mat", mdict={'beta': plot_beta})
    sio.savemat(final_directory+"/result_hsp.mat", mdict={'hsp': plot_hsp})

else:
    None 
  
