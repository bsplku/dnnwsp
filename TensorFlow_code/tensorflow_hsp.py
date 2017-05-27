# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import scipy.io
import os.path

################################################# Customization part #################################################
"""
autoencoder or not
"""
autoencoder=False

"""
Select the sparsity control mode
'layer' for layer wise sparsity control
'node' for node wise sparsity control
"""
mode = 'node'


"""
Select optimizer
'Grad' for GradientDescentOptimizer
'Ada' for AdagradOptimizer
'Moment' for MomentumOptimizer
'Adam' for AdamOptimizer
'RMSP' for RMSPropOptimizer
"""
optimizer_algorithm='Grad'

"""
Load your own data here
"""
dataset = scipy.io.loadmat('/home/hailey/01_study/prni2017_samples/lhrhadvs_sample_data2.mat')


""" 
Set the number of nodes for input, output and each hidden layer here
"""
nodes=[74484,100,100,100,4]
#nodes=[74484,500,74484]

"""
Set learning parameters
"""
# Set total epoch
total_epoch=600
# Set mini batch size
batch_size=100
# Let anealing to begin after 5th epoch
beginAnneal=75
# Set initial learning rate and minimum                     
lr_init = 1e-3    
min_lr = 1e-4

# Set learning rate of beta for weight sparsity control
lr_beta = 0.02
# Set L2 parameter for L2 regularization
L2_param= 1e-5


"""
Set maximum beta value of each hidden layer (usually 0.01~0.2) 
and set target sparsness value (0:dense~1:sparse)
"""
max_beta = [0.05, 0.8, 0.8]
tg_hsp = [0.7, 0.65, 0.65]

#max_beta = [0.02]
#tg_hsp = [0.5]

################################################# Input data part #################################################

# Split the dataset into traning input
train_input = dataset['train_x']
# Split the dataset into test input
test_input = dataset['test_x']


# Split the dataset into traning output 
train_output = np.zeros((np.shape(dataset['train_y'])[0],np.max(dataset['train_y'])+1))
# trainsform classes into One-hot
for i in np.arange(np.shape(dataset['train_y'])[0]):
    train_output[i][dataset['train_y'][i][0]]=1 
dataset['train_y']

# Split the dataset into test output
test_output = np.zeros((np.shape(dataset['test_y'])[0],np.max(dataset['test_y'])+1))
# trainsform classes into One-hot
for i in np.arange(np.shape(dataset['test_y'])[0]):
    test_output[i][dataset['test_y'][i][0]]=1 




################################################# Structure part #################################################



# We need 'node_index' for split placeholder (hidden_nodes=[100, 100, 100] -> nodes_index=[0, 100, 200, 300])
nodes_index= [int(np.sum(nodes[1:i+1])) for i in np.arange(np.shape(nodes)[0]-1)]

# Make placeholders to make our model in advance, then fill the values later when training or testing
X=tf.placeholder(tf.float32,[None,nodes[0]])
Y=tf.placeholder(tf.float32,[None,nodes[-1]])

# Make weight variables which are randomly initialized
w_init=[tf.div(tf.random_normal([nodes[i],nodes[i+1]]), tf.sqrt(float(nodes[i])/2)) for i in np.arange(np.shape(nodes)[0]-1)]
w=[tf.Variable(w_init[i], dtype=tf.float32) for i in np.arange(np.shape(nodes)[0]-1)]
# Make bias variables which are randomly initialized
b=[tf.Variable(tf.random_normal([nodes[i+1]])) for i in np.arange(np.shape(nodes)[0]-1)]

# Finally build our DNN model 
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

output_layer=tf.add(tf.matmul(hidden_layers[-1],w[-1]),b[-1])
logRegression_layer=tf.nn.tanh(output_layer)


                                 



############################################# Learning part #############################################



# Make placeholders for total beta vectors (make a long one to concatenate every beta vector) 
def build_betavec():
    if mode=='layer':
        Beta_vec=tf.placeholder(tf.float32,[np.shape(nodes)[0]-2])
    elif mode=='node':
        Beta_vec=tf.placeholder(tf.float32,[np.sum(nodes[1:-1])])

    return Beta_vec


# Make L1 loss term and L2 loss term for regularisation
def build_L1loss():
    if mode=='layer':
        L1_loss=[Beta_vec[i]*tf.reduce_sum(abs(w[i])) for i in np.arange(np.shape(nodes)[0]-2)]
    elif mode=='node':
        L1_loss=[tf.reduce_sum(tf.matmul(abs(w[i]),tf.cast(tf.diag(Beta_vec[nodes_index[i]:nodes_index[i+1]]),tf.float32))) for i in np.arange(np.shape(nodes)[0]-2)]

    return L1_loss

       

# Define cost term with cross entropy and L1 and L2 tetm     
def build_cost():
    if autoencoder:
        cost=tf.reduce_mean(tf.pow(X - output_layer, 2)) + tf.reduce_sum(L1_loss) + L2_param*tf.reduce_sum(L2_loss)
    else:
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logRegression_layer, labels=Y)) \
                                         + tf.reduce_sum(L1_loss) + L2_param*tf.reduce_sum(L2_loss)                                         
    return cost


# Define optimizer
def build_optimizer(Lr):
    if optimizer_algorithm=='Grad':
        optimizer=tf.train.GradientDescentOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Ada':
        optimizer=tf.train.AdagradOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Adam':
        optimizer=tf.train.AdamOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='Moment':
        optimizer=tf.train.MomentumOptimizer(Lr).minimize(cost) 
    elif optimizer_algorithm=='RMSP':
        optimizer=tf.train.RMSPropOptimizer(Lr).minimize(cost) 

    return optimizer



if mode=='layer':
    # Weight sparsity control with Hoyer's sparsness (Layer wise)  
    def Hoyers_sparsity_control(w_,b,max_b,tg):
        
        # Get value of weight
        W=sess.run(w_)
        [nodes,dim]=W.shape  
        num_elements=nodes*dim
 
        Wvec=W.flatten()
        
        # Calculate L1 and L2 norm     
        L1=LA.norm(Wvec,1)
        L2=LA.norm(Wvec,2)
        
        # Calculate hoyer's sparsness
        h=(np.sqrt(num_elements)-(L1/L2))/(np.sqrt(num_elements)-1)
        
        # Update beta
        b-=lr_beta*np.sign(h-tg)
        
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
        
        # Calculate L1 and L2 norm 
        L1=LA.norm(W,1,axis=0)
        L2=LA.norm(W,2,axis=0)
        
        h_vec = np.zeros((1,dim))
        tg_vec = np.ones(dim)*tg
        
        # Calculate hoyer's sparsness
        h_vec=(np.sqrt(nodes)-(L1/L2))/(np.sqrt(nodes)-1)
        
        # Update beta
        b_vec-=lr_beta*np.sign(h_vec-tg_vec)
        
        # Trim value
        b_vec[b_vec<0.0]=0.0
        b_vec[b_vec>max_b]=max_b
        
               
        return [h_vec,b_vec]
    





lr = lr_init

Beta_vec = build_betavec()

L1_loss = build_L1loss()
L2_loss = [tf.reduce_sum(tf.square(w[i])) for i in np.arange(np.shape(nodes)[0]-1)] 

cost = build_cost()


# Make learning rate as placeholder to update learning rate every iterarion 
Lr=tf.placeholder(tf.float32)
optimizer=build_optimizer(Lr)
  

if not autoencoder:
    correct_prediction=tf.equal(tf.argmax(output_layer,1),tf.argmax(Y,1))  
        
    # calculate mean accuracy depending on the frequency it predicts correctly
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))      








############################################# Condition check part #############################################


condition=False

print()

if np.shape(nodes)[0] <3:
    print("Error : Not enough hidden layer number.")
elif (autoencoder==False) & (np.shape(train_input)[0] != np.shape(train_output)[0]):
    print("Error : The sizes of input train dataset and output train dataset don't match. ")  
elif (autoencoder==False) & (np.shape(test_input)[0] != np.shape(test_output)[0]):
    print("Error : The sizes of input test dataset and output test dataset don't match. ")     
elif not ((mode=='layer') | (mode=='node')):
    print("Error : Select a valid mode. ") 
elif (np.any(np.array(tg_hsp)<0)) | (np.any(np.array(tg_hsp)>1)):  
    print("Error : Please set the target sparsities appropriately.")
elif autoencoder!=True & autoencoder!=False:
    print("Error : Please set the autoencoder mode appropriately.")
else:
    condition=True


################################################ Training & test part ################################################


if condition==True:
    
    # make initializer        
    init = tf.global_variables_initializer()              

    
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        
        # run tensorflow variable initialization
        sess.run(init)
    
        # initialization    
        def initialization():           
            if mode=='layer': 
                beta=np.zeros(np.shape(nodes)[0]-2)
                beta_vec = np.zeros(np.shape(nodes)[0]-2)
                hsp = np.zeros(np.shape(nodes)[0]-2)            
                plot_beta = np.zeros(np.shape(nodes)[0]-2)
                plot_hsp = np.zeros(np.shape(nodes)[0]-2)
                           
            elif mode=='node':                       
                beta = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]  
                beta_vec=np.zeros(np.sum(nodes[1:-1]))
                hsp = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]            
                plot_beta = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]
                plot_hsp = [np.zeros(nodes[i+1]) for i in np.arange(np.shape(nodes)[0]-2)]
                
            # make arrays to plot results
            plot_lr=np.zeros(1)
            plot_cost=np.zeros(1)
            
            return beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost
                
        beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost = initialization()
        
               

        
           
        # Calculate how many mini-batch iterations
        total_batch=int(np.shape(train_input)[0]/batch_size) 
               
        # train and get cost
        cost_avg=0.0
        for epoch in np.arange(total_epoch):            
            cost_epoch=0.0
            
            # Begin Annealing
            if beginAnneal == 0:
                lr = lr * 1.0
            elif epoch+1 > beginAnneal:
                lr = max( min_lr, (-0.01*(epoch+1) + (1+0.0001*beginAnneal)) * lr )        
            
            
            # Train at each mini batch    
            for batch in np.arange(total_batch):
                batch_x = train_input[batch*batch_size:(batch+1)*batch_size]
                if autoencoder==False:
                    batch_y = train_output[batch*batch_size:(batch+1)*batch_size]
                
                # Get cost and optimize the model
                if autoencoder:
                    cost_batch,_=sess.run([cost,optimizer],feed_dict={Lr:lr, X:batch_x, Beta_vec:beta_vec })
                else:                   
                    cost_batch,_=sess.run([cost,optimizer],feed_dict={Lr:lr, X:batch_x, Y:batch_y, Beta_vec:beta_vec })
                    
                cost_epoch+=cost_batch/total_batch      
        
                # make space to plot beta, sparsity level
                plot_lr=np.hstack([plot_lr,[lr]])
                plot_cost=np.hstack([plot_cost,[cost_batch]])

                
                # save footprint for plot
                if mode=='layer':
                    plot_hsp=[np.vstack([plot_hsp[i],[hsp[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[beta_vec[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    
                elif mode=='node':
                    plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]

                
        
                # run weight sparsity control function
                for i in np.arange(np.shape(nodes)[0]-2):
                    [hsp[i],beta[i]]=Hoyers_sparsity_control(w[i], beta[i], max_beta[i], tg_hsp[i])   
                    
                if mode=='layer':               
                    beta_vec=beta                      
                elif mode=='node':                              
                    beta_vec=[item for sublist in beta for item in sublist]
                    
                    
            # Print cost at each epoch        
            print("< Epoch", "{:02d}".format(epoch+1),"> Cost : ", "{:.4f}".format(cost_epoch))



        # Print final accuracy of test set
        if not autoencoder:
            print("Accuracy :",sess.run(accuracy,feed_dict={X:test_input, Y:test_output}))
            
else:
    # Don't run the sesstion but print 'failed' if any condition is unmet
    print("Failed!")  
     
    
    
################################################ Plot & save results part ################################################
if condition==True:
       
    # Plot the change of learning rate
    plt.title("Learning rate plot",fontsize=16)
    plot_lr=plot_lr[1:]
    plt.ylim(0.0, lr_init*1.2)
    plt.plot(plot_lr)
    plt.show()
    
    # Plot the change of cost
    plt.title("Cost plot",fontsize=16)
    plot_cost=plot_cost[1:]
    plt.plot(plot_cost)
    plt.show()      
    
    
    # Plot the change of beta value
    print("")       
    for i in np.arange(np.shape(nodes)[0]-2):
        print("")
        print("                  < Hidden layer",i+1,">")
        plt.title("Beta plot",fontsize=16)
        plot_beta[i]=plot_beta[i][1:]
        plt.plot(plot_beta[i])
        plt.ylim(0.0, np.max(max_beta)*1.2)
        plt.show()
    
    # Plot the change of Hoyer's sparsness
    print("")            
    for i in np.arange(np.shape(nodes)[0]-2):
        print("")
        print("                  < Hidden layer",i+1,">")
        plt.title("Hoyer's sparsness plot",fontsize=16)
        plot_hsp[i]=plot_hsp[i][1:]
        plt.plot(plot_hsp[i])
        plt.ylim(0.0, 1.0)
        plt.show()
    
    # make a new 'results' directory in the current directory
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'results')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory) 
        
    # save results as .mat file
    scipy.io.savemat("results/result_learningrate.mat", mdict={'lr': plot_lr})
    scipy.io.savemat("results/result_cost.mat", mdict={'cost': plot_cost})
    scipy.io.savemat("results/result_beta.mat", mdict={'beta': plot_beta})
    scipy.io.savemat("results/result_hsp.mat", mdict={'hsp': plot_hsp})

else:
    None 
  
    
    
    
#             if i<(np.shape(nodes)[0]-2):
#            print("                  < Hidden layer",i+1,">")
#        else:
#            print("                  < Output layer >")

