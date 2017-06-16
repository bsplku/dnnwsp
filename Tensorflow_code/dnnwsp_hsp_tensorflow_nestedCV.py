#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:09:12 2017

@author: hailey
"""

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
mode = 'layer'


"""
Select optimizer
'GradientDescent'
'Adagrad'
'Adam'
Momentum'
'RMSProp'
"""
optimizer_algorithm='Adam'

"""
Load your own data here
"""
dataset = sio.loadmat('/home/hailey/01_study/prni2017_samples/lhrhadvs_sample_data2.mat')


""" 
Set the number of nodes for input, output and each hidden layer here
"""
nodes=[74484,100,100,100,4]

"""
Set learning parameters
"""

k_folds=5

# Set total epoch
total_epoch=100
# Set mini batch size
batch_size=50
# Let anealing to begin after **th epoch
beginAnneal=100
# anealing decay rate
decay_rate=1e-4
# Set initial learning rate and minimum                     
lr_init = 1e-3    
min_lr = 1e-4

# Set learning rate of beta for weight sparsity control
lr_beta = 0.02
# Set L2 parameter for L2 regularization
L2_param= 1e-5


"""
Set maximum beta value of each hidden layer (usually 0.01~0.5) 
and set target sparsness value (0:dense~1:sparse)
"""

max_beta = [0.05, 0.75, 0.7]
tg_hsp_cand=[ [0.5, 0.5, 0.5] , [0.3, 0.5, 0.3] , [0.5, 0.5, 0.3] , [0.5, 0.7, 0.7] , [0.7 , 0.5 , 0.7] , [0.7 , 0.7 , 0.7]]



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

# Split the dataset into test output
test_output = np.zeros((np.shape(dataset['test_y'])[0],np.max(dataset['test_y'])+1))
# trainsform classes into One-hot
for i in np.arange(np.shape(dataset['test_y'])[0]):
    test_output[i][dataset['test_y'][i][0]]=1 


total_images=np.vstack([train_input,test_input])
total_labels=np.vstack([train_output,test_output])

num_data=np.size(total_images,axis=0)        # total number of examples to use
num_1fold=int(num_data/k_folds)




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
#        L1_loss=[Beta_vec[i]*tf.reduce_mean(abs(w[i])) for i in np.arange(np.shape(nodes)[0]-2)]
    elif mode=='node':
        L1_loss=[tf.reduce_sum(tf.matmul(abs(w[i]),tf.cast(tf.diag(Beta_vec[nodes_index[i]:nodes_index[i+1]]),tf.float32))) for i in np.arange(np.shape(nodes)[0]-2)]
#        L1_loss=[tf.reduce_mean(tf.matmul(abs(w[i]),tf.cast(tf.diag(Beta_vec[nodes_index[i]:nodes_index[i+1]]),tf.float32))) for i in np.arange(np.shape(nodes)[0]-2)]


    return L1_loss

       

# Define cost term with cross entropy and L1 and L2 tetm     
def build_cost():
    if autoencoder==False:
        # A softmax regression has two steps: 
        # first we add up the evidence of our input being in certain classes, 
        # and then we convert that evidence into probabilities.
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logRegression_layer, labels=Y)) \
                                         + tf.reduce_sum(L1_loss) + L2_param*tf.reduce_sum(L2_loss)   
        
    else:              
        cost=tf.reduce_mean(tf.pow(X - output_layer, 2)) + tf.reduce_sum(L1_loss) + L2_param*tf.reduce_sum(L2_loss)
       
    return cost


# Define optimizer
# TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function.
def build_optimizer(Lr):
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
#L2_loss = [tf.reduce_mean(tf.square(w[i])) for i in np.arange(np.shape(nodes)[0]-1)] 


cost = build_cost()


# Make learning rate as placeholder to update learning rate every iterarion 
Lr=tf.placeholder(tf.float32)
optimizer=build_optimizer(Lr)
  

if autoencoder==False:
    correct_prediction=tf.equal(tf.argmax(output_layer,1),tf.argmax(Y,1))  
        
    # calculate mean error(accuracy) depending on the frequency it predicts correctly   
    error=1-tf.reduce_mean(tf.cast(correct_prediction,tf.float32))      




############################################# Condition check part #############################################


condition=False

print()

if np.size(nodes) <3:
    print("Error : The number of total layers is not enough.")
elif (np.size(nodes)-2) != np.size(max_beta):
    print("Error : The number of hidden layers and max beta values don't match. ")
elif (np.size(nodes)-2) != np.size(tg_hsp_cand,axis=1):
    print("Error : The number of hidden layers and target sparsity values don't match.")
elif (autoencoder==False) & (np.size(train_input,axis=0) != np.size(train_output,axis=0)):
    print("Error : The sizes of input train dataset and output train dataset don't match. ")  
elif (autoencoder==False) & (np.size(test_input,axis=0) != np.size(test_output,axis=0)):
    print("Error : The sizes of input test dataset and output test dataset don't match. ")     
elif (autoencoder!=False) & (autoencoder!=True):
    print("Error : Autoencoder mode is wrong.")
else:
    condition=True




################################################ Training & test part ################################################



if condition==True:
    
    # variables are not initialized when you call tf.Variable. 
    # To initialize all the variables in a TensorFlow program, you must explicitly call a special operation         
    init = tf.global_variables_initializer()              

    
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        
        # run tensorflow variable initialization
        sess.run(init)
        
        tg_hsp_selected_list=[]
        fianl_accuracy_list=[]
        
        
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
            plot_train_err=np.zeros(1)
            plot_test_err=np.zeros(1)
            
            return beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err
                
   
        
        for outer in range(k_folds):
        
        
            print("************************* outer fold (",outer+1, ") ****************************")        
            
            beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = initialization()
            
            # outer_train_list=[0,1,3,4], if this outer is 2
            outer_train_list=np.delete(np.arange(k_folds),outer)        
            
            
            error_list=list() 
            
            # for each candidate sets
            for tg_hsp in tg_hsp_cand:
                
                
     
                avg_err=0.0
                
                ######################################## Inner train ################################################
                for inner in outer_train_list:
                    
                    images_train=[]
                    labels_train=[]
                    images_valid=[]
                    labels_valid=[]
                    images_train_shuff=[]
                    labels_train_shuff=[]
                    
                    
                    inner_train_list=np.delete(outer_train_list,np.argwhere(outer_train_list==inner))  
                    
                    # make training data set
                    for j in inner_train_list:
                        if np.size(images_train)==0 & np.size(labels_train)==0:
                            images_train=total_images[j*num_1fold:(j+1)*num_1fold]         
                            labels_train=total_labels[j*num_1fold:(j+1)*num_1fold] 
                        else:
                            images_train=np.vstack([images_train, total_images[j*num_1fold:(j+1)*num_1fold]])
                            labels_train=np.vstack([labels_train, total_labels[j*num_1fold:(j+1)*num_1fold]])
                            
                    images_valid=total_images[inner*num_1fold:(inner+1)*num_1fold]
                    labels_valid=total_labels[inner*num_1fold:(inner+1)*num_1fold]        
                            
                    total_batch=int(np.shape(labels_train)[0]/batch_size)
                    # train and get cost    
                    for epoch in np.arange(total_epoch):            
                        cost_epoch=0.0
                        
                        # Begin Annealing
                        if beginAnneal == 0:
                            lr = lr * 1.0
                        elif epoch+1 > beginAnneal:
                            lr = max( min_lr, (-decay_rate*(epoch+1) + (1+decay_rate*beginAnneal)) * lr ) 
                            
                            
                        # shuffle data in every epoch           
                        total_sample = np.size(images_train, axis=0)
                        sample_ids = np.arange(total_sample)
                        np.random.shuffle(sample_ids) 
                                        
                        
                        for batch in np.arange(total_batch):                       
                            batch_x = images_train[sample_ids[batch*batch_size:(batch+1)*batch_size]]
                            batch_y = labels_train[sample_ids[batch*batch_size:(batch+1)*batch_size]]
                                         
                            
                            # Get cost and optimize the model
                            if autoencoder==False:
                                cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Y:batch_y, Beta_vec:beta_vec })
                                
                            else:                      
                                cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Beta_vec:beta_vec })
                                
                            cost_epoch+=cost_batch/total_batch    
                            
                            
                            # run weight sparsity control function
                            for i in np.arange(np.shape(nodes)[0]-2):
                                [hsp[i],beta[i]]=Hoyers_sparsity_control(w[i], beta[i], max_beta[i], tg_hsp[i])   
                                
                            if mode=='layer':               
                                beta_vec=beta                      
                            elif mode=='node':                              
                                beta_vec=[item for sublist in beta for item in sublist]
                        
                        if autoencoder==False:
                            images_train_shuff = np.array([ images_train[i] for i in sample_ids])
                            labels_train_shuff = np.array([ labels_train[i] for i in sample_ids])
                            
                            train_err_epoch=sess.run(error,{X:images_train_shuff, Y:labels_train_shuff})
                            plot_train_err=np.hstack([plot_train_err,[train_err_epoch]])
                            
                            test_err_epoch=sess.run(error,{X:images_valid, Y:labels_valid})
                            plot_test_err=np.hstack([plot_test_err,[test_err_epoch]])
            
            
                        
                        # save footprint for plot
                        if mode=='layer':
                            plot_hsp=[np.vstack([plot_hsp[i],[hsp[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                            plot_beta=[np.vstack([plot_beta[i],[beta_vec[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                            
                        elif mode=='node':
                            plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]
                            plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]
            
                        
       
                    ######################################## Inner validate ################################################   
                 
                    # Print final accuracy of test set
                    if autoencoder==False:       
                        # Plot train & test error
                        plt.title("Training & Test error",fontsize=16)
                        plot_train_err=plot_train_err[1:]
                        plt.plot(plot_train_err)
                        plt.hold
                        plot_test_err=plot_test_err[1:]
                        plt.plot(plot_test_err)
                        plt.ylim(0.0, 1.0)
                        plt.legend(['Training error', 'Test error'],loc='upper right')
                    #    plt.yscale('log')
                        plt.show() 
                        
                        print(">>>> (", np.argwhere([tg_hsp==i for i in tg_hsp_cand])[0][0]+1 ,") Target hsp",tg_hsp ,"<<<<")
                        print("outer fold :",outer+1,"/",k_folds," &  inner fold :",np.argwhere(outer_train_list==inner)[0][0]+1,"/",k_folds-1)
                        err=sess.run(error,{X:images_valid, Y:labels_valid})
                        print("Accuracy :","{:.4f}".format(1-err))
                        
                    print("Mean beta :",np.mean(np.mean(plot_beta,axis=1),axis=1))
                    print("Mean hsp :",np.mean(np.mean(plot_hsp,axis=1),axis=1))
                    print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
                        
                    avg_err+=err/np.size(outer_train_list)
                    
                    sess.run(init)
                    lr = lr_init
                    beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = initialization()
                    
                error_list.append(avg_err)
                print("")
                print("##############################################################################")
                print("# Avg validation error on this iteration for (", np.argwhere([tg_hsp==i for i in tg_hsp_cand])[0][0]+1 ,")",tg_hsp,"is" ,"{:.4f}".format(avg_err),"#")
                print("##############################################################################")
                print("")
                
            # select target sparsity set when validation error is min
            where_is_best=np.argmin(error_list)
            # store selected set in a list
            tg_hsp_selected_list.append(tg_hsp_cand[where_is_best])
            
        
            ######################################## Outer train ################################################              
            images_train=[]
            labels_train=[]
            images_valid=[]
            labels_valid=[]
            images_train_shuff=[]
            labels_train_shuff=[]
            
            # make training data set
            for j in outer_train_list:
                if np.size(images_train)==0 & np.size(labels_train)==0:
                    images_train=total_images[j*num_1fold:(j+1)*num_1fold]         
                    labels_train=total_labels[j*num_1fold:(j+1)*num_1fold] 
                else:
                    images_train=np.vstack([images_train, total_images[j*num_1fold:(j+1)*num_1fold]])
                    labels_train=np.vstack([labels_train, total_labels[j*num_1fold:(j+1)*num_1fold]])
                    
            images_test=total_images[outer*num_1fold:(outer+1)*num_1fold]
            labels_test=total_labels[outer*num_1fold:(outer+1)*num_1fold]  
            
            total_batch=int(np.shape(labels_train)[0]/batch_size)    
            
            # train and get cost    
            for epoch in np.arange(total_epoch):            
                cost_epoch=0.0
                
                # Begin Annealing
                if beginAnneal == 0:
                    lr = lr * 1.0
                elif epoch+1 > beginAnneal:
                    lr = max( min_lr, (-decay_rate*(epoch+1) + (1+decay_rate*beginAnneal)) * lr ) 
                    
                    
                # shuffle data in every epoch           
                total_sample = np.size(images_train, axis=0)
                sample_ids = np.arange(total_sample)
                np.random.shuffle(sample_ids) 
                                
                
                for batch in np.arange(total_batch):                       
                    batch_x = images_train[sample_ids[batch*batch_size:(batch+1)*batch_size]]
                    batch_y = labels_train[sample_ids[batch*batch_size:(batch+1)*batch_size]]
                    
                    # Get cost and optimize the model
                    if autoencoder==False:
                        cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Y:batch_y, Beta_vec:beta_vec })
                        
                    else:                      
                        cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Beta_vec:beta_vec })
                        
                    cost_epoch+=cost_batch/total_batch        
                
            
    
                     # run weight sparsity control function
                    for i in np.arange(np.shape(nodes)[0]-2):
                        [hsp[i],beta[i]]=Hoyers_sparsity_control(w[i], beta[i], max_beta[i], tg_hsp[i])   
                        
                    if mode=='layer':               
                        beta_vec=beta                      
                    elif mode=='node':                              
                        beta_vec=[item for sublist in beta for item in sublist]
                
                if autoencoder==False:
                    images_train_shuff = np.array([ images_train[i] for i in sample_ids])
                    labels_train_shuff = np.array([ labels_train[i] for i in sample_ids])
                    
                    train_err_epoch=sess.run(error,{X:images_train_shuff, Y:labels_train_shuff})
                    plot_train_err=np.hstack([plot_train_err,[train_err_epoch]])
                    
                    test_err_epoch=sess.run(error,{X:images_test, Y:labels_test})
                    plot_test_err=np.hstack([plot_test_err,[test_err_epoch]])
    
    
                # make space to plot beta, sparsity level
                plot_lr=np.hstack([plot_lr,[lr]])
                plot_cost=np.hstack([plot_cost,[cost_epoch]])
                
                # save footprint for plot
                if mode=='layer':
                    plot_hsp=[np.vstack([plot_hsp[i],[hsp[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[beta_vec[i]]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    
                elif mode=='node':
                    plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta[i])]]) for i in np.arange(np.shape(nodes)[0]-2)]


                
                
            ######################################## Outer test ################################################ 
           
            
            # Print final accuracy of test set
            if autoencoder==False:         
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
            #    plt.yscale('log')
                plt.show()   
                
              
                # Plot train & test error
                plt.title("Training & Test error",fontsize=16)
                plot_train_err=plot_train_err[1:]
                plt.plot(plot_train_err)
                plt.hold
                plot_test_err=plot_test_err[1:]
                plt.plot(plot_test_err)
                plt.ylim(0.0, 1.0)
                plt.legend(['Training error', 'Test error'],loc='upper right')
            #    plt.yscale('log')
                plt.show() 
                
                err=sess.run(error,{X:images_test, Y:labels_test})
                fianl_accuracy_list.append(1-err)
            
        
 
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
                
                
            sess.run(init)
            lr = lr_init
            beta, beta_vec, hsp, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = initialization()

        print("")
        print("")
        print("******************************* Final results **********************************")   
        for k in np.arange(k_folds) :   
            print("Finally selected target hsp in outer-loop",k+1,":",tg_hsp_selected_list[k], ", Accuracy :","{:.4f}".format(fianl_accuracy_list[k]))
        print("=> Final average accuracy :","{:.4f}".format(np.array(fianl_accuracy_list).mean())) 
                

            
        
            

            
else:
    # Don't run the sesstion but print 'failed' if any condition is unmet
    print("Failed!")  
     







