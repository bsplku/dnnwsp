#!/usr/bin/env python3
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
import itertools
import timeit 
import datetime

################################################# Customization part #################################################


# Time markers
date_array=[]
# 1st element of date_array
date_array.append(str(timeit.time.ctime()))
start_time = timeit.default_timer()

    
"""
autoencoder or not
"""
autoencoder=False



"""
Set learning parameters
"""

k_folds=5

"""
Select optimizer
'GradientDescent'
'Adagrad'
'Adam'
'Momentum'
'RMSProp'
"""
optimizer_algorithm='GradientDescent'

momentum=0.01

""" 
Set the number of nodes for input, output and each hidden layer here
"""
n_subjects=3
n_nodes=[74484,100,100,100,4]

"""
Set learning parameters
"""

# Set total epoch
n_epochs=30
# Set mini batch size
batch_size=40
# Let anealing to begin after **th epoch
beginAnneal=50
#beginAnneal=100
# anealing decay rate
decay_rate=0.0003
#decay_rate=0.0002
# Set initial learning rate and minimum                     
lr_init = 0.002   
lr_min = 0

# Set learning rate of beta for weight sparsity control
beta_lrates = 0.95
# Set L2 parameter for L2 regularization
L2_reg= 1e-4


"""
Select the sparsity control mode
'layer' for layer wise sparsity control
'node' for node wise sparsity control
"""
mode = 'layer'


"""
Set maximum beta value of each hidden layer
and set target sparsness value (0:dense~1:sparse)
"""
max_beta = [0.015, 0.25, 0.25]
#max_beta = [0.0001, 0.7, 0.7]

# automatically makes combination sets [(0.3 or 0.7) , (0.3 or 0.7), (0.3 or 0.7)]
tg_hspset_list = list(itertools.product([0.3, 0.7],[0.3, 0.7],[0.3, 0.7]))
tg_hspset_list=[list(i) for i in tg_hspset_list]
n_tg_hspset_list = len(tg_hspset_list)


#current_directory = os.getcwd()
current_directory = '/home/hailey/03_code/weight_sparsity_control'

dtime=datetime.datetime.now()
# make a new result directory in the current directory
dir_root = os.path.join(current_directory, r'results_CV_%s_%d%02d%02d_%02d%02d'%(mode,dtime.year,dtime.month,dtime.day,dtime.hour,dtime.minute))
if not os.path.exists(dir_root):
    os.makedirs(dir_root) 



f = open(dir_root+"/parameters.txt",'w') 
f.write('subject # : '+str(n_subjects)+'\n')
f.write('mode : '+str(mode)+'\n')
f.write('optimizer_algorithm : '+str(optimizer_algorithm)+'\n')
f.write('n_epochs : '+str(n_epochs)+'\n')
f.write('batch_size : '+str(batch_size)+'\n')
f.write('beginAnneal : '+str(beginAnneal)+'\n')
f.write('decay_rate : '+str(decay_rate)+'\n')
f.write('lr_init : '+str(lr_init)+'\n')
f.write('lr_min : '+str(lr_min)+'\n')
f.write('beta_lrates : '+str(beta_lrates)+'\n')
f.write('L2_reg : '+str(L2_reg)+'\n')
f.write('max_beta : '+str(max_beta)+'\n')
f.close()

################################################# Input data #################################################

datasets = sio.loadmat('/home/hailey/03_code/weight_sparsity_control/lhrhadvs_sample_data.mat')

train_x_ = datasets['train_x']
train_y_ = np.zeros((np.shape(datasets['train_y'])[0],np.max(datasets['train_y'])+1))
# transform into One-hot
for i in np.arange(np.shape(datasets['train_y'])[0]):
    train_y_[i][datasets['train_y'][i][0]]=1 

test_x_ = datasets['test_x']
test_y_ = np.zeros((np.shape(datasets['test_y'])[0],np.max(datasets['test_y'])+1))
# transform into One-hot
for i in np.arange(np.shape(datasets['test_y'])[0]):
    test_y_[i][datasets['test_y'][i][0]]=1 

total_x=np.vstack([train_x_,test_x_])
total_y=np.vstack([train_y_,test_y_])

#datasets = sio.loadmat('mym_vectors_and_labels.mat')
#
#total_x = datasets['samples']
#total_y = datasets['labels']


num_total = np.size(total_x, axis=0)  # total number of examples to use
num_1fold = int(num_total / k_folds)


    
################################################# Build Model #################################################


#with tf.device('/cpu:0'):    
with tf.device('/gpu:0'):
    
    # 'node_index' to split placeholder, for an example, given hidden_nodes=[100, 100, 100], nodes_index=[0, 100, 200, 300]
    nodes_index= [int(np.sum(n_nodes[1:i+1])) for i in np.arange(np.shape(n_nodes)[0]-1)]
    
    # Make two placeholders to fill the values later when training or testing
    X=tf.placeholder(tf.float32,[None,n_nodes[0]])
    Y=tf.placeholder(tf.float32,[None,n_nodes[-1]])
    
    # Create randomly initialized weight variables 
    w_init=[tf.random_uniform([n_nodes[i],n_nodes[i+1]], minval=-tf.sqrt(tf.div(6.0,n_nodes[i]+n_nodes[i+1])), maxval=tf.sqrt(tf.div(6.0,n_nodes[i]+n_nodes[i+1]))) for i in np.arange(np.shape(n_nodes)[0]-1)]
    w=[tf.Variable(w_init[i], dtype=tf.float32) for i in np.arange(np.shape(n_nodes)[0]-1)]
    
    # Create randomly initialized bias variables 
    b_init=[tf.zeros([n_nodes[i+1]]) for i in np.arange(np.shape(n_nodes)[0]-1)]
    b=[tf.Variable(b_init[i], dtype=tf.float32) for i in np.arange(np.shape(n_nodes)[0]-1)]
    
    # Build MLP model 
    layers_hidden=[0.0]*(np.shape(n_nodes)[0]-2)
    for i in np.arange(np.shape(n_nodes)[0]-2):
        # Input layer
        if i==0:
            layers_hidden[i] = tf.add(tf.matmul(X,w[i]),b[i])
            layers_hidden[i] = tf.nn.tanh(layers_hidden[i])
                       
        # The other layers    
        else:     
            layers_hidden[i]=tf.add(tf.matmul(layers_hidden[i-1],w[i]),b[i])
            layers_hidden[i]=tf.nn.tanh(layers_hidden[i])
    # Output layer
    layers_output=tf.add(tf.matmul(layers_hidden[-1],w[-1]),b[-1])
    
    # Logistic regression layer
    layer_logRegression=tf.nn.tanh(layers_output)
                       
    
    ############################################# Function Definition #############################################
    
    
    # Make placeholders for total beta array (make a long one to concatenate every beta vector) 
    def init_beta():
        if mode=='layer':
            # The size is same with the number of layers
            Beta=tf.placeholder(tf.float32,[np.shape(n_nodes)[0]-2])
        elif mode=='node':
            # The size is same with the number of nodes
            Beta=tf.placeholder(tf.float32,[np.sum(n_nodes[1:-1])])
    
        return Beta
    
    
    # Make L1 loss term for regularization
    def init_L1loss():
        if mode=='layer':
            # Get L1 loss term by simply multiplying beta(scalar value) and L1 norm of weight for each layer
            L1_loss=[tf.reduce_sum(abs(w[i])*Beta[i]) for i in np.arange(np.shape(n_nodes)[0]-2)]
        elif mode=='node':
            # Get L1 loss term by multiplying beta(vector values as many as nodes) and L1 norm of weight for each layer
            L1_loss=[tf.reduce_sum(tf.matmul(abs(w[i]),tf.reshape(Beta[nodes_index[i]:nodes_index[i+1]],[-1,1])) for i in np.arange(np.shape(n_nodes)[0]-2))]
            
        L1_loss_total=tf.reduce_sum(L1_loss)
    
        return L1_loss_total
    
    
    # Make L2 loss term for regularization
    def init_L2loss():
        L2_loss=[tf.reduce_sum(tf.square(w[i])*L2_reg) for i in np.arange(np.shape(n_nodes)[0]-1)] 
        
        L2_loss_total=tf.reduce_sum(L2_loss) 
        
        return L2_loss_total
    
    
           
    
           
    
    # Define cost term with cross entropy and L1 and L2 tetm     
    def init_cost():
        if autoencoder==False:
            # A softmax regression : it adds up the evidence of our input being in certain classes, and converts that evidence into probabilities.
            cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_logRegression, labels=Y)) \
                                         + L1_loss_total + L2_loss_total 
    
        else:              
            cost=tf.reduce_mean(tf.pow(X - layers_output, 2)) + L1_loss_total + L2_loss_total
           
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
            optimizer=tf.train.MomentumOptimizer(Lr,momentum).minimize(cost) 
        elif optimizer_algorithm=='RMSProp':
            optimizer=tf.train.RMSPropOptimizer(Lr).minimize(cost) 
    
        return optimizer
    
    
    
    # initialization   
    def init_otherVariables():           
        if mode=='layer': 
            beta_val = np.zeros(np.shape(n_nodes)[0]-2)
            beta = np.zeros(np.shape(n_nodes)[0]-2)
            hsp_val = np.zeros(np.shape(n_nodes)[0]-2)            
            plot_beta = np.zeros(np.shape(n_nodes)[0]-2)
            plot_hsp = np.zeros(np.shape(n_nodes)[0]-2)
                       
        elif mode=='node':                       
            beta_val = [np.zeros(n_nodes[i+1]) for i in np.arange(np.shape(n_nodes)[0]-2)]  
            beta = np.zeros(np.sum(n_nodes[1:-1]))
            hsp_val = [np.zeros(n_nodes[i+1]) for i in np.arange(np.shape(n_nodes)[0]-2)]            
            plot_beta = [np.zeros(n_nodes[i+1]) for i in np.arange(np.shape(n_nodes)[0]-2)]
            plot_hsp = [np.zeros(n_nodes[i+1]) for i in np.arange(np.shape(n_nodes)[0]-2)]
        
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
    L1_loss_total = init_L1loss()
    L2_loss_total = init_L2loss()
    cost = init_cost()
    
    optimizer=init_optimizer(Lr)
    
    if autoencoder==False:
        predict_ans=tf.argmax(tf.nn.softmax(layer_logRegression),1)
        correct_ans=tf.argmax(Y,1)
        correct_prediction=tf.equal(predict_ans,correct_ans)  
        # calculate an average error depending on how frequent it classified correctly   
        error=1-tf.reduce_mean(tf.cast(correct_prediction,tf.float32))      
        
    
    
    
    lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = init_otherVariables()
    
    
    
    
    
    if mode=='layer':
        # Weight sparsity control with Hoyer's sparsness (Layer wise)  
        def Hoyers_sparsity_control(w_,b,max_b,tg):
            
            # Get value of weight
            W=sess.run(w_)
            [dim,n_nodes]=W.shape  
            num_elements=n_nodes*dim
     
            Wvec=W.flatten()
             
            # Calculate L1 and L2 norm     
            L1=LA.norm(Wvec,1)
            L2=LA.norm(Wvec,2)
            
            # Calculate hoyer's sparsness
            h=(np.sqrt(num_elements)-(L1/L2))/(np.sqrt(num_elements)-1)
            
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
            [dim,n_nodes]=W.shape
            
            # Calculate L1 and L2 norm 
            L1=LA.norm(W,1,axis=0)
            L2=LA.norm(W,2,axis=0)
            
            h_vec = np.zeros((1,n_nodes))
            tg_vec = np.ones(n_nodes)*tg
            
            # Calculate hoyer's sparsness
            h_vec=(np.sqrt(dim)-(L1/L2))/(np.sqrt(dim)-1)
            
            # Update beta
            b_vec-=beta_lrates*np.sign(h_vec-tg_vec)
            
            # Trim value
            b_vec[b_vec<0.0]=0.0
            b_vec[b_vec>max_b]=max_b
            
                   
            return [h_vec,b_vec]
    
    
    

############################################ Condition check #############################################


condition=True

    
################################################ Learning ################################################



if condition==True:
    
    # variables are not initialized when you call tf.Variable. 
    # To initialize all the variables in a TensorFlow program, you must explicitly call a special operation         
    init = tf.global_variables_initializer()              

    
#    with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
        # run tensorflow variable initialization
        sess.run(init)
        
        tg_hsp_selected_list=[]
        fianl_accuracy_list=[]
             
 
        for outer in range(k_folds):
        
        
            print("*********************** outer fold (",outer+1, ") **************************")        
            
            lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = init_otherVariables()

            
            # outer_train_list
            outer_train_list=np.delete(np.arange(k_folds),outer)        
            
            
            error_list=list() 
            
            # for each candidate sets
            for tg_hspset in tg_hspset_list:
                            
     
                avg_err=0.0
                
                ######################################## Inner train ################################################
                for inner in outer_train_list:
                    
                    inner_train_list = np.delete(outer_train_list, np.argwhere(outer_train_list == inner))
                    
                    train_x = []
                    train_y = []
                    valid_x = []
                    valid_y = []

                    # make training data set
                    for j in inner_train_list:
                        if np.size(train_x) == 0 & np.size(train_y) == 0:
                            train_x = total_x[j*num_1fold : (j+1)*num_1fold]
                            train_y = total_y[j*num_1fold : (j+1)*num_1fold]
                        else:
                            train_x = np.vstack([train_x, total_x[j*num_1fold : (j+1)*num_1fold]])
                            train_y = np.vstack([train_y, total_y[j*num_1fold : (j+1)*num_1fold]])

                    valid_x = total_x[inner * num_1fold:(inner+1) * num_1fold]
                    valid_y = total_y[inner * num_1fold:(inner+1) * num_1fold]

                    
                    # train and get cost    
                    for epoch in np.arange(n_epochs):            
                        
                        
                        # Begin Annealing
                        if beginAnneal == 0:
                            lr = lr * 1.0
                        elif epoch+1 > beginAnneal:
                            lr = max( lr_min, (-decay_rate*(epoch+1) + (1+decay_rate*beginAnneal)) * lr ) 
                            
                            
                        # shuffle data in every epoch           
                        total_sample = np.size(train_y, axis=0)
                        sample_ids = np.arange(total_sample)
                        np.random.shuffle(sample_ids) 

                        
                        train_x = train_x[[sample_ids]]
                        train_y = train_y[[sample_ids]]
                        
                        # Calculate how many mini-batch iterations we need
                        total_batch = int(np.shape(train_x)[0]/batch_size) 
                        
                        cost_epoch=0.0
                        
                        # minibatch based training  
                        for batch in np.arange(total_batch):                       
                            batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                            batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
                                         
                            
                            # Get cost and optimize the model
                            if autoencoder==False:
                                cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Y:batch_y, Beta:beta })
                                
                            else:                      
                                cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Beta:beta })
                                
                            cost_epoch+=cost_batch/total_batch    
                            
                            
                            # weight sparsity control    
                            for i in np.arange(np.shape(n_nodes)[0]-2):
                                [hsp_val[i], beta_val[i]] = Hoyers_sparsity_control(w[i], beta_val[i], max_beta[i], tg_hspset[i]) 
                                
                            if mode=='layer':                     
                                beta=beta_val                      
            
                            elif mode=='node':                              
                                # flatten beta_val (shape (3, 100) -> (300,))
                                beta=[item for sublist in beta_val for item in sublist]
                        
                        if autoencoder==False:
                            
                            train_err_epoch=sess.run(error,{X:train_x, Y:train_y})
                            plot_train_err=np.hstack([plot_train_err,[train_err_epoch]])
                            
                            test_err_epoch=sess.run(error,{X:valid_x, Y:valid_y})
                            plot_test_err=np.hstack([plot_test_err,[test_err_epoch]])
            
            
                        
                        # Save the results to plot at the end
                        plot_lr=np.hstack([plot_lr,[lr]])
                        plot_cost=np.hstack([plot_cost,[cost_epoch]])
                        
                        if mode=='layer':
                            plot_hsp=[np.vstack([plot_hsp[i],[hsp_val[i]]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                            plot_beta=[np.vstack([plot_beta[i],[beta[i]]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                            
                        elif mode=='node':
                            plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp_val[i])]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                            plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta_val[i])]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
            
                        
       
                    ######################################## Inner validate ################################################   
                 
                    # Print final accuracy of test set
                    if autoencoder==False:      
         
                        # make a new 'results' directory in the current directory
                        tghsp=['0'+str(int(i*10)) for i in tg_hspset]
                        tghsp=''.join(tghsp)
                        
                        # make a new 'results' directory in the current directory
                        final_directory = os.path.join(dir_root, r'outer%d/tg_%s/inner%d'%(outer+1,tghsp,inner+1))
                        if not os.path.exists(final_directory):
                            os.makedirs(final_directory) 


                        # Plot the change of learning rate
                        plot=plt.figure() 
                        plt.title("Learning rate plot",fontsize=16)
                        plot_lr=plot_lr[1:]
                        plt.ylim(0.0, lr_init*1.2)
                        plt.plot(plot_lr)
                        plt.savefig(final_directory+'/learning_rate.png')
                        plt.show(block=False)
                        plt.close(plot)
                        
                        # Plot the change of cost
                        plot=plt.figure() 
                        plt.title("Cost plot",fontsize=16)
                        plot_cost=plot_cost[1:]
                        plt.plot(plot_cost)
                        plt.savefig(final_directory+'/cost.png')
                        plt.show(block=False)
                        plt.close(plot)
        
                        # Plot train & test error
                        plot=plt.figure() 
                        plt.title("Train & Validation error",fontsize=16)
                        plot_train_err=plot_train_err[1:]
                        plt.plot(plot_train_err)
                        plt.hold
                        plot_test_err=plot_test_err[1:]
                        plt.plot(plot_test_err)
                        plt.ylim(0.0, 1.0)
                        plt.legend(['Train error', 'Validation error'],loc='upper right')
                        plt.savefig(final_directory+'/error.png')
                        plt.show(block=False) 
                        plt.close(plot)
                        
                        
                        
                    # Plot the change of beta value
                    print("")   
                    plot=plt.figure()
                    for i in np.arange(np.shape(n_nodes)[0]-2):            
                        plot_beta[i]=plot_beta[i][1:]
                        plt.plot(plot_beta[i], label='layer%d'%(i+1))
                        plt.hold
                    plt.title("Beta plot",fontsize=16)
                    plt.ylim(0.0, np.max(max_beta)*1.2)
                    plt.legend()
                    plt.savefig(final_directory+'/beta.png')
                    plt.show(block=False)
                    plt.close(plot)
                    
                    
                    # Plot the change of Hoyer's sparsity
                    print("")   
                    plot=plt.figure()
                    for i in np.arange(np.shape(n_nodes)[0]-2):            
                        plot_hsp[i]=plot_hsp[i][1:]
                        plt.plot(plot_hsp[i], label='layer%d'%(i+1))
                        plt.hold
                    plt.title("Hoyer's sparsity plot",fontsize=16)
                    plt.ylim(0.0, 1.0)
                    plt.legend()
                    plt.savefig(final_directory+'/hsp.png')
                    plt.show(block=False)
                    plt.close(plot)
         

                    print("")
                    print(">>>> (", np.argwhere([tg_hspset==i for i in tg_hspset_list])[0][0]+1 ,") Target hsp",tg_hspset ,"<<<<")
                    print("outer fold :",outer+1,"/",k_folds," &  inner fold :",np.argwhere(outer_train_list==inner)[0][0]+1,"/",k_folds-1)
                    print("Accuracy :","{:.3f}".format(1-plot_test_err[-1]))
                    
                    if mode=='layer':
                        print("beta :",['%.3f' %plot_beta[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)], " / hsp :",['%.3f' %plot_hsp[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)])
                    elif mode=='node':
                        print("beta :",['%.3f' %plot_beta[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)], " / hsp :",['%.3f' %np.mean(plot_hsp[i][-1]) for i in np.arange(np.shape(n_nodes)[0]-2)])
                    print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")
                        
                    avg_err+=plot_test_err[-1]/np.size(outer_train_list)
                    
                    
                    
                    
                        
                    # save results as .mat file
                    sio.savemat(final_directory+"/result_learningrate.mat", mdict={'lr': plot_lr})
                    sio.savemat(final_directory+"/result_cost.mat", mdict={'cost': plot_cost})
                    sio.savemat(final_directory+"/result_train_err.mat", mdict={'trainErr': plot_train_err})
                    sio.savemat(final_directory+"/result_validation_err.mat", mdict={'validationErr': plot_test_err})
                    sio.savemat(final_directory+"/result_beta.mat", mdict={'beta': plot_beta})
                    sio.savemat(final_directory+"/result_hsp.mat", mdict={'hsp': plot_hsp})
                    sio.savemat(final_directory+"/result_init_weight.mat", mdict={'init_weight':sess.run(w_init)})
                    sio.savemat(final_directory+"/result_weight.mat", mdict={'weight':sess.run(w)})
                    sio.savemat(final_directory+"/result_init_bias.mat", mdict={'init_bias':sess.run(b_init)})
                    sio.savemat(final_directory+"/result_bias.mat", mdict={'bias':sess.run(b)})

                    

                    # save time 
                    f = open(final_directory+"/time_info.txt",'w')           # opens file with name of "time_info.txt"
                    f.write('saved at * '+str(timeit.time.ctime())+' * \n')
                    f.close()                     
                    
                    
                    sess.run(init)
                    lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = init_otherVariables()
                    
                error_list.append(avg_err)
                print("")
                print("##############################################################################")
                print("# Avg validation error on this iteration for (", np.argwhere([tg_hspset==i for i in tg_hspset_list])[0][0]+1 ,")",tg_hspset,"is" ,"{:.4f}".format(avg_err),"#")
                print("##############################################################################")
                print("")
                
                
                
            # select target sparsity set when validation error is min
            where_is_best=np.argmin(error_list)
            # store selected set in a list
            tg_hsp_selected_list.append(tg_hspset_list[where_is_best])
            
        
            ######################################## Outer train ################################################ 
                         
            train_x = []
            train_y = []
            valid_x = []
            valid_y = []
                   

            # make training data set
            for j in outer_train_list:
                if np.size(train_x) == 0 & np.size(train_y) == 0:
                    train_x = total_x[j*num_1fold:(j+1) * num_1fold]
                    train_y = total_y[j*num_1fold:(j+1) * num_1fold]
                else:
                    train_x = np.vstack([train_x, total_x[j * num_1fold:(j+1) * num_1fold]])
                    train_y = np.vstack([train_y, total_y[j * num_1fold:(j+1) * num_1fold]])

            test_x = total_x[outer*num_1fold : (outer+1)*num_1fold]
            test_y = total_y[outer*num_1fold : (outer+1)*num_1fold]

            train_predict_ans=np.zeros((n_epochs,np.size(train_y, axis=0)))
            train_correct_ans=np.zeros((n_epochs,np.size(train_y, axis=0)))
            test_predict_ans=np.zeros((n_epochs,np.size(test_y, axis=0)))
            test_correct_ans=np.zeros((n_epochs,np.size(test_y, axis=0)))
            
            
            # train and get cost    
            for epoch in np.arange(n_epochs):            
                
                # Begin Annealing
                if beginAnneal == 0:
                    lr = lr * 1.0
                elif epoch+1 > beginAnneal:
                    lr = max( lr_min, (-decay_rate*(epoch+1) + (1+decay_rate*beginAnneal)) * lr ) 
                    
                # Shuffle training data at the begining of each epoch           
                total_sample = np.size(train_y, axis=0)
                sample_ids = np.arange(total_sample)
                np.random.shuffle(sample_ids) 
                
                train_x = np.array([train_x[i] for i in sample_ids])
                train_y = np.array([train_y[i] for i in sample_ids])
                
                    
                    
                # Calculate how many mini-batch iterations we need
                total_batch=int(np.shape(train_y)[0]/batch_size) 
                
                cost_epoch=0.0
                
                # minibatch based training  
                for batch in np.arange(total_batch):                       
                    batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
                    batch_y = train_y[batch*batch_size:(batch+1)*batch_size]
                    
                    # Get cost and optimize the model
                    if autoencoder==False:
                        cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Y:batch_y, Beta:beta })
                        
                    else:                      
                        cost_batch,_=sess.run([cost,optimizer],{Lr:lr, X:batch_x, Beta:beta })
                        
                    cost_epoch+=cost_batch/total_batch        
                
            

                    # weight sparsity control  
                    for i in np.arange(np.shape(n_nodes)[0]-2):
                        [hsp_val[i], beta_val[i]] = Hoyers_sparsity_control(w[i], beta_val[i], max_beta[i], tg_hsp_selected_list[-1][i])  
                    
                    if mode=='layer':                                            
                        beta=beta_val                      
    
                    elif mode=='node':                             
                        # flatten beta_val (shape (3, 100) -> (300,))
                        beta=[item for sublist in beta_val for item in sublist]

                
                if autoencoder==False:            
                    
                    [train_err_epoch,train_predict_ans[epoch],train_correct_ans[epoch]]=sess.run([error,predict_ans,correct_ans],{X:train_x, Y:train_y})
                    plot_train_err=np.hstack([plot_train_err,[train_err_epoch]])
                    
                    [test_err_epoch,test_predict_ans[epoch],test_correct_ans[epoch]]=sess.run([error,predict_ans,correct_ans],{X:test_x, Y:test_y})
                    plot_test_err=np.hstack([plot_test_err,[test_err_epoch]])
                    
 
    
                # Save the results to plot at the end
                plot_lr=np.hstack([plot_lr,[lr]])
                plot_cost=np.hstack([plot_cost,[cost_epoch]])
                
          
                if mode=='layer':
                    plot_hsp=[np.vstack([plot_hsp[i],[hsp_val[i]]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[beta[i]]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                    
                elif mode=='node':
                    plot_hsp=[np.vstack([plot_hsp[i],[np.transpose(hsp_val[i])]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
                    plot_beta=[np.vstack([plot_beta[i],[np.transpose(beta_val[i])]]) for i in np.arange(np.shape(n_nodes)[0]-2)]
            

         
                
            ######################################## Outer test ################################################ 
           
            
            # Print final accuracy of test set
            if autoencoder==False: 
                
                # make a new 'results' directory in the current directory
                selectedhsp=['0'+str(int(i*10)) for i in tg_hsp_selected_list[-1]]
                selectedhsp=''.join(selectedhsp)
                
                final_directory = os.path.join(dir_root, r'outer%d_selected_%s'%(outer+1,selectedhsp))
                if not os.path.exists(final_directory):
                    os.makedirs(final_directory) 
                    
                # Plot the change of learning rate
                plot=plt.figure() 
                plt.title("Learning rate plot",fontsize=16)
                plot_lr=plot_lr[1:]
                plt.ylim(0.0, lr_init*1.2)
                plt.plot(plot_lr)
                plt.savefig(final_directory+'/learning_rate.png')
                plt.show(block=False)
                plt.close(plot)
                
                # Plot the change of cost
                plot=plt.figure() 
                plt.title("Cost plot",fontsize=16)
                plot_cost=plot_cost[1:]
                plt.plot(plot_cost)
                plt.savefig(final_directory+'/cost.png')
                plt.show(block=False)  
                plt.close(plot)

                # Plot train & test error
                plot=plt.figure() 
                plt.title("Train & Test error",fontsize=16)
                plot_train_err=plot_train_err[1:]
                plt.plot(plot_train_err)
                plt.hold
                plot_test_err=plot_test_err[1:]
                plt.plot(plot_test_err)
                plt.ylim(0.0, 1.0)
                plt.legend(['Train error', 'Test error'],loc='upper right')
                plt.savefig(final_directory+'/error.png')
                plt.show(block=False) 
                plt.close(plot)
                
                fianl_accuracy_list.append(1-plot_test_err[-1])
            
        
 
            # Plot the change of beta value
            print("")   
            plot=plt.figure()
            for i in np.arange(np.shape(n_nodes)[0]-2):            
                plot_beta[i]=plot_beta[i][1:]
                plt.plot(plot_beta[i], label='layer%d'%(i+1))
                plt.hold
            plt.title("Beta plot",fontsize=16)
            plt.ylim(0.0, np.max(max_beta)*1.2)
            plt.legend()
            plt.savefig(final_directory+'/beta.png')
            plt.show(block=False)
            plt.close(plot)
            
            
            # Plot the change of Hoyer's sparsity
            print("")   
            plot=plt.figure()
            for i in np.arange(np.shape(n_nodes)[0]-2):            
                plot_hsp[i]=plot_hsp[i][1:]
                plt.plot(plot_hsp[i], label='layer%d'%(i+1))
                plt.hold
            plt.title("Hoyer's sparsity plot",fontsize=16)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.savefig(final_directory+'/hsp.png')
            plt.show(block=False)
            plt.close(plot)
                        
            
            print("")
            print(">>>> (Selected) Target hsp",tg_hsp_selected_list[-1] ,"<<<<")
            print("outer fold :",outer+1,"/",k_folds)
            print("Accuracy :","{:.3f}".format(1-plot_test_err[-1]))
            
            if mode=='layer':
                print("beta :",['%.3f' %plot_beta[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)], " / hsp :",['%.3f' %plot_hsp[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)])
            elif mode=='node':
                print("beta :",['%.3f' %plot_beta[i][-1][0] for i in np.arange(np.shape(n_nodes)[0]-2)], " / hsp :",['%.3f' %np.mean(plot_hsp[i][-1]) for i in np.arange(np.shape(n_nodes)[0]-2)])
            print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")

        
                
            # save results as .mat file
            sio.savemat(final_directory+"/result_learningrate.mat", mdict={'lr': plot_lr})
            sio.savemat(final_directory+"/result_cost.mat", mdict={'cost': plot_cost})
            sio.savemat(final_directory+"/result_train_err.mat", mdict={'trainErr': plot_train_err})
            sio.savemat(final_directory+"/result_test_err.mat", mdict={'testErr': plot_test_err})
            sio.savemat(final_directory+"/result_beta.mat", mdict={'beta': plot_beta})
            sio.savemat(final_directory+"/result_hsp.mat", mdict={'hsp': plot_hsp})
            sio.savemat(final_directory+"/result_weight.mat", mdict={'weight':sess.run(w)})
            sio.savemat(final_directory+"/result_init_weight.mat", mdict={'init_weight':sess.run(w_init)})
            sio.savemat(final_directory+"/result_bias.mat", mdict={'bias':sess.run(b)})
            sio.savemat(final_directory+"/result_init_bias.mat", mdict={'init_bias':sess.run(b_init)})
            sio.savemat(final_directory+"/train_predict_ans.mat", mdict={'train_predict_ans':train_predict_ans})
            sio.savemat(final_directory+"/train_correct_ans.mat", mdict={'train_correct_ans':train_correct_ans})
            sio.savemat(final_directory+"/test_predict_ans.mat", mdict={'test_predict_ans':test_predict_ans})
            sio.savemat(final_directory+"/test_correct_ans.mat", mdict={'test_correct_ans':test_correct_ans})

            # save time 
            f = open(final_directory+"/time_info.txt",'w')           # opens file with name of "time_info.txt"
            f.write('saved at * '+str(timeit.time.ctime())+' * \n')
            f.close()
            
            
            # 2nd~6th elements of date_array
            date_array.append(str(timeit.time.ctime()))
            
            sess.run(init)
            lr, beta_val, beta, hsp_val, plot_beta, plot_hsp, plot_lr, plot_cost, plot_train_err, plot_test_err = init_otherVariables()
            
                                         

        # 7th element of date_array
        date_array.append(str(timeit.time.ctime()))
        end_time = timeit.default_timer()

        

        f = open(dir_root+"/time_info_and_results.txt",'w')         
        
        for date in date_array:
            f.write(date+'\n')
        f.write('=> when the code was run \n=> when 1st~5th outer loop finished \n=> when the code ended\n')     
        f.write('')
        f.write(str(os.path.split(__file__)[1])+' ran for %.2f hours = %d mins' %((end_time-start_time)/(60*60),(end_time-start_time)/60)+'\n\n')
        f.write("******************************* Final results **********************************\n")
        print("")
        print("")
        print("******************************* Final results **********************************")   
        for k in np.arange(k_folds) :   
            print("Finally selected target hsp in outer-loop",k+1,":",tg_hsp_selected_list[k], ", Accuracy :","{:.4f}".format(fianl_accuracy_list[k]))
            f.write("Finally selected target hsp in outer-loop"+str(k+1)+":"+str(tg_hsp_selected_list[k])+", Accuracy :"+str(fianl_accuracy_list[k])+'\n')
        FinalAvgAccuracy=np.array(fianl_accuracy_list).mean()        
        print("=> Final average accuracy :","{:.4f}".format(FinalAvgAccuracy)) 
        f.write("=> Final average accuracy :"+str(FinalAvgAccuracy))                
        f.close()
        
        print()
        print(str(os.path.split(__file__)[1])+' ran for %.2f hours = %d mins' %((end_time-start_time)/(60*60),(end_time-start_time)/60))
                    
                
                
                    
else:
    # Don't run the sesstion but print 'failed' if any condition is unmet
    print("Failed!")  
     







