
"""
Deep neural network (DNN) with weight sparsity control (DNN-WSP; i.e., L1-norm regularization)
improved the classification performance using whole-brain resting-state functional 
connectivity patterns of schizophrenia patient and healthy groups. Initializing DNN's
weights through stacked auto-encoder enhanced the classification performance as well. 
(Jang et al., Neuroimage 2017, Kim et al., Neuroimage, 2016). The Python codes were 
modified from the DeepLearningTutorials (https://github.com/lisa-lab/DeepLearningTutorials)
to apply a node-wise and layer-wise control of weight sparsity via Hoyer sparseness (Kim and Lee, PRNI2016 & ICASSP2017).

This code is for the regression analysis using DNN with the nonde-wise weight sparsity control.
 
"""

################################################# Import #################################################

import os
import sys # To print error or simple message
import timeit  # To calculate computational time

import numpy # NumPy is the fundamental package for scientific computing with Python.
import numpy as np  # Simplification
from numpy import linalg as LA 

import scipy.io as sio # The module for file input and output
import scipy.stats # This module contains a large number of probability distributions as well as a growing library of statistical functions.

import theano # Theano is the fundamental package for scientific computing with Python.
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

########################################## Function definition #################################################
# Define the node-wise control of weight sparsity via Hoyer sparseness (Hoyer, 2014, Kim and Lee PRNI2016, Kim and Lee ICASSP 2017)

def hsp_fnc_inv_mat_cal(val_L1_ly, W, thre, tg, lrate):
    W = np.array(W.get_value(borrow=True));
    
    [dim, nodes] = W.shape
    
    cnt_L1_ly = val_L1_ly;
    
    hsp_vec = np.zeros((1,nodes));  
    
    tg_vec = np.ones(nodes)*tg;
    sqrt_nsamps = pow(dim,0.5)
        
    n1_W = LA.norm(W,1,axis=0);    n2_W = LA.norm(W,2,axis=0);
    hsp_vec = (sqrt_nsamps - (n1_W/n2_W))/(sqrt_nsamps-1)
    
    cnt_L1_ly -= lrate*np.sign(hsp_vec-tg_vec)
        
    for ii in range(0,nodes):
        if cnt_L1_ly[ii] < 0:
            cnt_L1_ly[ii] = 0
        if cnt_L1_ly[ii] > thre:
            cnt_L1_ly[ii] = thre
        
    hspset =[hsp_vec, cnt_L1_ly]

    return hspset
   
def get_corrupted_input(input,corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``corruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``
                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input


########################################## Class definition #################################################

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.ones((n_in, n_out), \
                                                 dtype=theano.config.floatX), \
                                                    name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.ones((n_out,), \
                                                 dtype=theano.config.floatX), \
                                                    name='b', borrow=True)

        self.p_y_given_x =  T.dot(input, self.W) + self.b
        self.y_pred = self.p_y_given_x[:,0]
#         print self.y_pred.size
        # parameters of the model
        self.params = [self.W, self.b]
        
    def errors(self, y):
       # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y',  y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            return T.sum(abs(y-self.y_pred))
        else:
            raise NotImplementedError()
      

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
       
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

# start-snippet-2
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_hidden3, n_out,corruption_level,is_train):
        
        #Get corrupted input data 
        corrupted_x = get_corrupted_input(input,corruption_level)
        input_x = T.switch(T.neq(is_train, 0), corrupted_x, input)
        
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input_x,
            n_in=n_in,
            n_out=n_hidden1,
            activation=T.tanh
        )
        
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden1,
            n_out=n_hidden2,
            activation=T.tanh
        )
        
        self.hiddenLayer3 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_hidden2,
            n_out=n_hidden3,
            activation=T.tanh
        )
         # The Linear regression layer gets as input the hidden units
        # of the hidden layer
        self.linearRegressionLayer = LinearRegression(
            input=self.hiddenLayer3.output,
            n_in=n_hidden3,
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1_layer1 = (
            abs(self.hiddenLayer1.W).sum()
        )
        self.L1_layer2 = (
            abs(self.hiddenLayer2.W).sum()
        )
        self.L1_layer3 = (
            abs(self.hiddenLayer3.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.hiddenLayer3.W ** 2).sum()
            + (self.linearRegressionLayer.W ** 2).sum()
        )
        
        self.errors = self.linearRegressionLayer.errors
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.hiddenLayer3.params+ self.linearRegressionLayer.params
        self.oldparams = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape, dtype=theano.config.floatX)) for p in self.params]
        self.input = input

########################################## Parameters of dnnwsp #################################################

def test_mlp():
#    rootpath = '/root/sharedfolder/code/demo_18aug22'
#    save_path = '/root/sharedfolder/code/demo_18aug22'
    rootpath = '/root/sharedfolder/code/emt_dnn/test'
    save_path = '/root/sharedfolder/code/emt_dnn/test'
              
    # save_path = '/Users/bspl/Desktop/regssion'
    
    save_name = '%s/rst_vlnc_predcition.mat' % (save_path)  # a directory to save dnnwsp result  

    [n_in, n_hidden1, n_hidden2, n_hidden3, n_output] =[55417,20,20,20,1] # DNN strcture
    val_L2 = 1e-5;    # L2-norm parameter
    itlrate = 0.0005;   # learning rate 
    batch_size = 2;   # batch size 
    momentum =0.01;   # momentum
    n_epochs = 500;   # the total number of epoch
    scal_ref = 10;    # the scale for the emotion response
    dcay_rate = 0.99; # decay learning rate for the learning rate 
    corruption_level = 0.3 
    # entries of the inputs the same and zero-out randomly selected subset of size corruption_level
        
    # Parameters for the node-wise control of weight sparsity
    # If you have three hidden layer, the number of target Hoyer's sparseness should be same 
    hsp_level = [0.7, 0.5, 0.3];  # Target sparsity     
    max_beta = [0.03,0.5,0.5];  # Maximum beta changes
    beta_lrates = 1e-2;
    
    rng = np.random.RandomState(8000)

    ########################################## Input data  #################################################
    sbjinfo = sio.loadmat('%s/emt_valence_sample.mat' % rootpath)
    
    ############# emt_sample_data.mat #############
    # train_x  = 64 volumes x 55417 voxels  
    # train_x  = 64 volumes x 1 [valence, arousal or dominance scores for traing]
    # test_x  = 16 volumes x 55417 voxels
    # test_y  = 16 volumes x 1 [valence, arousal or dominance scores for test]
    ############################################################
    
    start_time = timeit.default_timer()
        
    train_x = sbjinfo['train_x']; 
    train_y = np.asarray(sbjinfo['train_y'],'float32').flatten() / scal_ref ;
    
    test_x = sbjinfo['test_x'];
    test_y =  np.asarray(sbjinfo['test_y'],'float32').flatten() / scal_ref ;
    
    n_train_set_x = scipy.stats.zscore(train_x,axis=1,ddof=1)
    n_test_set_x = scipy.stats.zscore(test_x,axis=1,ddof=1)
    
    n_trvld_batches = int(train_x.shape[0] / batch_size)
    n_test_batches = int(n_test_set_x.shape[0] / batch_size)
    
    ########################################## Build model #################################################
      
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.fvectors('y')  # the emotion responses are presented as a 1D vector
    is_train = T.iscalar('is_train') # pseudo boolean for switching between training and prediction
    
    L1p_ly1 = T.fvector()  # index to a [mini]batch
    L1p_ly2 = T.fvector()
    L1p_ly3 = T.fvector()
    L2p_ly = T.fscalar()
    lrate = T.fscalar()

    [op_tg_L1_ly1, op_tg_L1_ly2, op_tg_L1_ly3]= hsp_level
    [max_beta_ly1, max_beta_ly2, max_beta_ly3] = max_beta

    print ('... optimal HSP!!')
    print ('%1.1f-%1.1f-%1.1f' % (op_tg_L1_ly1,op_tg_L1_ly2,op_tg_L1_ly3))
    
    hsp_ly1 = 0; hsp_ly2 = 0; hsp_ly3 =0;     val_L1_ly1 = 0; val_L1_ly2 = 0; val_L1_ly3=0;
    list_hsp_ly1 = np.zeros((n_epochs,1));     list_hsp_ly2 = np.zeros((n_epochs,1));    list_hsp_ly3 = np.zeros((n_epochs,1))
    list_L1_ly1 = np.zeros((n_epochs,1));     list_L1_ly2 = np.zeros((n_epochs,1));    list_L1_ly3 = np.zeros((n_epochs,1))
    list_tr_err = np.zeros((n_epochs,1));    list_ts_err = np.zeros((n_epochs,1)) 
    lrate_list =np.zeros((n_epochs,1));

    train_set_x = theano.shared(np.asarray(n_train_set_x, dtype=theano.config.floatX))
    train_set_y = T.cast(theano.shared(train_y,borrow=True),'float32')

    test_set_x = theano.shared(np.asarray(n_test_set_x, dtype=theano.config.floatX)); 
    test_set_y = T.cast(theano.shared(test_y,borrow=True),'float32')
    
    lrate_val = itlrate
    
    # construct the MLP class
    
    classifier = MLP(
            rng=rng,                            
            input=x,                            
            n_in= n_in,
            n_hidden1=n_hidden1,                         
            n_hidden2=n_hidden2,                           
            n_hidden3=n_hidden3,
            n_out=n_output,
            corruption_level = corruption_level,
            is_train=is_train,
        )
            
    # cost function
    cost = ((classifier.linearRegressionLayer.y_pred-y)**2).sum()
    cost += (T.dot(abs(classifier.hiddenLayer1.W),L1p_ly1)).sum(); 
    cost += (T.dot(abs(classifier.hiddenLayer2.W),L1p_ly2)).sum();
    cost += (T.dot(abs(classifier.hiddenLayer3.W),L1p_ly3)).sum();
    cost += L2p_ly * classifier.L2_sqr
    
#    gparams = [T.grad(cost, param) for param in classifier.params]
    
    new_gparams =[];                                    
    gparams = [T.grad(cost, param) for param in classifier.params]
    new_gparams = [i/float(batch_size) for i in gparams]
        
    updates = []
    
    for param, gparam, oldparam in zip(classifier.params, new_gparams, classifier.oldparams):
        delta = lrate * gparam + momentum * oldparam
        updates.append((param, param - delta))
        updates.append((oldparam, delta))
                       
    trvld_model = theano.function(
        inputs=[index, L1p_ly1, L1p_ly2, L1p_ly3, L2p_ly, lrate],

        outputs=[classifier.errors(y), classifier.linearRegressionLayer.y_pred],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: np.cast['int32'](1)

        },
        allow_input_downcast = True,
        on_unused_input = 'ignore',
    )

    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), classifier.linearRegressionLayer.y_pred],
        givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size],
                is_train: np.cast['int32'](0)
            },
       on_unused_input='ignore'

    )
    
    list_trvld_err = np.zeros((n_epochs,1)); tst_err = numpy.zeros((n_epochs,1));
    #pct_trvld = np.zeros((n_epochs,n_trvld_batches*batch_size), dtype='float32')
    pct_trvld = np.zeros((n_epochs,n_trvld_batches*batch_size))
    #pct_tst = np.zeros((n_epochs,n_test_batches*batch_size), dtype='float32')
    pct_tst = np.zeros((n_epochs,n_test_batches*batch_size))
    
    hsp_val_ly1 = np.zeros((n_epochs+1,n_hidden1));    hsp_val_ly2 = np.zeros((n_epochs+1,n_hidden2));   hsp_val_ly3 = np.zeros((n_epochs+1,n_hidden3));
    L1_val_ly1 = np.zeros((n_epochs+1,n_hidden1));    L1_val_ly2 = np.zeros((n_epochs+1,n_hidden2));    L1_val_ly3 = np.zeros((n_epochs+1,n_hidden3));
    
    ########################################## Learning model #################################################
   
    print ('... Training & Test')
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        trvld_score = np.zeros((n_trvld_batches,1));
        tmp_trvld_pct =0;
        
        for minibatch_index in range(n_trvld_batches):
            tmp_mat_ly1 = (L1_val_ly1[epoch-1,:]);            tmp_mat_ly2 = (L1_val_ly2[epoch-1,:]);            tmp_mat_ly3 = (L1_val_ly3[epoch-1,:]);
            trvld_out = trvld_model(minibatch_index,tmp_mat_ly1,tmp_mat_ly2,tmp_mat_ly3,val_L2,lrate_val)
            if minibatch_index ==0:
                tmp_trvld_pct = trvld_out[1]
            else:
                tmp_trvld_pct = np.concatenate((tmp_trvld_pct,trvld_out[1]),axis=0)
            
            [hsp_val_ly1[epoch,:], L1_val_ly1[epoch,:]] = hsp_fnc_inv_mat_cal(L1_val_ly1[epoch-1,:],classifier.hiddenLayer1.W,max_beta_ly1,op_tg_L1_ly1,beta_lrates)
            [hsp_val_ly2[epoch,:], L1_val_ly2[epoch,:]] = hsp_fnc_inv_mat_cal(L1_val_ly2[epoch-1,:],classifier.hiddenLayer2.W,max_beta_ly2,op_tg_L1_ly2,beta_lrates)
            [hsp_val_ly3[epoch,:], L1_val_ly3[epoch,:]] = hsp_fnc_inv_mat_cal(L1_val_ly3[epoch-1,:],classifier.hiddenLayer3.W,max_beta_ly3,op_tg_L1_ly3,beta_lrates)
            
        trvld_score=0;
        trvld_score = (np.mean(abs(tmp_trvld_pct-train_y[numpy.arange(0,len(tmp_trvld_pct))])))
        list_trvld_err[epoch-1] = trvld_score * scal_ref
        pct_trvld[epoch-1][:] = tmp_trvld_pct
        
        tmp_test_pct=0;
        for i in range(n_test_batches):
            test_out = test_model(i)
            if i ==0:
                tmp_test_pct = test_out[1]
            else:
                tmp_test_pct = np.concatenate((tmp_test_pct,test_out[1]),axis=0)
                
        test_score = 0;
        test_score = (np.mean(abs(tmp_test_pct-test_y[numpy.arange(0,len(tmp_test_pct))])))
        tst_err[epoch-1] = test_score * scal_ref
        pct_tst[epoch-1][:] = tmp_test_pct
                           
        lrate_val *= dcay_rate    
        lrate_list[epoch-1] = lrate_val                        
                
        print('#######')         
        print('CP %.2f inv_hsp-lrate %6f, test epoch %i/%i, minibatch %i/%i, tr_err %f, test_err %f' %
            (corruption_level, lrate_list[epoch-1],epoch,n_epochs, minibatch_index+1, n_trvld_batches,trvld_score * scal_ref, test_score * scal_ref))
        print (("hsp_ly1= %.3f/%.3f, L1p_ly1= %.3f, hsp_ly2= %.3f/%.3f, L1p_ly2= %.3f, hsp_ly3= %.3f/%.3f, L1p_ly3= %.3f ")
               % (np.mean(hsp_val_ly1[epoch-1,:]),op_tg_L1_ly1,np.mean(L1_val_ly1[epoch-1,:]),
                  np.mean(hsp_val_ly2[epoch-1,:]),op_tg_L1_ly2,np.mean(L1_val_ly2[epoch-1,:]),
                  np.mean(hsp_val_ly3[epoch-1,:]),op_tg_L1_ly3,np.mean(L1_val_ly3[epoch-1,:])))           
        
        list_ts_err[epoch-1] = test_score * scal_ref
        
    ########################################## Save variables #################################################
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    end_time = timeit.default_timer()
    cst_time = (end_time - start_time) / 60.
        
    sio.savemat(save_name, {'w1': classifier.hiddenLayer1.W.get_value(borrow=True),'b1': classifier.hiddenLayer1.b.get_value(borrow=True),
                       'w2': classifier.hiddenLayer2.W.get_value(borrow=True),'b2': classifier.hiddenLayer2.b.get_value(borrow=True),
                       'w3': classifier.hiddenLayer3.W.get_value(borrow=True),'b3': classifier.hiddenLayer3.b.get_value(borrow=True),
                       'w4': classifier.linearRegressionLayer.W.get_value(borrow=True),'b4': classifier.linearRegressionLayer.b.get_value(borrow=True),
                       'pct_trvld':pct_trvld,'pct_tst':pct_tst,'trvld_err':list_trvld_err,'ts_err':list_ts_err,'L2_val':val_L2,
                       'l1ly1':L1_val_ly1,'l1ly2':L1_val_ly2,'l1ly3':L1_val_ly3,'hsply1':hsp_val_ly1,'hsply2':hsp_val_ly2,'hsply3':hsp_val_ly3,
                       'l_rate':lrate_list,'cst_time':cst_time,'epch':epoch,'max_beta':max_beta,'beta_lrates':beta_lrates,
                        'test_y':test_y,'train_y':train_y,'mtum':momentum,'btch_size':batch_size,'opt_hsp':hsp_level,'cp_lev':corruption_level})
    print ('...done!')

if __name__ == '__main__':
    test_mlp()
    

