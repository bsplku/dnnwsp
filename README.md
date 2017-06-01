# Deep neural netowrk (DNN) with weight sparsity contorl 

> Deep neural network (DNN) has been recently gained growing interest in neuroimaging data analysis. The challenge is, however, to address the so-called, ‘curse-of-dimensionality’ issue when the DNN is applied to neuroimaging data, in which an overfitting can be occurred when the limited number of samples and the high dimensional neuroimaging data are used to train the DNN. To tackle this issue and to increase the utility of DNN to the neuroimaging data analysis, an explicit control of DNN weights has been proposed and its efficacy has been shown for classification tasks using whole-brain functional connectivity and activation data [1, 2]. 

In this toobox, we provide Python-based software toolboxes of the method utilizing either Theano or TensorFlow library. 

# Sample data
> We provide fMRI data as the sample data (bspl.korea.ac.kr/lhrhadvs_sample_data.mat) that were acquired during the four sensorimotor tasks inclduing lef-thand clecnhing , right-hand clecnhing, auditory attention, and visual stimulus tasks [2]. 

# Requestment 
> Python3.6 with libraries including the Theano and/or TensorFlow. 

## References 
> [1] Jang, H., Plis, S.M., Calhoun, V.D. and Lee, J.H., 2017. Task-specific feature extraction and classification of fMRI volumes using a deep neural network initialized with a deep belief network: Evaluation using sensorimotor tasks. Neuroimage, 145, pp.314-328. 

> [2] Kim, J., Calhoun, V.D., Shim, E. and Lee, J.H., 2016. Deep neural network with weight sparsity control and pre-training extracts hierarchical features and enhances classification performance: Evidence from whole-brain resting-state functional connectivity patterns of schizophrenia. NeuroImage, 124, pp.127-146.
