# Deep neural network (DNN) with weight sparsity control 

* DNN has been recently gained growing interest in neuroimaging data analysis. The challenge is, however, to address the so-called, ‘curse-of-dimensionality’ issue when the DNN is applied to neuroimaging data, in which an overfitting can be occurred when the limited number of samples and the high dimensional neuroimaging data are used to train the DNN. To tackle this issue and to increase the utility of DNN to the neuroimaging data analysis, an explicit control of DNN weights has been proposed and its efficacy has been shown for classification tasks using whole-brain functional connectivity and activation data [1, 2]. 

* In these Python-based software toolboxes using either Theano or TensorFlow library, we implemented our DNN method so that novice users to the DNN can try our model using our sample data and finally to adopt the method to their dataset. 

# Sample data
* We prepared sample [fMRI data](http://bspl.korea.ac.kr/lhrhadvs_sample_data.mat) that were acquired during the four sensorimotor tasks including left-thand clenching, right-hand clenching, auditory attention, and visual stimulus tasks [1]. 

# Prerequisite
* Python3.6 with libraries including the Theano and/or TensorFlow. 

## References 
* [1] Jang, H., Plis, S.M., Calhoun, V.D. and Lee, J.H., 2017. Task-specific feature extraction and classification of fMRI volumes using a deep neural network initialized with a deep belief network: Evaluation using sensorimotor tasks. Neuroimage, 145, pp.314-328. 

* [2] Kim, J., Calhoun, V.D., Shim, E. and Lee, J.H., 2016. Deep neural network with weight sparsity control and pre-training extracts hierarchical features and enhances classification performance: Evidence from whole-brain resting-state functional connectivity patterns of schizophrenia. NeuroImage, 124, pp.127-146.
