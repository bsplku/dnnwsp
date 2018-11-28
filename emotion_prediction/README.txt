"""
Deep neural network (DNN) with weight sparsity control (DNN-WSP; i.e., L1-norm regularization) improved 
the classification performance using whole-brain resting-state functional connectivity patterns of 
schizophrenia patient and healthy groups. Also, initializing DNN's weights through stacked auto-encoder enhanced 
the classification performance (Jang et al., Neuroimage 2017, Kim et al., Neuroimage, 2016). 

The DNN code is for regression anlaysis to predict emotional responses (i.e., scores) from whole-brain fMRI data. 
The Python codes were modified from the DeepLearningTutorials (https://github.com/lisa-lab/DeepLearningTutorials) 
to apply a node-wise and layer-wise control of weight sparsity via Hoyer sparseness (Kim and Lee, PRNI2016 & ICASSP2017).

dnnwsp_hsp_denoise.py: a code for a DNN model with weight sparsity control 

dnnwsp_reg_h3_wt_denoising.ipynb: detailed information about the code for a DNN model with weight sparsity control 

dnnwsp_rst_check.ipynb: a code to investigate the results obtained from the DNN with weight sparsity control

emt_valence_sample.mat: sample data
############# emt_sample_data.mat #############
# train_x  = 64 volumes x 55417 voxels  
# train_y  = 64 volumes x 1 [valence, arousal or dominance scores for training]
# test_x  = 16 volumes x 55417 voxels
# test_y  = 16 volumes x 1 [valence, arousal or dominance scores for test]
###############################################

rst_vlnc_predcition.mat: results obtained from the DNN with weight sparsity control
