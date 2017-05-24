import numpy as np
from numpy import linalg as LA

def hsp_fnc_inv_mat_cal(beta_val_L1, W, max_beta, tg_hsp, beta_lrate):

    W = np.array(W.get_value(borrow=True));
    [dim, nodes] = W.shape
    
    cnt_L1_ly = beta_val_L1;
    
    hsp_vec = np.zeros((1,nodes));  
    
    tg_hsp_vec = np.ones(nodes)*tg_hsp;
    sqrt_nsamps = pow(dim,0.5)
        
    n1_W = LA.norm(W,1,axis=0);    n2_W = LA.norm(W,2,axis=0);
    hsp_vec = (sqrt_nsamps - (n1_W/n2_W))/(sqrt_nsamps-1)
    
    cnt_L1_ly -= beta_lrate*np.sign(hsp_vec-tg_hsp_vec)

    for ii in xrange(0,nodes):
        if cnt_L1_ly[ii] < 0:
            cnt_L1_ly[ii] = 0
        if cnt_L1_ly[ii] > max_beta:
            cnt_L1_ly[ii] = max_beta
        if cnt_L1_ly[ii] < (beta_lrate*1e-1):
            cnt_L1_ly[ii] = 0
    
    return [hsp_vec, cnt_L1_ly]


    
    
 
