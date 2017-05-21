import numpy as np
from numpy import linalg as LA

def hsp_fnc(betaval_L1, W, max_beta, tg_hsp, beta_lrate):
    
    W = W.get_value(borrow=True)
    Wvec = W.flatten();
    sqrt_nsamps = pow((Wvec.shape[0]),0.5)
    
    n1_W = LA.norm(Wvec,1);    n2_W = LA.norm(Wvec,2);
    hspvalue = (sqrt_nsamps-(n1_W/n2_W))/(sqrt_nsamps-1);

    betaval_L1 -= beta_lrate*np.sign(hspvalue-tg_hsp) 

    betaval_L1 = 0 if betaval_L1<0 else betaval_L1
    betaval_L1 = max_beta if betaval_L1>max_beta else betaval_L1

    return [hspvalue, betaval_L1]