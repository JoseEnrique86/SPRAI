from keras.models import load_model
import os
import numpy as np
import common as cm
import tensorflow as tf
from tensorflow.python.ops import math_ops

################################################################################################
#                       CONVERSION TO DIMENSIONLESS VALUES                                     #
################################################################################################

def convert_dim(si,cof):
    
    a = si.shape[0]
    b = si.shape[1]
    s = tuple((a,b))
    out = np.zeros(s)
    
    if cof == 1:
        cm.scale1 = si[:,0] # store the scale equal to lenght
    
    if cof == 2:
        cm.scale2 = si[:,0] # store the scale equal to lenght


    # inputs: L, B, T, Cb, Cf, Cm, Cc, Xb, Zb, GMT, GML
    out[:,0] = 1.0
    out[:,1] = si[:,1]/si[:,0] # dimensionless B/L
    out[:,2] = si[:,2]/si[:,0] # dimensionless T/L
    out[:,3] = si[:,3]
    out[:,4] = si[:,4]
    out[:,5] = si[:,5]
    out[:,6] = si[:,6]
    out[:,7] = si[:,7]/si[:,0] # dimensionless XB/L
    out[:,8] = si[:,8]/si[:,2] # dimensionless ZB/T
    out[:,9] = si[:,9]
    out[:,10]= si[:,10]
    out[:,10:] = si[:,10:]
    
    return out
  
################################################################################################
#                               PREDICITION FUNCTIONS                                          #
################################################################################################

def MNRE_1(y_true, y_pred):
    bool_idx=tf.keras.backend.greater(abs(y_true),1.0)
    loss1 = abs(y_pred - y_true)/(1.0)
    loss2 = abs(y_pred - y_true)/(abs(y_true))
    error=tf.keras.backend.switch(bool_idx,loss2,loss1)
    loss = math_ops.square(error)
    loss = tf.keras.backend.mean(loss)
    loss = tf.keras.backend.sqrt(loss)

    return loss

## MODELS FOR FN EQUAL TO ZERO
def load_models_seakeeping1(path_ANN):
    
    models1 = []

    # list of h5 files
    # ANN for Fn equal to zero
    model_name1= ['AM.h5','D.h5', 'FeX_Cos_0-60.h5','FeX_Cos_90.h5', 'FeX_Cos_120-180.h5', 'FeX_Sin_0-60.h5', 'FeX_Sin_90.h5', 'FeX_Sin_120-180.h5', 
     'FeY_Cos_30-150.h5', 'FeY_Sin_30-150.h5', 'FeZ_Cos.h5', 'FeZ_Sin.h5', 'MeX_Cos_30-150.h5', 'MeX_Sin_30-150.h5', 
     'MeY_Cos.h5', 'MeY_Sin.h5', 'MeZ_Cos_30-150.h5', 'MeZ_Sin_30-150.h5']
    
    for m in model_name1:
        ANN = load_model(os.path.join(str(path_ANN),str(m)), compile=False)
        models1.append(ANN)

    return models1

## MODELS FOR FN GREATER THAN ZERO
def load_models_seakeeping2(path_ANN):
    
    models2 = []

    # list of h5 files
    # ANN for Fn greater than zero
    model_name2 = ['AM_11-13-31.h5', 'AM_8dof.h5', 'AM15.h5', 'AM26.h5', 'AM35.h5', 'AM46.h5', 'AM51.h5', 'AM53.h5', 'AM55.h5', 'AM62.h5', 'AM64.h5', 'AM66.h5',
                   'D_8dof.h5', 'D15.h5', 'D26.h5', 'D35.h5', 'D46.h5', 'D51.h5', 'D53.h5', 'D55.h5', 'D62.h5', 'D64.h5', 'D66.h5',
                   'FeX_Din_Cos_0-60.h5', 'FeX_Din_Sin_0-60.h5', 'FeX_Din_Cos_90.h5', 'FeX_Din_Sin_90.h5', 'FeX_Din_Cos_120-180.h5', 'FeX_Din_Sin_120-180.h5',
                   'FeY_Din_Cos_30-150.h5', 'FeY_Din_Sin_30-150.h5', 'FeZ_Din_Cos.h5', 'FeZ_Din_Sin.h5', 
                   'MeX_Din_Cos_30-60.h5', 'MeX_Din_Sin_30-60.h5', 'MeX_Din_Cos_90.h5', 'MeX_Din_Sin_90.h5', 'MeX_Din_Cos_120-150.h5', 'MeX_Din_Sin_120-150.h5',
                   'MeY_Din_Cos_0-60.h5', 'MeY_Din_Sin_0-60.h5', 'MeY_Din_Cos_90.h5', 'MeY_Din_Sin_90.h5',
                   'MeY_Din_Cos_120-180.h5', 'MeY_Din_Sin_120-180.h5', 'MeZ_Din_Cos_30-150.h5', 'MeZ_Din_Sin_30-150.h5']
    
    for m in model_name2:
        ANN = load_model(os.path.join(str(path_ANN),str(m)), compile=False)
        models2.append(ANN)
    
    return models2

# PREDICTION FOR FN EQUAL TO ZERO
def prediction_seakeeping1(inputs,models):
    # si = ships inputs from csv
    # models = pretrained RNN 
    # inputs: L, B, T, Cb, Cf, Cm, Cc, Xb, Zb, GMT, GML, Fn, beta

    frequencies = cm.sea_var.frequencies
    nfreqs = len(frequencies)

    ns = np.shape(inputs)
    nships = ns[0]

    #s=tuple((30,9))
    s=tuple((nfreqs,9))#!!!!!!!!!!!!!!!!!!
    si = np.zeros(s)

    for i in range(nships):

        # B,T,CB,CF,CM,CC,XB,ZB,fn,beta
        a = inputs[i,[1,2,3,4,5,6,7,8]] ## dummy variable
        
        si[:,0]=a[0]; si[:,1]=a[1]; si[:,2]=a[2]; si[:,3]=a[3]
        si[:,4]=a[4]; si[:,5]=a[5]; si[:,6]=a[6]; si[:,7]=a[7]

        for j in range(nfreqs):
            ## append the wave  frequencies
            si[j,8]=frequencies[j]
            
        ## prediction added masses
        am_i = models[0].predict(si)
        ## prediction damping
        d_i = models[1].predict(si)
        ## prediction forces X direction
        fxc_0_60_i = models[2].predict(si); fxc_90_i = models[3].predict(si); fxc_120_180_i = models[4].predict(si)
        fxs_0_60_i = models[5].predict(si); fxs_90_i = models[6].predict(si); fxs_120_180_i = models[7].predict(si)
        ## prediction forces Y direction
        fyc_30_150_i = models[8].predict(si); fys_30_150_i =  models[9].predict(si)
        ## prediction forces Z direction
        fzc_i = models[10].predict(si); fzs_i = models[11].predict(si)
        ## prediction moments X axis
        mxc_30_150_i = models[12].predict(si); mxs_30_150_i = models[13].predict(si)
        ## prediction moments Y axis
        myc_i = models[14].predict(si); mys_i = models[15].predict(si)
        ## prediction moments Z axis
        mzc_30_150_i = models[16].predict(si); mzs_30_150_i = models[17].predict(si)

        ## storing values of predictions for each ship
        cm.Added_M[:,:,i] = am_i
        cm.Damp[:,:,i] = d_i

        ## Forces X direction
        cm.FEX_cos0_60[:,:,i] = fxc_0_60_i; cm.FEX_cos90[:,:,i] = fxc_90_i; cm.FEX_cos120_180[:,:,i] = fxc_120_180_i
        cm.FEX_sin0_60[:,:,i] = fxs_0_60_i; cm.FEX_sin90[:,:,i] = fxs_90_i; cm.FEX_sin120_180[:,:,i] = fxs_120_180_i
        ## Phases of forces X direction
        cm.Phase_FEX0_60[:,:,i] = np.arctan(fxs_0_60_i/fxc_0_60_i)
        cm.Phase_FEX90[:,:,i] = np.arctan(fxs_90_i/fxc_90_i)
        cm.Phase_FEX120_180[:,:,i] = np.arctan(fxs_120_180_i/fxc_120_180_i)
        ## Forces Y direction
        cm.FEY_sin30_150[:,:,i] = fys_30_150_i; cm.FEY_cos30_150[:,:,i] = fyc_30_150_i
        ## Phases of forces Y direction
        cm.Phase_FEY30_150[:,:,i] = np.arctan(fys_30_150_i/fyc_30_150_i)
        ## Forces Z direction
        cm.FEZ_sin[:,:,i] = fzs_i; cm.FEZ_cos[:,:,i] = fzc_i
        ## Phases of forces Z direction
        cm.Phase_FEZ[:,:,i] = np.arctan(fzs_i/fzc_i)
        ## Moments around X axis
        cm.MEX_sin30_150[:,:,i] = mxs_30_150_i; cm.MEX_cos30_150[:,:,i] = mxc_30_150_i
        ## Phases of moments in X axis
        cm.Phase_MEX30_150[:,:,i] = np.arctan(mxs_30_150_i/mxc_30_150_i)
        ## Moments around Y axis
        cm.MEY_sin[:,:,i] = mys_i; cm.MEY_cos[:,:,i] = myc_i
        ## Phases of moments in Y axis
        cm.Phase_MEY[:,:,i] = np.arctan(mys_i/myc_i)
        ## Moments around Z axis
        cm.MEZ_sin30_150[:,:,i] = mzs_30_150_i; cm.MEZ_cos30_150[:,:,i] = mzc_30_150_i
        ## Phases of moments in Z axis
        cm.Phase_MEZ30_150[:,:,i] = np.arctan(mzs_30_150_i/mzc_30_150_i)

# PREDICTION FOR FN GREATER THAN ZERO
def prediction_seakeeping2(inputs,models):
    # si = ships inputs from csv
    # m = pretrained RNN (tuple)
    # inputs: L, B, T, Cb, Cf, Cm, Cc, Xb, Zb, GMT, GML, Fn, beta

    frequencies2 = cm.sea_var.frequencies2
    nfreqs = len(frequencies2)
    grav = cm.sea_var.grav
    angles = cm.sea_var.angles
    nheads = len(angles)

    ns =  np.shape(inputs)
    nships = ns[0]

    st1 = tuple((nfreqs,9))
    st2 = tuple((nfreqs,8))
    st3 = tuple((nfreqs,10))

    # inputs for fn!=0 (8 dof AM & D)
    si8dof = np.zeros(st1)
    # inputs for fn!=0 (AM11, AM13 & AM31)
    si2 = np.zeros(st2)
    # inputs for fn!=0 (AM55, AM15, AM35, AM51, AM53, B55, B15, B51, B35, B53)
    si3 = np.zeros(st1)
    # inputs for fn!=0 (AM66, AM62, B66, B26, B62)
    si4 = np.zeros(st1)
    # inputs for fn!=0 (AM46, AM64, B46, B64)
    si5 = np.zeros(st3)
    # inputs for fn!=0 (FeX, FeY, FeZ, MeX, MeY, MeZ sin & cos)
    si6 = np.zeros(st3)

    for i in range(nships):
        # 0 1 2  3  4  5  6  7  8
        # B,T,CB,CF,CM,CC,XB,ZB,fn,beta
        a = inputs[i,[1,2,3,4,5,6,7,8,14,15]] ## dummy variable
        
        fn = a[8] # Froude number
        v = fn*np.sqrt(grav) # ship velocity (m/s)

        ########## INPUTS FOR PREDICTIONS ######################3
        # B,T,CB,CF,CM,CC,XB,ZB,W_e
        si8dof[:,0]=a[0]; si8dof[:,1]=a[1]; si8dof[:,2]=a[2]; si8dof[:,3]=a[3]
        si8dof[:,4]=a[4]; si8dof[:,5]=a[5]; si8dof[:,6]=a[6]; si8dof[:,7]=a[7]
        # B,T,CB,CF,CM,XB,ZB,W_e
        si2[:,0]=a[0]; si2[:,1]=a[1]; si2[:,2]=a[2]; si2[:,3]=a[3]
        si2[:,4]=a[4]; si2[:,5]=a[6]; si2[:,6]=a[7]
        # B,T,CB,CF,CM,XB,ZB,Fn,W_e
        si3[:,0]=a[0]; si3[:,1]=a[1]; si3[:,2]=a[2]; si3[:,3]=a[3]
        si3[:,4]=a[4]; si3[:,5]=a[6]; si3[:,6]=a[7]; si3[:,7]=a[8]
        # B,T,CB,CM,CC,XB,ZB,Fn,W_e
        si4[:,0]=a[0]; si4[:,1]=a[1]; si4[:,2]=a[2]; si4[:,3]=a[4]
        si4[:,4]=a[5]; si4[:,5]=a[6]; si4[:,6]=a[7]; si4[:,7]=a[8]
        # B,T,CB,CF,CM,CC,XB,ZB,Fn,W_e.
        si5[:,0]=a[0]; si5[:,1]=a[1]; si5[:,2]=a[2]; si5[:,3]=a[3]
        si5[:,4]=a[4]; si5[:,5]=a[5]; si5[:,6]=a[6]; si5[:,7]=a[7]; si5[:,8]=a[8]
        # B,T,CB,CF,CM,CC,XB,ZB,Fn,W_ola.
        si6[:,0]=a[0]; si6[:,1]=a[1]; si6[:,2]=a[2]; si6[:,3]=a[3]
        si6[:,4]=a[4]; si6[:,5]=a[5]; si6[:,6]=a[6]; si6[:,7]=a[7]; si6[:,8]=a[8]
            
        for j in range(nheads):
            # counterwise
            #                /
            #               / beta
            #              /
            #  ___________/_______
            #  |         /        \______________\ 0 deg
            #  |__________________/              /
            #  
            #beta = angles[j] - a[9]
            beta = angles[j]

            for t in range(nfreqs):
            ## append the encounter frequencies
                wencount = frequencies2[t]*(1-frequencies2[t]*(v/grav)*np.cos(beta*2*np.pi/360))
                if wencount < 1.0:
                    we = 9999.9
                else:
                    we = wencount
                
                si8dof[t,8]=we; si2[t,7]=we; si3[t,8]=we; si4[t,8]=we; si5[t,9]=we; si6[t,9]=frequencies2[t]

            # prediction of added masses
            am11_13_31 = models[0].predict(si2); am_8dof = models[1].predict(si8dof); am15 = models[2].predict(si3)
            am26 = models[3].predict(si4);       am35 = models[4].predict(si3);       am46 = models[5].predict(si5)
            am51 = models[6].predict(si3);       am53 = models[7].predict(si3);       am55 = models[8].predict(si3)
            am62 = models[9].predict(si4);       am64 = models[10].predict(si5);      am66 = models[11].predict(si4)

            # prediction of damping
            d8dof = models[12].predict(si8dof); d15 = models[13].predict(si3); d26 = models[14].predict(si4)
            d35 = models[15].predict(si3);      d46 = models[16].predict(si5); d51 = models[17].predict(si3)
            d53 = models[18].predict(si3);      d55 = models[19].predict(si3); d62 = models[20].predict(si4)
            d64 = models[21].predict(si5);      d66 = models[22].predict(si4)

            ## storing values of predictions for each ship
            ## added masses f = freqs;  j =heads; k= addedmass; i= ships
            # am11                                       am13                                         am15
            cm.Added_M_0_180[:,j,0,i] = am11_13_31[:,0]; cm.Added_M_0_180[:,j,1,i] = am11_13_31[:,1]; cm.Added_M_0_180[:,j,2,i] = am15[:,0]
            # am22                                       am24                                         am26 
            cm.Added_M_0_180[:,j,3,i] = am_8dof[:,2];    cm.Added_M_0_180[:,j,4,i] = am_8dof[:,3];    cm.Added_M_0_180[:,j,5,i] = am26[:,0]
            # am31                                       am33                                         am35
            cm.Added_M_0_180[:,j,6,i] = am11_13_31[:,2]; cm.Added_M_0_180[:,j,7,i] = am_8dof[:,5];    cm.Added_M_0_180[:,j,8,i] = am35[:,0]
            # am42                                       am44                                         am46
            cm.Added_M_0_180[:,j,9,i] = am_8dof[:,6];    cm.Added_M_0_180[:,j,10,i] = am_8dof[:,7];   cm.Added_M_0_180[:,j,11,i] = am46[:,0]
            # am51                                       am53                                         am55
            cm.Added_M_0_180[:,j,12,i] = am51[:,0];      cm.Added_M_0_180[:,j,13,i] = am53[:,0];      cm.Added_M_0_180[:,j,14,i] = am55[:,0]
            # am62                                       am64                                         am66
            cm.Added_M_0_180[:,j,15,i] = am62[:,0];      cm.Added_M_0_180[:,j,16,i] = am64[:,0];      cm.Added_M_0_180[:,j,17,i] = am66[:,0]
            
            ## damping
            # b11                                 b13                                   b15
            cm.Damp_0_180[:,j,0,i] = d8dof[:,0];  cm.Damp_0_180[:,j,1,i] = d8dof[:,1];  cm.Damp_0_180[:,j,2,i] = d15[:,0]
            # b22                                 b24                                   b26
            cm.Damp_0_180[:,j,3,i] = d8dof[:,2];  cm.Damp_0_180[:,j,4,i] = d8dof[:,3];  cm.Damp_0_180[:,j,5,i] = d26[:,0]
            # b31                                 b33                                   b35
            cm.Damp_0_180[:,j,6,i] = d8dof[:,4];  cm.Damp_0_180[:,j,7,i] = d8dof[:,5];  cm.Damp_0_180[:,j,8,i] = d35[:,0]
            # b42                                 b44                                   b46
            cm.Damp_0_180[:,j,9,i] = d8dof[:,6];  cm.Damp_0_180[:,j,10,i] = d8dof[:,7]; cm.Damp_0_180[:,j,11,i] = d46[:,0]
            # b51                                 b53                                   b55
            cm.Damp_0_180[:,j,12,i] = d51[:,0];   cm.Damp_0_180[:,j,13,i] = d53[:,0];   cm.Damp_0_180[:,j,14,i] = d55[:,0]
            # b62                                 b64                                   b66
            cm.Damp_0_180[:,j,15,i] = d62[:,0];   cm.Damp_0_180[:,j,16,i] = d64[:,0];   cm.Damp_0_180[:,j,17,i] = d66[:,0]

        #for j in range(len(frequencies2)):
        #    si6[j,9]=frequencies2[j]

        # prediction of forces Fx
        fex_din_c_0_60 = models[23].predict(si6);    fex_din_s_0_60 = models[24].predict(si6)
        fex_din_c_90 = models[25].predict(si6);      fex_din_s_90 = models[26].predict(si6)
        fex_din_c_120_180 = models[27].predict(si6); fex_din_s_120_180 = models[28].predict(si6)
        
        # prediction of forces Fy
        fey_din_c_30_150 = models[29].predict(si6); fey_din_s_30_150 = models[30].predict(si6)
        
        # prediction of forces Fz
        fez_din_c = models[31].predict(si6); fez_din_s = models[32].predict(si6)
       
        # prediction of moments Mx
        mex_din_c_30_60 = models[33].predict(si6);   mex_din_s_30_60 = models[34].predict(si6)
        mex_din_c_90 = models[35].predict(si6);      mex_din_s_90 = models[36].predict(si6)
        mex_din_c_120_150 = models[37].predict(si6); mex_din_s_120_150 = models[38].predict(si6)
        
        # prediction of moments My
        mey_din_c_0_60 = models[39].predict(si6);    mey_din_s_0_60 = models[40].predict(si6)
        mey_din_c_90 = models[41].predict(si6);      mey_din_s_90 = models[42].predict(si6)
        mey_din_c_120_180 = models[43].predict(si6); mey_din_s_120_180 = models[44].predict(si6)
        
        # prediction of moments Mz
        mez_din_c_30_150 = models[45].predict(si6); mez_din_s_30_150 = models[46].predict(si6)
        
        ## forces
        ## Forces X direction
        cm.FEX_cos0_60_din[:,:,i] = fex_din_c_0_60; cm.FEX_cos90_din[:,:,i] = fex_din_c_90; cm.FEX_cos120_180_din[:,:,i] = fex_din_c_120_180
        cm.FEX_sin0_60_din[:,:,i] = fex_din_s_0_60; cm.FEX_sin90_din[:,:,i] = fex_din_s_90; cm.FEX_sin120_180_din[:,:,i] = fex_din_s_120_180
        ## Phases of forces X direction
        cm.Phase_FEX0_60_din[:,:,i] = np.arctan(fex_din_s_0_60/fex_din_c_0_60)
        cm.Phase_FEX90_din[:,:,i] = np.arctan(fex_din_s_90/fex_din_c_90)
        cm.Phase_FEX120_180_din[:,:,i] = np.arctan(fex_din_s_120_180/fex_din_c_120_180)

        ## Forces Y direction
        cm.FEY_sin30_150_din[:,:,i] = fey_din_s_30_150; cm.FEY_cos30_150_din[:,:,i] = fey_din_c_30_150
        ## Phases of forces Y direction
        cm.Phase_FEY30_150_din[:,:,i] = np.arctan(fey_din_s_30_150/fey_din_c_30_150)

        ## Forces Z direction
        cm.FEZ_sin_din[:,:,i] = fez_din_s; cm.FEZ_cos_din[:,:,i] = fez_din_c
        ## Phases of forces Z direction
        cm.Phase_FEZ_din[:,:,i] = np.arctan(fez_din_s/fez_din_c)

        ## Moments around X axis
        cm.MEX_sin30_150_din[:,0,i] = mex_din_s_30_60[:,0];   cm.MEX_cos30_150_din[:,0,i] = mex_din_c_30_60[:,0]
        cm.MEX_sin30_150_din[:,1,i] = mex_din_s_30_60[:,1];   cm.MEX_cos30_150_din[:,1,i] = mex_din_c_30_60[:,1]
        cm.MEX_sin30_150_din[:,2,i] = mex_din_s_90[:,0];      cm.MEX_cos30_150_din[:,2,i] = mex_din_c_90[:,0]
        cm.MEX_sin30_150_din[:,3,i] = mex_din_s_120_150[:,0]; cm.MEX_cos30_150_din[:,3,i] = mex_din_c_120_150[:,0]
        cm.MEX_sin30_150_din[:,4,i] = mex_din_s_120_150[:,1]; cm.MEX_cos30_150_din[:,4,i] = mex_din_c_120_150[:,1]
        ## Phases of moments in X axis
        cm.Phase_MEX30_150_din[:,0,i] = np.arctan( mex_din_s_30_60[:,0]/ mex_din_c_30_60[:,0])
        cm.Phase_MEX30_150_din[:,1,i] = np.arctan( mex_din_s_30_60[:,1]/ mex_din_c_30_60[:,1])
        cm.Phase_MEX30_150_din[:,2,i] = np.arctan( mex_din_s_90[:,0]/ mex_din_c_90[:,0])
        cm.Phase_MEX30_150_din[:,3,i] = np.arctan( mex_din_s_120_150[:,0]/ mex_din_c_120_150[:,0])
        cm.Phase_MEX30_150_din[:,4,i] = np.arctan( mex_din_s_120_150[:,1]/ mex_din_c_120_150[:,1])

        ## Moments around Y axis
        cm.MEY_sin_din[:,0,i] = mey_din_s_0_60[:,0];    cm.MEY_cos_din[:,0,i] = mey_din_c_0_60[:,0]
        cm.MEY_sin_din[:,1,i] = mey_din_s_0_60[:,1];    cm.MEY_cos_din[:,1,i] = mey_din_c_0_60[:,1]
        cm.MEY_sin_din[:,2,i] = mey_din_s_0_60[:,2];    cm.MEY_cos_din[:,2,i] = mey_din_c_0_60[:,2]
        cm.MEY_sin_din[:,3,i] = mey_din_s_90[:,0];      cm.MEY_cos_din[:,3,i] = mey_din_c_90[:,0]
        cm.MEY_sin_din[:,4,i] = mey_din_s_120_180[:,0]; cm.MEY_cos_din[:,4,i] = mey_din_c_120_180[:,0]
        cm.MEY_sin_din[:,5,i] = mey_din_s_120_180[:,1]; cm.MEY_cos_din[:,5,i] = mey_din_c_120_180[:,1]
        cm.MEY_sin_din[:,6,i] = mey_din_s_120_180[:,2]; cm.MEY_cos_din[:,6,i] = mey_din_c_120_180[:,2]
        ## Phases of moments in Y axis
        cm.Phase_MEY_din[:,0,i] = np.arctan(mey_din_s_0_60[:,0]/mey_din_c_0_60[:,0])
        cm.Phase_MEY_din[:,1,i] = np.arctan(mey_din_s_0_60[:,1]/mey_din_c_0_60[:,1])
        cm.Phase_MEY_din[:,2,i] = np.arctan(mey_din_s_0_60[:,2]/mey_din_c_0_60[:,2])
        cm.Phase_MEY_din[:,3,i] = np.arctan(mey_din_s_90[:,0]/mey_din_c_90[:,0])
        cm.Phase_MEY_din[:,4,i] = np.arctan(mey_din_s_120_180[:,0]/mey_din_c_120_180[:,0])
        cm.Phase_MEY_din[:,5,i] = np.arctan(mey_din_s_120_180[:,1]/mey_din_c_120_180[:,1])
        cm.Phase_MEY_din[:,6,i] = np.arctan(mey_din_s_120_180[:,2]/mey_din_c_120_180[:,2])

        ## Moments around X axis
        cm.MEZ_sin30_150_din[:,:,i] = mez_din_s_30_150; cm.MEZ_cos30_150_din[:,:,i] = mez_din_c_30_150

        #cm.MEZ_sin30_150_din[:,0,i] = mez_din_s_30_150[:,0]; cm.MEZ_cos30_150_din[:,0,i] = mez_din_c_30_150[:,0]
        #cm.MEZ_sin30_150_din[:,1,i] = mez_din_s_30_150[:,1]; cm.MEZ_cos30_150_din[:,1,i] = mez_din_c_30_150[:,1]
        #cm.MEZ_sin30_150_din[:,2,i] = mez_din_s_30_150[:,0]; cm.MEZ_cos30_150_din[:,2,i] = mez_din_c_30_150[:,0]
        #cm.MEZ_sin30_150_din[:,3,i] = mez_din_s_30_150[:,0]; cm.MEZ_cos30_150_din[:,3,i] = mez_din_c_30_150[:,0]
        #cm.MEZ_sin30_150_din[:,4,i] = mez_din_s_30_150[:,1]; cm.MEZ_cos30_150_din[:,4,i] = mez_din_c_30_150[:,1]

        ## Phases of moments in Z axis
        cm.Phase_MEZ30_150_din[:,:,i] = np.arctan( mez_din_s_30_150/ mez_din_c_30_150)

        #cm.Phase_MEZ30_150_din[:,0,i] = np.arctan( mez_din_s_30_150[:,0]/ mez_din_c_30_150[:,0])
        #cm.Phase_MEZ30_150_din[:,1,i] = np.arctan( mez_din_s_30_150[:,1]/ mez_din_c_30_150[:,1])
        #cm.Phase_MEZ30_150_din[:,2,i] = np.arctan( mez_din_s_30_150[:,2]/ mez_din_c_30_150[:,2])
        #cm.Phase_MEZ30_150_din[:,3,i] = np.arctan( mez_din_s_30_150[:,3]/ mez_din_c_30_150[:,4])
        #cm.Phase_MEZ30_150_din[:,4,i] = np.arctan( mez_din_s_30_150[:,4]/ mez_din_c_30_150[:,4])

################################################################################################
#                              FULL SCALE HYDRODYNAMIC PARAMETERS                              #
################################################################################################

# CALCULATION OF DISPLACEMENT
def displacements(si):
    # si = ships inputs for prediction
    rho = cm.sea_var.rho

    ## Reading dimensions in full scale
    Lt = si.iloc[:,0]; Bt = si.iloc[:,1] ; Tt = si.iloc[:,2] ; CBt = si.iloc[:,3]
    
    # Define array with displacement: Units [Kg]
    cm.Displaz = Lt*Bt*Tt*CBt*rho

# PREDICTION FOR FN EQUAL TO ZERO
def full_scale_hydrodynamic_param1(si):
    ## transform hidrodynamic parameter in full scale
    # si = ships inputs for prediction
    # inputs: L, B, T, Cb, Cf, Cm, Cc, Xb, Zb, GMT, GML, Fn, beta

    scale = cm.scale1 ## all ships
    Displaz = cm.Displaz ## all ships

    frequencies = cm.sea_var.frequencies
    nfreqs = len(frequencies)
    nheads = len(cm.sea_var.angles)
    grav = cm.sea_var.grav

    ns = np.shape(si)
    nships = ns[0]

    ## Wave caracteristics full scale
    ## Wave model
    ## k_ad== wave number in scale model; length_wave_ad==wavelength in scale model
    k_ad = np.zeros(nfreqs)
    freq_ad = np.zeros(nfreqs)

    s = (nfreqs, nships) # tupple
    length_wave_fs = np.zeros(s)
    k_fs = np.zeros(s)
    fq = np.zeros(s)

    ## Wave full scale
    ## length_wave==wavelength in full scale;
    ## k==wave number in full scale;
    ## freq==frequencies in full scale
    for b in range(nships):
        
        index = cm.sea_var.list_fn_0[b]

        ## Ship caracteristic full scale
        Li = scale[b]*si[b,0]
        Bi = scale[b]*si[b,1] 
        Delta = Displaz[index]

        Ix = Delta*np.power((Bi/4.0),2)
        Iy = Delta*np.power((Li/4.0),2)
        Iz = Delta*np.power(0.3*(np.sqrt(np.power(Bi,2) + np.power(Li,2))),2)

        freq_ad = np.array(frequencies)
        k_ad = (np.power(freq_ad,2))/grav
        length_wave_ad = 2*np.pi/k_ad

        for f in range(nfreqs):
            ## storing variables
            length_wave_fs[f,b] = length_wave_ad[f]*scale[b]
            k_fs[f,b] = 2.0*np.pi/length_wave_fs[f,b]
            fq[f,b] = np.sqrt(grav*k_fs[f,b])

            fr = fq[f,b]
            w2 = np.power(fr,2); 
            k = k_fs[f,b]
            for i in range(nheads):
                 
                ## added mass in full scale
                cm.AM11_fs[f,i,b] = cm.Added_M[f,0,b]*Delta; cm.AM13_fs[f,i,b] = cm.Added_M[f,1,b]*Delta;   cm.AM15_fs[f,i,b] = cm.Added_M[f,2,b]*Delta/k
                cm.AM22_fs[f,i,b] = cm.Added_M[f,3,b]*Delta; cm.AM24_fs[f,i,b] = cm.Added_M[f,4,b]*Delta/k; cm.AM26_fs[f,i,b] = cm.Added_M[f,5,b]*Delta/k 
                cm.AM33_fs[f,i,b] = cm.Added_M[f,7,b]*Delta; cm.AM31_fs[f,i,b] = cm.Added_M[f,6,b]*Delta;   cm.AM35_fs[f,i,b] = cm.Added_M[f,8,b]*Delta/k 
                cm.AM44_fs[f,i,b] = cm.Added_M[f,10,b]*Ix;   cm.AM42_fs[f,i,b] = cm.Added_M[f,9,b]*Ix*k;    cm.AM46_fs[f,i,b] = cm.Added_M[f,11,b]*Ix
                cm.AM55_fs[f,i,b] = cm.Added_M[f,14,b]*Iy;   cm.AM51_fs[f,i,b] = cm.Added_M[f,12,b]*Iy*k;   cm.AM53_fs[f,i,b] = cm.Added_M[f,13,b]*Iy*k
                cm.AM66_fs[f,i,b] = cm.Added_M[f,17,b]*Iz;   cm.AM62_fs[f,i,b] = cm.Added_M[f,15,b]*Iz*k;   cm.AM64_fs[f,i,b] = cm.Added_M[f,16,b]*Iz
                
                ## damping in full scale
                cm.D11_fs[f,i,b] = cm.Damp[f,0,b]*Delta*fr; cm.D13_fs[f,i,b] = cm.Damp[f,1,b]*Delta*fr;   cm.D15_fs[f,i,b] = cm.Damp[f,2,b]*Delta*fr/k
                cm.D22_fs[f,i,b] = cm.Damp[f,3,b]*Delta*fr; cm.D24_fs[f,i,b] = cm.Damp[f,4,b]*Delta*fr/k; cm.D26_fs[f,i,b] = cm.Damp[f,5,b]*Delta*fr/k
                cm.D33_fs[f,i,b] = cm.Damp[f,7,b]*Delta*fr; cm.D31_fs[f,i,b] = cm.Damp[f,6,b]*Delta*fr;   cm.D35_fs[f,i,b] = cm.Damp[f,8,b]*Delta*fr/k
                cm.D44_fs[f,i,b] = cm.Damp[f,10,b]*Ix*fr;   cm.D42_fs[f,i,b] = cm.Damp[f,9,b]*Ix*fr*k;    cm.D46_fs[f,i,b] = cm.Damp[f,11,b]*Ix*fr
                cm.D55_fs[f,i,b] = cm.Damp[f,14,b]*Iy*fr;   cm.D51_fs[f,i,b] = cm.Damp[f,12,b]*Iy*fr*k;   cm.D53_fs[f,i,b] = cm.Damp[f,13,b]*Iy*fr*k   
                cm.D66_fs[f,i,b] = cm.Damp[f,17,b]*Iz*fr;   cm.D62_fs[f,i,b] = cm.Damp[f,15,b]*Iz*fr*k;   cm.D64_fs[f,i,b] = cm.Damp[f,16,b]*Iz*fr
                
            ## forces, moments and phases in 0 deg heading
            ## FX
            cm.FEX_cos_fs[f,0,b] = cm.FEX_cos0_60[f,0,b]*Delta*w2
            cm.FEX_sin_fs[f,0,b] = cm.FEX_sin0_60[f,0,b]*Delta*w2
            cm.FEX_fs[f,0,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,0,b],2)+ np.power(cm.FEX_sin_fs[f,0,b],2))
            cm.Phase_FEX_fs[f,0,b] = cm.Phase_FEX0_60[f,0,b]
            ## FY
            cm.FEY_cos_fs[f,0,b] = 0.0
            cm.FEY_sin_fs[:,0,b] = 0.0
            cm.FEY_fs[f,0,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,0,b],2) + np.power(cm.FEY_sin_fs[f,0,b],2))
            cm.Phase_FEY_fs[f,0,b] = 0.0
            ## FZ
            cm.FEZ_cos_fs[f,0,b] = cm.FEZ_cos[f,0,b]*Delta*w2
            cm.FEZ_sin_fs[f,0,b] = cm.FEZ_sin[f,0,b]*Delta*w2
            cm.FEZ_fs[f,0,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,0,b],2) + np.power(cm.FEZ_sin_fs[f,0,b],2))
            cm.Phase_FEZ_fs[f,0,b] = cm.Phase_FEZ[f,0,b]
            ## MX
            cm.MEX_cos_fs[f,0,b] = 0.0
            cm.MEX_sin_fs[f,0,b] = 0.0
            cm.MEX_fs[f,0,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,0,b],2)+ np.power(cm.MEX_sin_fs[f,0,b],2))
            cm.Phase_MEX_fs[f,0,b] = 0.0
            ## MY
            cm.MEY_cos_fs[f,0,b] = cm.MEY_sin[f,0,b]*Iy*w2*k
            cm.MEY_sin_fs[f,0,b] = cm.MEY_cos[f,0,b]*Iy*w2*k
            cm.MEY_fs[f,0,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,0,b],2) + np.power(cm.MEY_sin_fs[f,0,b],2))
            cm.Phase_MEY_fs[f,0,b] = cm.Phase_MEY[f,0,b]
            ## MZ
            cm.MEZ_cos_fs[f,0,b] = 0.0
            cm.MEZ_sin_fs[f,0,b] = 0.0
            cm.MEZ_fs[f,0,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,0,b],2) + np.power(cm.MEZ_sin_fs[f,0,b],2))
            cm.Phase_MEZ_fs[f,0,b] = 0.0
            ## forces, moments and phases in 30 deg heading
            ## FX
            cm.FEX_cos_fs[f,1,b] = cm.FEX_cos0_60[f,1,b]*Delta*w2
            cm.FEX_sin_fs[f,1,b] = cm.FEX_sin0_60[f,1,b]*Delta*w2
            cm.FEX_fs[f,1,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,1,b],2)+ np.power(cm.FEX_sin_fs[f,1,b],2))
            cm.Phase_FEX_fs[f,1,b] = cm.Phase_FEX0_60[f,1,b]
            ## FY
            cm.FEY_cos_fs[f,1,b] = cm.FEY_cos30_150[f,0,b]*Delta*w2
            cm.FEY_sin_fs[f,1,b] = cm.FEY_sin30_150[f,0,b]*Delta*w2
            cm.FEY_fs[f,1,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,1,b],2) + np.power(cm.FEY_sin_fs[f,1,b],2))
            cm.Phase_FEY_fs[f,1,b] = cm.Phase_FEY30_150[f,0,b]
            ## FZ
            cm.FEZ_cos_fs[f,1,b] = cm.FEZ_cos[f,1,b]*Delta*w2
            cm.FEZ_sin_fs[f,1,b] = cm.FEZ_sin[f,1,b]*Delta*w2
            cm.FEZ_fs[f,1,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,1,b],2) + np.power(cm.FEZ_sin_fs[f,1,b],2))
            cm.Phase_FEZ_fs[f,1,b] = cm.Phase_FEZ[f,1,b]
            ## MX
            cm.MEX_sin_fs[f,1,b] = cm.MEX_sin30_150[f,0,b]*Ix*w2*k
            cm.MEX_cos_fs[f,1,b] = cm.MEX_cos30_150[f,0,b]*Ix*w2*k
            cm.MEX_fs[f,1,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,1,b],2)+ np.power(cm.MEX_sin_fs[f,1,b],2))
            cm.Phase_MEX_fs[f,1,b] = cm.Phase_MEX30_150[f,0,b]
            ## MY
            cm.MEY_cos_fs[f,1,b] = cm.MEY_sin[f,1,b]*Iy*w2*k
            cm.MEY_sin_fs[f,1,b] = cm.MEY_cos[f,1,b]*Iy*w2*k
            cm.MEY_fs[f,1,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,1,b],2)+ np.power(cm.MEY_sin_fs[f,1,b],2))
            cm.Phase_MEY_fs[f,1,b] = cm.Phase_MEY[f,1,b]
            ## MZ
            cm.MEZ_sin_fs[f,1,b] = cm.MEZ_sin30_150[f,0,b]*Iz*w2*k
            cm.MEZ_cos_fs[f,1,b] = cm.MEZ_cos30_150[f,0,b]*Iz*w2*k
            cm.MEZ_fs[f,1,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,1,b],2) + np.power(cm.MEZ_sin_fs[f,1,b],2))
            cm.Phase_MEZ_fs[f,1,b] = cm.Phase_MEZ30_150[f,0,b]
            ## forces, moments and phases in 60 deg heading
            ## FX
            cm.FEX_cos_fs[f,2,b] = cm.FEX_cos0_60[f,2,b]*Delta*w2
            cm.FEX_sin_fs[f,2,b] = cm.FEX_sin0_60[f,2,b]*Delta*w2
            cm.FEX_fs[f,2,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,2,b],2)+ np.power(cm.FEX_sin_fs[f,2,b],2))
            cm.Phase_FEX_fs[f,2,b] = cm.Phase_FEX0_60[f,2,b]
            ## FY
            cm.FEY_cos_fs[f,2,b] = cm.FEY_cos30_150[f,1,b]*Delta*w2
            cm.FEY_sin_fs[f,2,b] = cm.FEY_sin30_150[f,1,b]*Delta*w2
            cm.FEY_fs[f,2,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,2,b],2) + np.power(cm.FEY_sin_fs[f,2,b],2))
            cm.Phase_FEY_fs[f,2,b] = cm.Phase_FEY30_150[f,1,b]
            ## FZ
            cm.FEZ_cos_fs[f,2,b] = cm.FEZ_cos[f,2,b]*Delta*w2
            cm.FEZ_sin_fs[f,2,b] = cm.FEZ_sin[f,2,b]*Delta*w2
            cm.FEZ_fs[f,2,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,2,b],2) + np.power(cm.FEZ_sin_fs[f,2,b],2))
            cm.Phase_FEZ_fs[f,2,b] = cm.Phase_FEZ[f,2,b]
            ## MX
            cm.MEX_sin_fs[f,2,b] = cm.MEX_sin30_150[f,1,b]*Ix*w2*k
            cm.MEX_cos_fs[f,2,b] = cm.MEX_cos30_150[f,1,b]*Ix*w2*k
            cm.MEX_fs[f,2,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,2,b],2)+ np.power(cm.MEX_sin_fs[f,2,b],2))
            cm.Phase_MEX_fs[f,2,b] = cm.Phase_MEX30_150[f,1,b]
            ## MY
            cm.MEY_sin_fs[f,2,b] = cm.MEY_sin[f,2,b]*Iy*w2*k
            cm.MEY_cos_fs[f,2,b] = cm.MEY_cos[f,2,b]*Iy*w2*k
            cm.MEY_fs[f,2,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,2,b],2)+ np.power(cm.MEY_sin_fs[f,2,b],2))
            cm.Phase_MEY_fs[f,2,b] = cm.Phase_MEY[f,2,b]
            ## MZ
            cm.MEZ_sin_fs[f,2,b] = cm.MEZ_sin30_150[f,1,b]*Iz*w2*k
            cm.MEZ_cos_fs[f,2,b] = cm.MEZ_cos30_150[f,1,b]*Iz*w2*k
            cm.MEZ_fs[f,2,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,2,b],2) + np.power(cm.MEZ_sin_fs[f,2,b],2))
            cm.Phase_MEZ_fs[f,2,b] = cm.Phase_MEZ30_150[f,1,b]
            ## forces, moments and phases in 90 deg heading
             ## FX
            cm.FEX_sin_fs[f,3,b] = cm.FEX_cos90[f,0,b]*Delta*w2
            cm.FEX_cos_fs[f,3,b] = cm.FEX_sin90[f,0,b]*Delta*w2
            cm.FEX_fs[f,3,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,3,b],2)+ np.power(cm.FEX_sin_fs[f,3,b],2))
            cm.Phase_FEX_fs[f,3,b] = cm.Phase_FEX90[f,0,b]
            ## FY
            cm.FEY_cos_fs[f,3,b] = cm.FEY_cos30_150[f,2,b]*Delta*w2
            cm.FEY_sin_fs[f,3,b] = cm.FEY_sin30_150[f,2,b]*Delta*w2
            cm.FEY_fs[f,3,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,3,b],2) + np.power(cm.FEY_sin_fs[f,3,b],2))
            cm.Phase_FEY_fs[f,3,b] = cm.Phase_FEY30_150[f,2,b]
            ## FZ
            cm.FEZ_cos_fs[f,3,b] = cm.FEZ_cos[f,3,b]*Delta*w2
            cm.FEZ_sin_fs[f,3,b] = cm.FEZ_sin[f,3,b]*Delta*w2
            cm.FEZ_fs[f,3,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,3,b],2) + np.power(cm.FEZ_sin_fs[f,3,b],2))
            cm.Phase_FEZ_fs[f,3,b] = cm.Phase_FEZ[f,3,b]
            ## MX
            cm.MEX_sin_fs[f,3,b] = cm.MEX_sin30_150[f,2,b]*Ix*w2*k
            cm.MEX_cos_fs[f,3,b] = cm.MEX_cos30_150[f,2,b]*Ix*w2*k
            cm.MEX_fs[f,3,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,3,b],2)+ np.power(cm.MEX_sin_fs[f,3,b],2))
            cm.Phase_MEX_fs[f,3,b] = cm.Phase_MEX30_150[f,2,b]
            ## MY
            cm.MEY_sin_fs[f,3,b] = cm.MEY_sin[f,3,b]*Iy*w2*k
            cm.MEY_cos_fs[f,3,b] = cm.MEY_cos[f,3,b]*Iy*w2*k
            cm.MEY_fs[f,3,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,3,b],2)+ np.power(cm.MEY_sin_fs[f,3,b],2))
            cm.Phase_MEY_fs[f,3,b] = cm.Phase_MEY[f,3,b]
            ## MZ
            cm.MEZ_sin_fs[f,3,b] = cm.MEZ_sin30_150[f,2,b]*Iz*w2*k
            cm.MEZ_cos_fs[f,3,b] = cm.MEZ_cos30_150[f,2,b]*Iz*w2*k
            cm.MEZ_fs[f,3,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,3,b],2) + np.power(cm.MEZ_sin_fs[f,3,b],2))
            cm.Phase_MEZ_fs[f,3,b] = cm.Phase_MEZ30_150[f,2,b]
            ## forces, moments and phases in 120 deg heading
            ## FX
            cm.FEX_cos_fs[f,4,b] = cm.FEX_cos120_180[f,0,b]*Delta*w2
            cm.FEX_sin_fs[f,4,b] = cm.FEX_sin120_180[f,0,b]*Delta*w2
            cm.FEX_fs[f,4,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,4,b],2)+ np.power(cm.FEX_sin_fs[f,4,b],2))
            cm.Phase_FEX_fs[f,4,b] = cm.Phase_FEX120_180[f,0,b]
            ## FY 
            cm.FEY_cos_fs[f,4,b] = cm.FEY_cos30_150[f,3,b]*Delta*w2
            cm.FEY_sin_fs[f,4,b] = cm.FEY_sin30_150[f,3,b]*Delta*w2
            cm.FEY_fs[f,4,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,4,b],2) + np.power(cm.FEY_sin_fs[f,4,b],2))
            cm.Phase_FEY_fs[f,4,b] = cm.Phase_FEY30_150[f,3,b]
            ## FZ
            cm.FEZ_cos_fs[f,4,b] = cm.FEZ_cos[f,4,b]*Delta*w2
            cm.FEZ_sin_fs[f,4,b] = cm.FEZ_sin[f,4,b]*Delta*w2
            cm.FEZ_fs[f,4,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,4,b],2) + np.power(cm.FEZ_sin_fs[f,4,b],2))
            cm.Phase_FEZ_fs[f,4,b] = cm.Phase_FEZ[f,4,b]
            ## MX
            cm.MEX_sin_fs[f,4,b] = cm.MEX_sin30_150[f,3,b]*Ix*w2*k
            cm.MEX_cos_fs[f,4,b] = cm.MEX_cos30_150[f,3,b]*Ix*w2*k
            cm.MEX_fs[f,4,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,4,b],2)+ np.power(cm.MEX_sin_fs[f,4,b],2))
            cm.Phase_MEX_fs[f,4,b] = cm.Phase_MEX30_150[f,3,b]
            ## MY
            cm.MEY_sin_fs[f,4,b] = cm.MEY_sin[f,4,b]*Iy*w2*k
            cm.MEY_cos_fs[f,4,b] = cm.MEY_cos[f,4,b]*Iy*w2*k
            cm.MEY_fs[f,4,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,4,b],2)+ np.power(cm.MEY_sin_fs[f,4,b],2))
            cm.Phase_MEY_fs[f,4,b] = cm.Phase_MEY[f,4,b]
            ## MZ
            cm.MEZ_sin_fs[f,4,b] = cm.MEZ_sin30_150[f,3,b]*Iz*w2*k
            cm.MEZ_cos_fs[f,4,b] = cm.MEZ_cos30_150[f,3,b]*Iz*w2*k
            cm.MEZ_fs[f,4,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,4,b],2) + np.power(cm.MEZ_sin_fs[f,4,b],2))
            cm.Phase_MEZ_fs[f,4,b] = cm.Phase_MEZ30_150[f,3,b]
            ### forces, moments and phases in 150 deg heading
            ## FX
            cm.FEX_cos_fs[f,5,b] = cm.FEX_cos120_180[f,1,b]*Delta*w2
            cm.FEX_sin_fs[f,5,b] = cm.FEX_sin120_180[f,1,b]*Delta*w2
            cm.FEX_fs[f,5,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,5,b],2)+ np.power(cm.FEX_sin_fs[f,5,b],2))
            cm.Phase_FEX_fs[f,5,b] = cm.Phase_FEX120_180[f,1,b]
            ## FY
            cm.FEY_cos_fs[f,5,b] = cm.FEY_cos30_150[f,4,b]*Delta*w2
            cm.FEY_sin_fs[f,5,b] = cm.FEY_sin30_150[f,4,b]*Delta*w2
            cm.FEY_fs[f,5,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,5,b],2) + np.power(cm.FEY_sin_fs[f,5,b],2))
            cm.Phase_FEY_fs[f,5,b] = cm.Phase_FEY30_150[f,4,b]
            ## FZ
            cm.FEZ_cos_fs[f,5,b] = cm.FEZ_cos[f,5,b]*Delta*w2
            cm.FEZ_sin_fs[f,5,b] = cm.FEZ_sin[f,5,b]*Delta*w2
            cm.FEZ_fs[f,5,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,5,b],2) + np.power(cm.FEZ_sin_fs[f,5,b],2))
            cm.Phase_FEZ_fs[f,5,b] = cm.Phase_FEZ[f,5,b]
            ## MX
            cm.MEX_sin_fs[f,5,b] = cm.MEX_sin30_150[f,4,b]*Ix*w2*k
            cm.MEX_cos_fs[f,5,b] = cm.MEX_cos30_150[f,4,b]*Ix*w2*k
            cm.MEX_fs[f,5,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,5,b],2)+ np.power(cm.MEX_sin_fs[f,5,b],2))
            cm.Phase_MEX_fs[f,5,b] = cm.Phase_MEX30_150[f,4,b]
            ## MY
            cm.MEY_sin_fs[f,5,b] = cm.MEY_sin[f,5,b]*Iy*w2*k
            cm.MEY_cos_fs[f,5,b] = cm.MEY_cos[f,5,b]*Iy*w2*k
            cm.MEY_fs[f,5,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,5,b],2)+ np.power(cm.MEY_sin_fs[f,5,b],2))
            cm.Phase_MEY_fs[f,5,b] = cm.Phase_MEY[f,5,b]
            ## MZ
            cm.MEZ_sin_fs[f,5,b] = cm.MEZ_sin30_150[f,4,b]*Iz*w2*k
            cm.MEZ_cos_fs[f,5,b] = cm.MEZ_cos30_150[f,4,b]*Iz*w2*k
            cm.MEZ_fs[f,5,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,5,b],2) + np.power(cm.MEZ_sin_fs[f,5,b],2))
            cm.Phase_MEZ_fs[f,5,b] = cm.Phase_MEZ30_150[f,4,b]
            ## forces, moments and phases in 180 deg heading
            ## FX
            cm.FEX_cos_fs[f,6,b] = cm.FEX_cos120_180[f,2,b]*Delta*w2
            cm.FEX_sin_fs[f,6,b] = cm.FEX_sin120_180[f,2,b]*Delta*w2
            cm.FEX_fs[f,6,b] = np.sqrt(np.power(cm.FEX_cos_fs[f,6,b],2)+ np.power(cm.FEX_sin_fs[f,6,b],2))
            cm.Phase_FEX_fs[f,6,b] = cm.Phase_FEX120_180[f,2,b]
            ## FY
            cm.FEY_cos_fs[f,6,b] = 0.0
            cm.FEY_sin_fs[f,6,b] = 0.0
            cm.FEY_fs[f,6,b] = np.sqrt(np.power(cm.FEY_cos_fs[f,6,b],2) + np.power(cm.FEY_sin_fs[f,6,b],2))
            cm.Phase_FEY_fs[f,6,b] = 0.0
            ## FZ
            cm.FEZ_cos_fs[f,6,b] = cm.FEZ_cos[f,6,b]*Delta*w2
            cm.FEZ_sin_fs[f,6,b] = cm.FEZ_sin[f,6,b]*Delta*w2
            cm.FEZ_fs[f,6,b] = np.sqrt(np.power(cm.FEZ_cos_fs[f,6,b],2) + np.power(cm.FEZ_sin_fs[f,6,b],2))
            cm.Phase_FEZ_fs[f,6,b] = cm.Phase_FEZ[f,6,b]
            ## MX
            cm.MEX_sin_fs[f,6,b] = 0.0
            cm.MEX_cos_fs[f,6,b] = 0.0
            cm.MEX_fs[f,6,b] = np.sqrt(np.power(cm.MEX_cos_fs[f,6,b],2)+ np.power(cm.MEX_sin_fs[f,6,b],2))
            cm.Phase_MEX_fs[f,6,b] = 0.0
            ## MY
            cm.MEY_sin_fs[f,6,b] = cm.MEY_sin[f,6,b]*Iy*w2*k
            cm.MEY_cos_fs[f,6,b] = cm.MEY_cos[f,6,b]*Iy*w2*k
            cm.MEY_fs[f,6,b] = np.sqrt(np.power(cm.MEY_cos_fs[f,6,b],2)+ np.power(cm.MEY_sin_fs[f,6,b],2))
            cm.Phase_MEY_fs[f,6,b] = cm.Phase_MEY[f,6,b]
            ## MZ
            cm.MEZ_sin_fs[f,6,b] = 0.0
            cm.MEZ_cos_fs[f,6,b] = 0.0
            cm.MEZ_fs[f,6,b] = np.sqrt(np.power(cm.MEZ_cos_fs[f,6,b],2) + np.power(cm.MEZ_sin_fs[f,6,b],2))
            cm.Phase_MEZ_fs[f,6,b] = 0.0
            
    # storing frequencies in full scale
    cm.fq1 = fq

# PREDITCTION FOR FN GREATER THAN ZERO
def full_scale_hydrodynamic_param2(si):
    
    ## transform hidrodynamic parameter in full scale
    # si = ships inputs for prediction
    # inputs: L, B, T, Cb, Cf, Cm, Cc, Xb, Zb, GMT, GML, Fn, beta

    scale = cm.scale2 # all ships
    Displaz = cm.Displaz # all ships

    frequencies2 = cm.sea_var.frequencies2
    grav = cm.sea_var.grav
    angles = cm.sea_var.angles
    
    nheads = len(angles)
    nfreqs = len(frequencies2)
    ns =  np.shape(si)
    nships = ns[0]

    ## Wave caracteristics full scale
    ## Wave model
    ## k_ad== wave number in scale model; length_wave_ad==wavelength in scale model
    k_ad = np.zeros(nfreqs)
    freq_ad = np.zeros(nfreqs)

    s = (nfreqs, nships) # tupple
    length_wave_fs = np.zeros(s)
    k_fs = np.zeros(s)
    fq = np.zeros(s)

    ## Wave full scale
    ## length_wave==wavelength in full scale;
    ## k==wave number in full scale;
    ## freq==frequencies in full scale
    
    for b in range(nships):

        # get the ship with fn >0
        index = cm.sea_var.list_fn[b]

        ## Ship caracteristic full scale
        Li = scale[b]*si[b,0]
        Bi = scale[b]*si[b,1] 
        Delta = Displaz[index]

        Ix = Delta*np.power((Bi/4.0),2)
        Iy = Delta*np.power((Li/4.0),2)
        Iz = Delta*np.power(0.3*(np.sqrt(np.power(Bi,2) + np.power(Li,2))),2)

        fn = si[b,14] # Froude number
        #alpha = si[b,15] # Wave heading

        kref = 2.0*np.pi/Li
        wref = np.sqrt(2.0*np.pi*grav/Li)
        wref2 = np.power(wref,2)
        v = fn*np.sqrt(grav*Li)
        
        freq_ad = np.array(frequencies2)
        k_ad = (np.power(freq_ad,2))/grav
        length_wave_ad = 2*np.pi/k_ad

        for f in range(nfreqs):
            ## reference values for dimensionless coefficients
            length_wave_fs[f,b] = length_wave_ad[f]*scale[b]
            k_fs[f,b] = 2.0*np.pi/length_wave_fs[f,b]
            fq[f,b] = np.sqrt(grav*k_fs[f,b])
            fr = fq[f,b]
            numWave = (fr**2)/grav
            
            for j in range(nheads):

                # counterwise
                #                /
                #               / beta
                #              /
                #  ___________/_______
                #  |         /        \______________\ 0 deg
                #  |__________________/              /
                #  
                #beta = angles[j] - alpha
                beta = angles[j]
                wencount = fr*(1-fr*(v/grav)*np.cos(beta*2*np.pi/360))
                we = wencount
                #if wencount < 1.0:
                    # if values are less than training limit we set predicted valued to zero
                    #we = Delta = Ix = Iy = Iz = 0.0
                #else:
                    #we = wencount

                if angles[j] == 0:
                    ## added mass in full scale 0 deg
                    cm.AM11_fs_0[f,b] = cm.Added_M_0_180[f,0,0,b]*Delta; cm.AM13_fs_0[f,b] = cm.Added_M_0_180[f,0,1,b]*Delta;      cm.AM15_fs_0[f,b] = cm.Added_M_0_180[f,0,2,b]*Delta/kref
                    cm.AM22_fs_0[f,b] = cm.Added_M_0_180[f,0,3,b]*Delta; cm.AM24_fs_0[f,b] = cm.Added_M_0_180[f,0,4,b]*Delta/kref; cm.AM26_fs_0[f,b] = cm.Added_M_0_180[f,0,5,b]*Delta/kref 
                    cm.AM33_fs_0[f,b] = cm.Added_M_0_180[f,0,7,b]*Delta; cm.AM31_fs_0[f,b] = cm.Added_M_0_180[f,0,6,b]*Delta;      cm.AM35_fs_0[f,b] = cm.Added_M_0_180[f,0,8,b]*Delta/kref 
                    cm.AM44_fs_0[f,b] = cm.Added_M_0_180[f,0,10,b]*Ix;   cm.AM42_fs_0[f,b] = cm.Added_M_0_180[f,0,9,b]*Ix*kref;    cm.AM46_fs_0[f,b] = cm.Added_M_0_180[f,0,11,b]*Ix
                    cm.AM55_fs_0[f,b] = cm.Added_M_0_180[f,0,14,b]*Iy;   cm.AM51_fs_0[f,b] = cm.Added_M_0_180[f,0,12,b]*Iy*kref;   cm.AM53_fs_0[f,b] = cm.Added_M_0_180[f,0,13,b]*Iy*kref
                    cm.AM66_fs_0[f,b] = cm.Added_M_0_180[f,0,17,b]*Iz;   cm.AM62_fs_0[f,b] = cm.Added_M_0_180[f,0,15,b]*Iz*kref;   cm.AM64_fs_0[f,b] = cm.Added_M_0_180[f,0,16,b]*Iz
                    ## damping in full scale 0 deg
                    cm.D11_fs_0[f,b] = cm.Damp_0_180[f,0,0,b]*Delta*we; cm.D13_fs_0[f,b] = cm.Damp_0_180[f,0,1,b]*Delta*we;      cm.D15_fs_0[f,b] = cm.Damp_0_180[f,0,2,b]*Delta*we/kref
                    cm.D22_fs_0[f,b] = cm.Damp_0_180[f,0,3,b]*Delta*we; cm.D24_fs_0[f,b] = cm.Damp_0_180[f,0,4,b]*Delta*we/kref; cm.D26_fs_0[f,b] = cm.Damp_0_180[f,0,5,b]*Delta*we/kref
                    cm.D33_fs_0[f,b] = cm.Damp_0_180[f,0,7,b]*Delta*we; cm.D31_fs_0[f,b] = cm.Damp_0_180[f,0,6,b]*Delta*we;      cm.D35_fs_0[f,b] = cm.Damp_0_180[f,0,8,b]*Delta*we/kref
                    cm.D44_fs_0[f,b] = cm.Damp_0_180[f,0,10,b]*Ix*we;   cm.D42_fs_0[f,b] = cm.Damp_0_180[f,0,9,b]*Ix*we*kref;    cm.D46_fs_0[f,b] = cm.Damp_0_180[f,0,11,b]*Ix*we
                    cm.D55_fs_0[f,b] = cm.Damp_0_180[f,0,14,b]*Iy*we;   cm.D51_fs_0[f,b] = cm.Damp_0_180[f,0,12,b]*Iy*we*kref;   cm.D53_fs_0[f,b] = cm.Damp_0_180[f,0,13,b]*Iy*we*kref   
                    cm.D66_fs_0[f,b] = cm.Damp_0_180[f,0,17,b]*Iz*we;   cm.D62_fs_0[f,b] = cm.Damp_0_180[f,0,15,b]*Iz*we*kref;   cm.D64_fs_0[f,b] = cm.Damp_0_180[f,0,16,b]*Iz*we

                if angles[j] == 30:
                    ## added mass in full scale 30 deg
                    cm.AM11_fs_30[f,b] = cm.Added_M_0_180[f,1,0,b]*Delta; cm.AM13_fs_30[f,b] = cm.Added_M_0_180[f,1,1,b]*Delta;      cm.AM15_fs_30[f,b] = cm.Added_M_0_180[f,1,2,b]*Delta/kref
                    cm.AM22_fs_30[f,b] = cm.Added_M_0_180[f,1,3,b]*Delta; cm.AM24_fs_30[f,b] = cm.Added_M_0_180[f,1,4,b]*Delta/kref; cm.AM26_fs_30[f,b] = cm.Added_M_0_180[f,1,5,b]*Delta/kref 
                    cm.AM33_fs_30[f,b] = cm.Added_M_0_180[f,1,7,b]*Delta; cm.AM31_fs_30[f,b] = cm.Added_M_0_180[f,1,6,b]*Delta;      cm.AM35_fs_30[f,b] = cm.Added_M_0_180[f,1,8,b]*Delta/kref 
                    cm.AM44_fs_30[f,b] = cm.Added_M_0_180[f,1,10,b]*Ix;   cm.AM42_fs_30[f,b] = cm.Added_M_0_180[f,1,9,b]*Ix*kref;    cm.AM46_fs_30[f,b] = cm.Added_M_0_180[f,1,11,b]*Ix
                    cm.AM55_fs_30[f,b] = cm.Added_M_0_180[f,1,14,b]*Iy;   cm.AM51_fs_30[f,b] = cm.Added_M_0_180[f,1,12,b]*Iy*kref;   cm.AM53_fs_30[f,b] = cm.Added_M_0_180[f,1,13,b]*Iy*kref
                    cm.AM66_fs_30[f,b] = cm.Added_M_0_180[f,1,17,b]*Iz;   cm.AM62_fs_30[f,b] = cm.Added_M_0_180[f,1,15,b]*Iz*kref;   cm.AM64_fs_30[f,b] = cm.Added_M_0_180[f,1,16,b]*Iz
                    ## damping in full scale 30 deg
                    cm.D11_fs_30[f,b] = cm.Damp_0_180[f,1,0,b]*Delta*we; cm.D13_fs_30[f,b] = cm.Damp_0_180[f,1,1,b]*Delta*we;      cm.D15_fs_30[f,b] = cm.Damp_0_180[f,1,2,b]*Delta*we/kref
                    cm.D22_fs_30[f,b] = cm.Damp_0_180[f,1,3,b]*Delta*we; cm.D24_fs_30[f,b] = cm.Damp_0_180[f,1,4,b]*Delta*we/kref; cm.D26_fs_30[f,b] = cm.Damp_0_180[f,1,5,b]*Delta*we/kref
                    cm.D33_fs_30[f,b] = cm.Damp_0_180[f,1,7,b]*Delta*we; cm.D31_fs_30[f,b] = cm.Damp_0_180[f,1,6,b]*Delta*we;      cm.D35_fs_30[f,b] = cm.Damp_0_180[f,1,8,b]*Delta*we/kref
                    cm.D44_fs_30[f,b] = cm.Damp_0_180[f,1,10,b]*Ix*we;   cm.D42_fs_30[f,b] = cm.Damp_0_180[f,1,9,b]*Ix*we*kref;    cm.D46_fs_30[f,b] = cm.Damp_0_180[f,1,11,b]*Ix*we
                    cm.D55_fs_30[f,b] = cm.Damp_0_180[f,1,14,b]*Iy*we;   cm.D51_fs_30[f,b] = cm.Damp_0_180[f,1,12,b]*Iy*we*kref;   cm.D53_fs_30[f,b] = cm.Damp_0_180[f,1,13,b]*Iy*we*kref   
                    cm.D66_fs_30[f,b] = cm.Damp_0_180[f,1,17,b]*Iz*we;   cm.D62_fs_30[f,b] = cm.Damp_0_180[f,1,15,b]*Iz*we*kref;   cm.D64_fs_30[f,b] = cm.Damp_0_180[f,1,16,b]*Iz*we

                if angles[j] == 60:
                    ## added mass in full scale 60 deg
                    cm.AM11_fs_60[f,b] = cm.Added_M_0_180[f,2,0,b]*Delta; cm.AM13_fs_60[f,b] = cm.Added_M_0_180[f,2,1,b]*Delta;      cm.AM15_fs_60[f,b] = cm.Added_M_0_180[f,2,2,b]*Delta/kref
                    cm.AM22_fs_60[f,b] = cm.Added_M_0_180[f,2,3,b]*Delta; cm.AM24_fs_60[f,b] = cm.Added_M_0_180[f,2,4,b]*Delta/kref; cm.AM26_fs_60[f,b] = cm.Added_M_0_180[f,2,5,b]*Delta/kref 
                    cm.AM33_fs_60[f,b] = cm.Added_M_0_180[f,2,7,b]*Delta; cm.AM31_fs_60[f,b] = cm.Added_M_0_180[f,2,6,b]*Delta;      cm.AM35_fs_60[f,b] = cm.Added_M_0_180[f,2,8,b]*Delta/kref 
                    cm.AM44_fs_60[f,b] = cm.Added_M_0_180[f,2,10,b]*Ix;   cm.AM42_fs_60[f,b] = cm.Added_M_0_180[f,2,9,b]*Ix*kref;    cm.AM46_fs_60[f,b] = cm.Added_M_0_180[f,2,11,b]*Ix
                    cm.AM55_fs_60[f,b] = cm.Added_M_0_180[f,2,14,b]*Iy;   cm.AM51_fs_60[f,b] = cm.Added_M_0_180[f,2,12,b]*Iy*kref;   cm.AM53_fs_60[f,b] = cm.Added_M_0_180[f,2,13,b]*Iy*kref
                    cm.AM66_fs_60[f,b] = cm.Added_M_0_180[f,2,17,b]*Iz;   cm.AM62_fs_60[f,b] = cm.Added_M_0_180[f,2,15,b]*Iz*kref;   cm.AM64_fs_60[f,b] = cm.Added_M_0_180[f,2,16,b]*Iz
                    ## damping in full scale 60 deg
                    cm.D11_fs_60[f,b] = cm.Damp_0_180[f,2,0,b]*Delta*we; cm.D13_fs_60[f,b] = cm.Damp_0_180[f,2,1,b]*Delta*we;      cm.D15_fs_60[f,b] = cm.Damp_0_180[f,2,2,b]*Delta*we/kref
                    cm.D22_fs_60[f,b] = cm.Damp_0_180[f,2,3,b]*Delta*we; cm.D24_fs_60[f,b] = cm.Damp_0_180[f,2,4,b]*Delta*we/kref; cm.D26_fs_60[f,b] = cm.Damp_0_180[f,2,5,b]*Delta*we/kref
                    cm.D33_fs_60[f,b] = cm.Damp_0_180[f,2,7,b]*Delta*we; cm.D31_fs_60[f,b] = cm.Damp_0_180[f,2,6,b]*Delta*we;      cm.D35_fs_60[f,b] = cm.Damp_0_180[f,2,8,b]*Delta*we/kref
                    cm.D44_fs_60[f,b] = cm.Damp_0_180[f,2,10,b]*Ix*we;   cm.D42_fs_60[f,b] = cm.Damp_0_180[f,2,9,b]*Ix*we*kref;    cm.D46_fs_60[f,b] = cm.Damp_0_180[f,2,11,b]*Ix*we
                    cm.D55_fs_60[f,b] = cm.Damp_0_180[f,2,14,b]*Iy*we;   cm.D51_fs_60[f,b] = cm.Damp_0_180[f,2,12,b]*Iy*we*kref;   cm.D53_fs_60[f,b] = cm.Damp_0_180[f,2,13,b]*Iy*we*kref   
                    cm.D66_fs_60[f,b] = cm.Damp_0_180[f,2,17,b]*Iz*we;   cm.D62_fs_60[f,b] = cm.Damp_0_180[f,2,15,b]*Iz*we*kref;   cm.D64_fs_60[f,b] = cm.Damp_0_180[f,2,16,b]*Iz*we

                if angles[j] == 90:  
                    ## added mass in full scale 90 deg
                    cm.AM11_fs_90[f,b] = cm.Added_M_0_180[f,3,0,b]*Delta; cm.AM13_fs_90[f,b] = cm.Added_M_0_180[f,3,1,b]*Delta;      cm.AM15_fs_90[f,b] = cm.Added_M_0_180[f,3,2,b]*Delta/kref
                    cm.AM22_fs_90[f,b] = cm.Added_M_0_180[f,3,3,b]*Delta; cm.AM24_fs_90[f,b] = cm.Added_M_0_180[f,3,4,b]*Delta/kref; cm.AM26_fs_90[f,b] = cm.Added_M_0_180[f,3,5,b]*Delta/kref 
                    cm.AM33_fs_90[f,b] = cm.Added_M_0_180[f,3,7,b]*Delta; cm.AM31_fs_90[f,b] = cm.Added_M_0_180[f,3,6,b]*Delta;      cm.AM35_fs_90[f,b] = cm.Added_M_0_180[f,3,8,b]*Delta/kref 
                    cm.AM44_fs_90[f,b] = cm.Added_M_0_180[f,3,10,b]*Ix;   cm.AM42_fs_90[f,b] = cm.Added_M_0_180[f,3,9,b]*Ix*kref;    cm.AM46_fs_90[f,b] = cm.Added_M_0_180[f,3,11,b]*Ix
                    cm.AM55_fs_90[f,b] = cm.Added_M_0_180[f,3,14,b]*Iy;   cm.AM51_fs_90[f,b] = cm.Added_M_0_180[f,3,12,b]*Iy*kref;   cm.AM53_fs_90[f,b] = cm.Added_M_0_180[f,3,13,b]*Iy*kref
                    cm.AM66_fs_90[f,b] = cm.Added_M_0_180[f,3,17,b]*Iz;   cm.AM62_fs_90[f,b] = cm.Added_M_0_180[f,3,15,b]*Iz*kref;   cm.AM64_fs_90[f,b] = cm.Added_M_0_180[f,3,16,b]*Iz
                    ## damping in full scale 90 deg
                    cm.D11_fs_90[f,b] = cm.Damp_0_180[f,3,0,b]*Delta*we; cm.D13_fs_90[f,b] = cm.Damp_0_180[f,3,1,b]*Delta*we;      cm.D15_fs_90[f,b] = cm.Damp_0_180[f,3,2,b]*Delta*we/kref
                    cm.D22_fs_90[f,b] = cm.Damp_0_180[f,3,3,b]*Delta*we; cm.D24_fs_90[f,b] = cm.Damp_0_180[f,3,4,b]*Delta*we/kref; cm.D26_fs_90[f,b] = cm.Damp_0_180[f,3,5,b]*Delta*we/kref
                    cm.D33_fs_90[f,b] = cm.Damp_0_180[f,3,7,b]*Delta*we; cm.D31_fs_90[f,b] = cm.Damp_0_180[f,3,6,b]*Delta*we;      cm.D35_fs_90[f,b] = cm.Damp_0_180[f,3,8,b]*Delta*we/kref
                    cm.D44_fs_90[f,b] = cm.Damp_0_180[f,3,10,b]*Ix*we;   cm.D42_fs_90[f,b] = cm.Damp_0_180[f,3,9,b]*Ix*we*kref;    cm.D46_fs_90[f,b] = cm.Damp_0_180[f,3,11,b]*Ix*we
                    cm.D55_fs_90[f,b] = cm.Damp_0_180[f,3,14,b]*Iy*we;   cm.D51_fs_90[f,b] = cm.Damp_0_180[f,3,12,b]*Iy*we*kref;   cm.D53_fs_90[f,b] = cm.Damp_0_180[f,3,13,b]*Iy*we*kref   
                    cm.D66_fs_90[f,b] = cm.Damp_0_180[f,3,17,b]*Iz*we;   cm.D62_fs_90[f,b] = cm.Damp_0_180[f,3,15,b]*Iz*we*kref;   cm.D64_fs_90[f,b] = cm.Damp_0_180[f,3,16,b]*Iz*we

                if angles[j] == 120:  
                    ## added mass in full scale 120 deg
                    cm.AM11_fs_120[f,b] = cm.Added_M_0_180[f,4,0,b]*Delta; cm.AM13_fs_120[f,b] = cm.Added_M_0_180[f,4,1,b]*Delta;      cm.AM15_fs_120[f,b] = cm.Added_M_0_180[f,4,2,b]*Delta/kref
                    cm.AM22_fs_120[f,b] = cm.Added_M_0_180[f,4,3,b]*Delta; cm.AM24_fs_120[f,b] = cm.Added_M_0_180[f,4,4,b]*Delta/kref; cm.AM26_fs_120[f,b] = cm.Added_M_0_180[f,4,5,b]*Delta/kref 
                    cm.AM33_fs_120[f,b] = cm.Added_M_0_180[f,4,7,b]*Delta; cm.AM31_fs_120[f,b] = cm.Added_M_0_180[f,4,6,b]*Delta;      cm.AM35_fs_120[f,b] = cm.Added_M_0_180[f,4,8,b]*Delta/kref 
                    cm.AM44_fs_120[f,b] = cm.Added_M_0_180[f,4,10,b]*Ix;   cm.AM42_fs_120[f,b] = cm.Added_M_0_180[f,4,9,b]*Ix*kref;    cm.AM46_fs_120[f,b] = cm.Added_M_0_180[f,4,11,b]*Ix
                    cm.AM55_fs_120[f,b] = cm.Added_M_0_180[f,4,14,b]*Iy;   cm.AM51_fs_120[f,b] = cm.Added_M_0_180[f,4,12,b]*Iy*kref;   cm.AM53_fs_120[f,b] = cm.Added_M_0_180[f,4,13,b]*Iy*kref
                    cm.AM66_fs_120[f,b] = cm.Added_M_0_180[f,4,17,b]*Iz;   cm.AM62_fs_120[f,b] = cm.Added_M_0_180[f,4,15,b]*Iz*kref;   cm.AM64_fs_120[f,b] = cm.Added_M_0_180[f,4,16,b]*Iz
                    ## damping in full scale 120 deg
                    cm.D11_fs_120[f,b] = cm.Damp_0_180[f,4,0,b]*Delta*we; cm.D13_fs_120[f,b] = cm.Damp_0_180[f,4,1,b]*Delta*we;      cm.D15_fs_120[f,b] = cm.Damp_0_180[f,4,2,b]*Delta*we/kref
                    cm.D22_fs_120[f,b] = cm.Damp_0_180[f,4,3,b]*Delta*we; cm.D24_fs_120[f,b] = cm.Damp_0_180[f,4,4,b]*Delta*we/kref; cm.D26_fs_120[f,b] = cm.Damp_0_180[f,4,5,b]*Delta*we/kref
                    cm.D33_fs_120[f,b] = cm.Damp_0_180[f,4,7,b]*Delta*we; cm.D31_fs_120[f,b] = cm.Damp_0_180[f,4,6,b]*Delta*we;      cm.D35_fs_120[f,b] = cm.Damp_0_180[f,4,8,b]*Delta*we/kref
                    cm.D44_fs_120[f,b] = cm.Damp_0_180[f,4,10,b]*Ix*we;   cm.D42_fs_120[f,b] = cm.Damp_0_180[f,4,9,b]*Ix*we*kref;    cm.D46_fs_120[f,b] = cm.Damp_0_180[f,4,11,b]*Ix*we
                    cm.D55_fs_120[f,b] = cm.Damp_0_180[f,4,14,b]*Iy*we;   cm.D51_fs_120[f,b] = cm.Damp_0_180[f,4,12,b]*Iy*we*kref;   cm.D53_fs_120[f,b] = cm.Damp_0_180[f,4,13,b]*Iy*we*kref   
                    cm.D66_fs_120[f,b] = cm.Damp_0_180[f,4,17,b]*Iz*we;   cm.D62_fs_120[f,b] = cm.Damp_0_180[f,4,15,b]*Iz*we*kref;   cm.D64_fs_120[f,b] = cm.Damp_0_180[f,4,16,b]*Iz*we
                
                if angles[j] == 150:
                    ## added mass in full scale 150 deg
                    cm.AM11_fs_150[f,b] = cm.Added_M_0_180[f,5,0,b]*Delta; cm.AM13_fs_150[f,b] = cm.Added_M_0_180[f,5,1,b]*Delta;      cm.AM15_fs_150[f,b] = cm.Added_M_0_180[f,5,2,b]*Delta/kref
                    cm.AM22_fs_150[f,b] = cm.Added_M_0_180[f,5,3,b]*Delta; cm.AM24_fs_150[f,b] = cm.Added_M_0_180[f,5,4,b]*Delta/kref; cm.AM26_fs_150[f,b] = cm.Added_M_0_180[f,5,5,b]*Delta/kref 
                    cm.AM33_fs_150[f,b] = cm.Added_M_0_180[f,5,7,b]*Delta; cm.AM31_fs_150[f,b] = cm.Added_M_0_180[f,5,6,b]*Delta;      cm.AM35_fs_150[f,b] = cm.Added_M_0_180[f,5,8,b]*Delta/kref 
                    cm.AM44_fs_150[f,b] = cm.Added_M_0_180[f,5,10,b]*Ix;   cm.AM42_fs_150[f,b] = cm.Added_M_0_180[f,5,9,b]*Ix*kref;    cm.AM46_fs_150[f,b] = cm.Added_M_0_180[f,5,11,b]*Ix
                    cm.AM55_fs_150[f,b] = cm.Added_M_0_180[f,5,14,b]*Iy;   cm.AM51_fs_150[f,b] = cm.Added_M_0_180[f,5,12,b]*Iy*kref;   cm.AM53_fs_150[f,b] = cm.Added_M_0_180[f,5,13,b]*Iy*kref
                    cm.AM66_fs_150[f,b] = cm.Added_M_0_180[f,5,17,b]*Iz;   cm.AM62_fs_150[f,b] = cm.Added_M_0_180[f,5,15,b]*Iz*kref;   cm.AM64_fs_150[f,b] = cm.Added_M_0_180[f,5,16,b]*Iz
                    ## damping in full scale 150 deg
                    cm.D11_fs_150[f,b] = cm.Damp_0_180[f,5,0,b]*Delta*we; cm.D13_fs_150[f,b] = cm.Damp_0_180[f,5,1,b]*Delta*we;      cm.D15_fs_150[f,b] = cm.Damp_0_180[f,5,2,b]*Delta*we/kref
                    cm.D22_fs_150[f,b] = cm.Damp_0_180[f,5,3,b]*Delta*we; cm.D24_fs_150[f,b] = cm.Damp_0_180[f,5,4,b]*Delta*we/kref; cm.D26_fs_150[f,b] = cm.Damp_0_180[f,5,5,b]*Delta*we/kref
                    cm.D33_fs_150[f,b] = cm.Damp_0_180[f,5,7,b]*Delta*we; cm.D31_fs_150[f,b] = cm.Damp_0_180[f,5,6,b]*Delta*we;      cm.D35_fs_150[f,b] = cm.Damp_0_180[f,5,8,b]*Delta*we/kref
                    cm.D44_fs_150[f,b] = cm.Damp_0_180[f,5,10,b]*Ix*we;   cm.D42_fs_150[f,b] = cm.Damp_0_180[f,5,9,b]*Ix*we*kref;    cm.D46_fs_150[f,b] = cm.Damp_0_180[f,5,11,b]*Ix*we
                    cm.D55_fs_150[f,b] = cm.Damp_0_180[f,5,14,b]*Iy*we;   cm.D51_fs_150[f,b] = cm.Damp_0_180[f,5,12,b]*Iy*we*kref;   cm.D53_fs_150[f,b] = cm.Damp_0_180[f,5,13,b]*Iy*we*kref   
                    cm.D66_fs_150[f,b] = cm.Damp_0_180[f,5,17,b]*Iz*we;   cm.D62_fs_150[f,b] = cm.Damp_0_180[f,5,15,b]*Iz*we*kref;   cm.D64_fs_150[f,b] = cm.Damp_0_180[f,5,16,b]*Iz*we
                
                if angles[j] == 180:
                    ## added mass in full scale 180
                    cm.AM11_fs_180[f,b] = cm.Added_M_0_180[f,6,0,b]*Delta; cm.AM13_fs_180[f,b] = cm.Added_M_0_180[f,6,1,b]*Delta;      cm.AM15_fs_180[f,b] = cm.Added_M_0_180[f,6,2,b]*Delta/kref
                    cm.AM22_fs_180[f,b] = cm.Added_M_0_180[f,6,3,b]*Delta; cm.AM24_fs_180[f,b] = cm.Added_M_0_180[f,6,4,b]*Delta/kref; cm.AM26_fs_180[f,b] = cm.Added_M_0_180[f,6,5,b]*Delta/kref 
                    cm.AM33_fs_180[f,b] = cm.Added_M_0_180[f,6,7,b]*Delta; cm.AM31_fs_180[f,b] = cm.Added_M_0_180[f,6,6,b]*Delta;      cm.AM35_fs_180[f,b] = cm.Added_M_0_180[f,6,8,b]*Delta/kref 
                    cm.AM44_fs_180[f,b] = cm.Added_M_0_180[f,6,10,b]*Ix;   cm.AM42_fs_180[f,b] = cm.Added_M_0_180[f,6,9,b]*Ix*kref;    cm.AM46_fs_180[f,b] = cm.Added_M_0_180[f,6,11,b]*Ix
                    cm.AM55_fs_180[f,b] = cm.Added_M_0_180[f,6,14,b]*Iy;   cm.AM51_fs_180[f,b] = cm.Added_M_0_180[f,6,12,b]*Iy*kref;   cm.AM53_fs_180[f,b] = cm.Added_M_0_180[f,6,13,b]*Iy*kref
                    cm.AM66_fs_180[f,b] = cm.Added_M_0_180[f,6,17,b]*Iz;   cm.AM62_fs_180[f,b] = cm.Added_M_0_180[f,6,15,b]*Iz*kref;   cm.AM64_fs_180[f,b] = cm.Added_M_0_180[f,6,16,b]*Iz
                    ## damping in full scale 180 deg
                    cm.D11_fs_180[f,b] = cm.Damp_0_180[f,6,0,b]*Delta*we; cm.D13_fs_180[f,b] = cm.Damp_0_180[f,6,1,b]*Delta*we;      cm.D15_fs_180[f,b] = cm.Damp_0_180[f,6,2,b]*Delta*we/kref
                    cm.D22_fs_180[f,b] = cm.Damp_0_180[f,6,3,b]*Delta*we; cm.D24_fs_180[f,b] = cm.Damp_0_180[f,6,4,b]*Delta*we/kref; cm.D26_fs_180[f,b] = cm.Damp_0_180[f,6,5,b]*Delta*we/kref
                    cm.D33_fs_180[f,b] = cm.Damp_0_180[f,6,7,b]*Delta*we; cm.D31_fs_180[f,b] = cm.Damp_0_180[f,6,6,b]*Delta*we;      cm.D35_fs_180[f,b] = cm.Damp_0_180[f,6,8,b]*Delta*we/kref
                    cm.D44_fs_180[f,b] = cm.Damp_0_180[f,6,10,b]*Ix*we;   cm.D42_fs_180[f,b] = cm.Damp_0_180[f,6,9,b]*Ix*we*kref;    cm.D46_fs_180[f,b] = cm.Damp_0_180[f,6,11,b]*Ix*we
                    cm.D55_fs_180[f,b] = cm.Damp_0_180[f,6,14,b]*Iy*we;   cm.D51_fs_180[f,b] = cm.Damp_0_180[f,6,12,b]*Iy*we*kref;   cm.D53_fs_180[f,b] = cm.Damp_0_180[f,6,13,b]*Iy*we*kref   
                    cm.D66_fs_180[f,b] = cm.Damp_0_180[f,6,17,b]*Iz*we;   cm.D62_fs_180[f,b] = cm.Damp_0_180[f,6,15,b]*Iz*we*kref;   cm.D64_fs_180[f,b] = cm.Damp_0_180[f,6,16,b]*Iz*we
                
            ## forces, moments and phases in 0 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,0,b] = cm.FEX_cos0_60_din[f,0,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,0,b] = cm.FEX_sin0_60_din[f,0,b]*Delta*wref2
            cm.FEX_fs_din[f,0,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,0,b],2)+ np.power(cm.FEX_sin_fs_din[f,0,b],2))
            cm.Phase_FEX_fs_din[f,0,b] = cm.Phase_FEX0_60_din[f,0,b]
            ## FY
            cm.FEY_cos_fs_din[f,0,b] = 0.0
            cm.FEY_sin_fs_din[:,0,b] = 0.0
            cm.FEY_fs_din[f,0,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,0,b],2) + np.power(cm.FEY_sin_fs_din[f,0,b],2))
            cm.Phase_FEY_fs_din[f,0,b] = 0.0
            ## FZ
            cm.FEZ_cos_fs_din[f,0,b] = cm.FEZ_cos_din[f,0,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,0,b] = cm.FEZ_sin_din[f,0,b]*Delta*wref2
            cm.FEZ_fs_din[f,0,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,0,b],2) + np.power(cm.FEZ_sin_fs_din[f,0,b],2))
            cm.Phase_FEZ_fs_din[f,0,b] = cm.Phase_FEZ_din[f,0,b]
            ## MX
            cm.MEX_cos_fs_din[f,0,b] = 0.0
            cm.MEX_sin_fs_din[f,0,b] = 0.0
            cm.MEX_fs_din[f,0,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,0,b],2)+ np.power(cm.MEX_sin_fs_din[f,0,b],2))
            cm.Phase_MEX_fs_din[f,0,b] = 0.0
            ## MY
            cm.MEY_cos_fs_din[f,0,b] = cm.MEY_sin_din[f,0,b]*Iy*wref2*numWave
            cm.MEY_sin_fs_din[f,0,b] = cm.MEY_cos_din[f,0,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,0,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,0,b],2) + np.power(cm.MEY_sin_fs_din[f,0,b],2))
            cm.Phase_MEY_fs_din[f,0,b] = cm.Phase_MEY_din[f,0,b]
            ## MZ
            cm.MEZ_cos_fs_din[f,0,b] = 0.0
            cm.MEZ_sin_fs_din[f,0,b] = 0.0
            cm.MEZ_fs_din[f,0,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,0,b],2) + np.power(cm.MEZ_sin_fs_din[f,0,b],2))
            cm.Phase_MEZ_fs_din[f,0,b] = 0.0
            ## forces, moments and phases in 30 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,1,b] = cm.FEX_cos0_60_din[f,1,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,1,b] = cm.FEX_sin0_60_din[f,1,b]*Delta*wref2
            cm.FEX_fs_din[f,1,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,1,b],2)+ np.power(cm.FEX_sin_fs_din[f,1,b],2))
            cm.Phase_FEX_fs_din[f,1,b] = cm.Phase_FEX0_60_din[f,1,b]
            ## FY
            cm.FEY_cos_fs_din[f,1,b] = cm.FEY_cos30_150_din[f,0,b]*Delta*wref2
            cm.FEY_sin_fs_din[f,1,b] = cm.FEY_sin30_150_din[f,0,b]*Delta*wref2
            cm.FEY_fs_din[f,1,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,1,b],2) + np.power(cm.FEY_sin_fs_din[f,1,b],2))
            cm.Phase_FEY_fs_din[f,1,b] = cm.Phase_FEY30_150_din[f,0,b]
            ## FZ
            cm.FEZ_cos_fs_din[f,1,b] = cm.FEZ_cos_din[f,1,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,1,b] = cm.FEZ_sin_din[f,1,b]*Delta*wref2
            cm.FEZ_fs_din[f,1,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,1,b],2) + np.power(cm.FEZ_sin_fs_din[f,1,b],2))
            cm.Phase_FEZ_fs_din[f,1,b] = cm.Phase_FEZ_din[f,1,b]
            ## MX
            cm.MEX_sin_fs_din[f,1,b] = cm.MEX_sin30_150_din[f,0,b]*Ix*wref2*numWave
            cm.MEX_cos_fs_din[f,1,b] = cm.MEX_cos30_150_din[f,0,b]*Ix*wref2*numWave
            cm.MEX_fs_din[f,1,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,1,b],2)+ np.power(cm.MEX_sin_fs_din[f,1,b],2))
            cm.Phase_MEX_fs_din[f,1,b] = cm.Phase_MEX30_150_din[f,0,b]
            ## MY
            cm.MEY_cos_fs_din[f,1,b] = cm.MEY_sin_din[f,1,b]*Iy*wref2*numWave
            cm.MEY_sin_fs_din[f,1,b] = cm.MEY_cos_din[f,1,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,1,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,1,b],2)+ np.power(cm.MEY_sin_fs_din[f,1,b],2))
            cm.Phase_MEY_fs_din[f,1,b] = cm.Phase_MEY_din[f,1,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,1,b] = cm.MEZ_sin30_150_din[f,0,b]*Iz*wref2*numWave
            cm.MEZ_cos_fs_din[f,1,b] = cm.MEZ_cos30_150_din[f,0,b]*Iz*wref2*numWave
            cm.MEZ_fs_din[f,1,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,1,b],2) + np.power(cm.MEZ_sin_fs_din[f,1,b],2))
            cm.Phase_MEZ_fs_din[f,1,b] = cm.Phase_MEZ30_150_din[f,0,b]
            ## forces, moments and phases in 60 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,2,b] = cm.FEX_cos0_60_din[f,2,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,2,b] = cm.FEX_sin0_60_din[f,2,b]*Delta*wref2
            cm.FEX_fs_din[f,2,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,2,b],2)+ np.power(cm.FEX_sin_fs_din[f,2,b],2))
            cm.Phase_FEX_fs_din[f,2,b] = cm.Phase_FEX0_60_din[f,2,b]
            ## FY
            cm.FEY_cos_fs_din[f,2,b] = cm.FEY_cos30_150_din[f,1,b]*Delta*wref2
            cm.FEY_sin_fs_din[f,2,b] = cm.FEY_sin30_150_din[f,1,b]*Delta*wref2
            cm.FEY_fs_din[f,2,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,2,b],2) + np.power(cm.FEY_sin_fs_din[f,2,b],2))
            cm.Phase_FEY_fs_din[f,2,b] = cm.Phase_FEY30_150_din[f,1,b]
            ## FZ
            cm.FEZ_cos_fs_din[f,2,b] = cm.FEZ_cos_din[f,2,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,2,b] = cm.FEZ_sin_din[f,2,b]*Delta*wref2
            cm.FEZ_fs_din[f,2,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,2,b],2) + np.power(cm.FEZ_sin_fs_din[f,2,b],2))
            cm.Phase_FEZ_fs_din[f,2,b] = cm.Phase_FEZ_din[f,2,b]
            ## MX
            cm.MEX_sin_fs_din[f,2,b] = cm.MEX_sin30_150_din[f,1,b]*Ix*wref2*numWave
            cm.MEX_cos_fs_din[f,2,b] = cm.MEX_cos30_150_din[f,1,b]*Ix*wref2*numWave
            cm.MEX_fs_din[f,2,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,2,b],2)+ np.power(cm.MEX_sin_fs_din[f,2,b],2))
            cm.Phase_MEX_fs_din[f,2,b] = cm.Phase_MEX30_150_din[f,1,b]
            ## MY
            cm.MEY_sin_fs_din[f,2,b] = cm.MEY_sin_din[f,2,b]*Iy*wref2*numWave
            cm.MEY_cos_fs_din[f,2,b] = cm.MEY_cos_din[f,2,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,2,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,2,b],2)+ np.power(cm.MEY_sin_fs_din[f,2,b],2))
            cm.Phase_MEY_fs_din[f,2,b] = cm.Phase_MEY_din[f,2,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,2,b] = cm.MEZ_sin30_150_din[f,1,b]*Iz*wref2*numWave
            cm.MEZ_cos_fs_din[f,2,b] = cm.MEZ_cos30_150_din[f,1,b]*Iz*wref2*numWave
            cm.MEZ_fs_din[f,2,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,2,b],2) + np.power(cm.MEZ_sin_fs_din[f,2,b],2))
            cm.Phase_MEZ_fs_din[f,2,b] = cm.Phase_MEZ30_150_din[f,1,b]
            ## forces, moments and phases in 90 deg heading
             ## FX
            cm.FEX_sin_fs_din[f,3,b] = cm.FEX_cos90_din[f,0,b]*Delta*wref2
            cm.FEX_cos_fs_din[f,3,b] = cm.FEX_sin90_din[f,0,b]*Delta*wref2
            cm.FEX_fs_din[f,3,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,3,b],2)+ np.power(cm.FEX_sin_fs_din[f,3,b],2))
            cm.Phase_FEX_fs_din[f,3,b] = cm.Phase_FEX90_din[f,0,b]
            ## FY
            cm.FEY_cos_fs_din[f,3,b] = cm.FEY_cos30_150_din[f,2,b]*Delta*wref2
            cm.FEY_sin_fs_din[f,3,b] = cm.FEY_sin30_150_din[f,2,b]*Delta*wref2
            cm.FEY_fs_din[f,3,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,3,b],2) + np.power(cm.FEY_sin_fs_din[f,3,b],2))
            cm.Phase_FEY_fs_din[f,3,b] = cm.Phase_FEY30_150_din[f,2,b]
            ## FZ
            cm.FEZ_cos_fs_din[f,3,b] = cm.FEZ_cos_din[f,3,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,3,b] = cm.FEZ_sin_din[f,3,b]*Delta*wref2
            cm.FEZ_fs_din[f,3,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,3,b],2) + np.power(cm.FEZ_sin_fs_din[f,3,b],2))
            cm.Phase_FEZ_fs_din[f,3,b] = cm.Phase_FEZ_din[f,3,b]
            ## MX
            cm.MEX_sin_fs_din[f,3,b] = cm.MEX_sin30_150_din[f,2,b]*Ix*wref2*numWave
            cm.MEX_cos_fs_din[f,3,b] = cm.MEX_cos30_150_din[f,2,b]*Ix*wref2*numWave
            cm.MEX_fs_din[f,3,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,3,b],2)+ np.power(cm.MEX_sin_fs_din[f,3,b],2))
            cm.Phase_MEX_fs_din[f,3,b] = cm.Phase_MEX30_150_din[f,2,b]
            ## MY
            cm.MEY_sin_fs_din[f,3,b] = cm.MEY_sin_din[f,3,b]*Iy*wref2*numWave
            cm.MEY_cos_fs_din[f,3,b] = cm.MEY_cos_din[f,3,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,3,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,3,b],2)+ np.power(cm.MEY_sin_fs_din[f,3,b],2))
            cm.Phase_MEY_fs_din[f,3,b] = cm.Phase_MEY_din[f,3,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,3,b] = cm.MEZ_sin30_150_din[f,2,b]*Iz*wref2*numWave
            cm.MEZ_cos_fs_din[f,3,b] = cm.MEZ_cos30_150_din[f,2,b]*Iz*wref2*numWave
            cm.MEZ_fs_din[f,3,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,3,b],2) + np.power(cm.MEZ_sin_fs_din[f,3,b],2))
            cm.Phase_MEZ_fs_din[f,3,b] = cm.Phase_MEZ30_150_din[f,2,b]
            ## forces, moments and phases in 120 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,4,b] = cm.FEX_cos120_180_din[f,0,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,4,b] = cm.FEX_sin120_180_din[f,0,b]*Delta*wref2
            cm.FEX_fs_din[f,4,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,4,b],2)+ np.power(cm.FEX_sin_fs_din[f,4,b],2))
            cm.Phase_FEX_fs_din[f,4,b] = cm.Phase_FEX120_180_din[f,0,b]
            ## FY 
            cm.FEY_cos_fs_din[f,4,b] = cm.FEY_cos30_150_din[f,3,b]*Delta*wref2
            cm.FEY_sin_fs_din[f,4,b] = cm.FEY_sin30_150_din[f,3,b]*Delta*wref2
            cm.FEY_fs_din[f,4,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,4,b],2) + np.power(cm.FEY_sin_fs_din[f,4,b],2))
            cm.Phase_FEY_fs_din[f,4,b] = cm.Phase_FEY30_150_din[f,3,b]
            ## FZ
            cm.FEZ_cos_fs_din[f,4,b] = cm.FEZ_cos_din[f,4,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,4,b] = cm.FEZ_sin_din[f,4,b]*Delta*wref2
            cm.FEZ_fs_din[f,4,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,4,b],2) + np.power(cm.FEZ_sin_fs_din[f,4,b],2))
            cm.Phase_FEZ_fs_din[f,4,b] = cm.Phase_FEZ_din[f,4,b]
            ## MX
            cm.MEX_sin_fs_din[f,4,b] = cm.MEX_sin30_150_din[f,3,b]*Ix*wref2*numWave
            cm.MEX_cos_fs_din[f,4,b] = cm.MEX_cos30_150_din[f,3,b]*Ix*wref2*numWave
            cm.MEX_fs_din[f,4,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,4,b],2)+ np.power(cm.MEX_sin_fs_din[f,4,b],2))
            cm.Phase_MEX_fs_din[f,4,b] = cm.Phase_MEX30_150_din[f,3,b]
            ## MY
            cm.MEY_sin_fs_din[f,4,b] = cm.MEY_sin_din[f,4,b]*Iy*wref2*numWave
            cm.MEY_cos_fs_din[f,4,b] = cm.MEY_cos_din[f,4,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,4,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,4,b],2)+ np.power(cm.MEY_sin_fs_din[f,4,b],2))
            cm.Phase_MEY_fs_din[f,4,b] = cm.Phase_MEY_din[f,4,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,4,b] = cm.MEZ_sin30_150_din[f,3,b]*Iz*wref2*numWave
            cm.MEZ_cos_fs_din[f,4,b] = cm.MEZ_cos30_150_din[f,3,b]*Iz*wref2*numWave
            cm.MEZ_fs_din[f,4,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,4,b],2) + np.power(cm.MEZ_sin_fs_din[f,4,b],2))
            cm.Phase_MEZ_fs_din[f,4,b] = cm.Phase_MEZ30_150_din[f,3,b]
            ### forces, moments and phases in 150 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,5,b] = cm.FEX_cos120_180_din[f,1,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,5,b] = cm.FEX_sin120_180_din[f,1,b]*Delta*wref2
            cm.FEX_fs_din[f,5,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,5,b],2)+ np.power(cm.FEX_sin_fs_din[f,5,b],2))
            cm.Phase_FEX_fs_din[f,5,b] = cm.Phase_FEX120_180_din[f,1,b]
            ## FY
            cm.FEY_cos_fs_din[f,5,b] = cm.FEY_cos30_150_din[f,4,b]*Delta*wref2
            cm.FEY_sin_fs_din[f,5,b] = cm.FEY_sin30_150_din[f,4,b]*Delta*wref2
            cm.FEY_fs_din[f,5,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,5,b],2) + np.power(cm.FEY_sin_fs_din[f,5,b],2))
            cm.Phase_FEY_fs_din[f,5,b] = cm.Phase_FEY30_150_din[f,4,b]
            ## FZ
            cm.FEZ_cos_fs_din[f,5,b] = cm.FEZ_cos_din[f,5,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,5,b] = cm.FEZ_sin_din[f,5,b]*Delta*wref2
            cm.FEZ_fs_din[f,5,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,5,b],2) + np.power(cm.FEZ_sin_fs_din[f,5,b],2))
            cm.Phase_FEZ_fs_din[f,5,b] = cm.Phase_FEZ_din[f,5,b]
            ## MX
            cm.MEX_sin_fs_din[f,5,b] = cm.MEX_sin30_150_din[f,4,b]*Ix*wref2*numWave
            cm.MEX_cos_fs_din[f,5,b] = cm.MEX_cos30_150_din[f,4,b]*Ix*wref2*numWave
            cm.MEX_fs_din[f,5,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,5,b],2)+ np.power(cm.MEX_sin_fs_din[f,5,b],2))
            cm.Phase_MEX_fs_din[f,5,b] = cm.Phase_MEX30_150_din[f,4,b]
            ## MY
            cm.MEY_sin_fs_din[f,5,b] = cm.MEY_sin_din[f,5,b]*Iy*wref2*numWave
            cm.MEY_cos_fs_din[f,5,b] = cm.MEY_cos_din[f,5,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,5,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,5,b],2)+ np.power(cm.MEY_sin_fs_din[f,5,b],2))
            cm.Phase_MEY_fs_din[f,5,b] = cm.Phase_MEY_din[f,5,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,5,b] = cm.MEZ_sin30_150_din[f,4,b]*Iz*wref2*numWave
            cm.MEZ_cos_fs_din[f,5,b] = cm.MEZ_cos30_150_din[f,4,b]*Iz*wref2*numWave
            cm.MEZ_fs_din[f,5,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,5,b],2) + np.power(cm.MEZ_sin_fs_din[f,5,b],2))
            cm.Phase_MEZ_fs_din[f,5,b] = cm.Phase_MEZ30_150_din[f,4,b]
            ## forces, moments and phases in 180 deg heading
            ## FX
            cm.FEX_cos_fs_din[f,6,b] = cm.FEX_cos120_180_din[f,2,b]*Delta*wref2
            cm.FEX_sin_fs_din[f,6,b] = cm.FEX_sin120_180_din[f,2,b]*Delta*wref2
            cm.FEX_fs_din[f,6,b] = np.sqrt(np.power(cm.FEX_cos_fs_din[f,6,b],2)+ np.power(cm.FEX_sin_fs_din[f,6,b],2))
            cm.Phase_FEX_fs_din[f,6,b] = cm.Phase_FEX120_180_din[f,2,b]
            ## FY
            cm.FEY_cos_fs_din[f,6,b] = 0.0
            cm.FEY_sin_fs_din[f,6,b] = 0.0
            cm.FEY_fs_din[f,6,b] = np.sqrt(np.power(cm.FEY_cos_fs_din[f,6,b],2) + np.power(cm.FEY_sin_fs_din[f,6,b],2))
            cm.Phase_FEY_fs_din[f,6,b] = 0.0
            ## FZ
            cm.FEZ_cos_fs_din[f,6,b] = cm.FEZ_cos_din[f,6,b]*Delta*wref2
            cm.FEZ_sin_fs_din[f,6,b] = cm.FEZ_sin_din[f,6,b]*Delta*wref2
            cm.FEZ_fs_din[f,6,b] = np.sqrt(np.power(cm.FEZ_cos_fs_din[f,6,b],2) + np.power(cm.FEZ_sin_fs_din[f,6,b],2))
            cm.Phase_FEZ_fs_din[f,6,b] = cm.Phase_FEZ_din[f,6,b]
            ## MX
            cm.MEX_sin_fs_din[f,6,b] = 0.0
            cm.MEX_cos_fs_din[f,6,b] = 0.0
            cm.MEX_fs_din[f,6,b] = np.sqrt(np.power(cm.MEX_cos_fs_din[f,6,b],2)+ np.power(cm.MEX_sin_fs_din[f,6,b],2))
            cm.Phase_MEX_fs_din[f,6,b] = 0.0
            ## MY
            cm.MEY_sin_fs_din[f,6,b] = cm.MEY_sin_din[f,6,b]*Iy*wref2*numWave
            cm.MEY_cos_fs_din[f,6,b] = cm.MEY_cos_din[f,6,b]*Iy*wref2*numWave
            cm.MEY_fs_din[f,6,b] = np.sqrt(np.power(cm.MEY_cos_fs_din[f,6,b],2)+ np.power(cm.MEY_sin_fs_din[f,6,b],2))
            cm.Phase_MEY_fs_din[f,6,b] = cm.Phase_MEY_din[f,6,b]
            ## MZ
            cm.MEZ_sin_fs_din[f,6,b] = 0.0
            cm.MEZ_cos_fs_din[f,6,b] = 0.0
            cm.MEZ_fs_din[f,6,b] = np.sqrt(np.power(cm.MEZ_cos_fs_din[f,6,b],2) + np.power(cm.MEZ_sin_fs_din[f,6,b],2))
            cm.Phase_MEZ_fs_din[f,6,b] = 0.0
    
    # storing frequencies in full scale
    cm.fq2 = fq

# RESTORATION CALCULATION
def restoration_calc(sinputs):
    # si = ships inputs for prediction
    ns = np.shape(sinputs)
    nships = ns[0]
    nheads = len(cm.sea_var.angles)
    #sinputs.to_numpy()

    grav = cm.sea_var.grav
    rho = cm.sea_var.rho
    Delta = cm.Displaz

    # common parameters
    #Li =(nships); Bi = (nships); Xbi = (nships); CFi = (nships); GMTi = (nships); GMLi = (nships)
    Li = sinputs.iloc[:,0];  Bi = sinputs.iloc[:,1];   Xbi = sinputs.iloc[:,7]
    CFi = sinputs.iloc[:,4]; GMTi = sinputs.iloc[:,12]; GMLi = sinputs.iloc[:,13]

    ## hydrostatic restoration in full scale
    for b in range(nships):
        for i in range(nheads):

            cm.K33_fs[:,i,b] = CFi[b]*Bi[b]*Li[b]*rho*grav
            cm.K44_fs[:,i,b] = grav*Delta[b]*GMTi[b]
            cm.K55_fs[:,i,b] = grav*Delta[b]*GMLi[b]
            
            ## ASSUMING FLOTATION CENTER IS EQUAL TO CENTER OF BOUYANCY
            ## ASSUMING CENTRE OF GRAVITY IS IN 0,0,0
            #cm.K35_fs[f,b] = -1.0*cm.K33_fs[f,b]*(0.5*Li[b]-Xbi[b])
            #cm.K53_fs[f,b] = -1.0*cm.K33_fs[f,b]*(0.5*Li[b]-Xbi[b])

            cm.K46_fs[:,i,b] = grav*Delta[b]*(0.5*Li[b] - Xbi[b])
            cm.K64_fs[:,i,b] = grav*Delta[b]*(0.5*Li[b] - Xbi[b])

def traslate_vector(pointA, pointB, vector):
    r = np.zeros(3)
    r[0] = pointA[0] - pointB[0]
    r[1] = pointA[1] - pointB[1]
    r[2] = pointA[2] - pointB[2]

    vector_tras = np.cross(r,vector)

    return vector_tras

