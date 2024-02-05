import os
import sys
import datetime
import shutil
import numpy as np
import common as cm
import predict as prd
import seakeeping as sk
import save as sv

################################################################################################
#                                             MAIN PROGRAM                                     #
################################################################################################
# instance of variables
workfiles = cm.workfiles()
var = cm.sea_var()

workdir = workfiles.workdirectory
nfolder = workfiles.nfolder
angles = var.angles
nfreqs = var.nfreqs
nheads = var.nheads
resfolder = workfiles.resfolder
gravity = var.grav
density = var.rho

# log filesid
lfile = str(workfiles.lfile)
ifile = str(workfiles.ifile)
f = open(lfile, 'a')
#var = os.path.exists(file)

init_dt = datetime.datetime.now() # current date
f.write(str(init_dt))

# read input file
# inputs: L, B, T, Cb, Cf, Cm, CC, Xb, Zb, XG, YG, ZG, GMT, GML, Fn, beta, Mod, Dir, Tp, Hs, Xp, Yp, Zp
shipinputs = cm.read_inputs(ifile,lfile)
# size of input file
ns = np.shape(shipinputs)
nships = ns[0] # number of total ships
ninputs = ns[1] # number of inputs

if ninputs != 23:
     sys.exit("Inputs are different of 23:  L, B, T, Cb, Cf, Cm, CC, Xb,\
              Zb, XG, YG, ZG, GMT, GML, Fn, beta, Mod, Dir, Tp, Hs, Xp, Yp, Zp")
     
if nships == 0:
    sys.exit("There is not inputs to evaluate")

## storing the inputs in two separated inputs clases those 
## with Fn equal to zero and other with Fn greater zero
count1 = 0
count2 = 0

for i in range(0,nships):
    fn = shipinputs.iloc[i,14]
    if fn == 0:
        cm.sea_var.list_fn_0.append(i)
        count1 += 1
    else:
        cm.sea_var.list_fn.append(i)
        count2 += 1

shipinputs1 = np.zeros((count1,23)) # ships with Fn equal to zero
shipinputs2 = np.zeros((count2,23)) # ships with Fn greater than zero

for i in range(0,count1):
    j = cm.sea_var.list_fn_0[i]
    shipinputs1[i,:] = shipinputs.iloc[j,:]

for i in range(0,count2):
    j = cm.sea_var.list_fn[i]
    shipinputs2[i,:] = shipinputs.iloc[j,:]

# store wave variables
cm.Mod = shipinputs.iloc[:,16]
cm.Dir = shipinputs.iloc[:,17]
cm.Tp = shipinputs.iloc[:,18]
cm.Hs = shipinputs.iloc[:,19]

# resize variables
## Fn greater than zero
if count1 == 0:
    cm.resize_AM_D(1)
    cm.resize_K(count2)
    cm.resize_forces(1)
    cm.resize_other(1,nships)
    cm.resize_RAO(count2)
    cm.resize_seakeeping(count2)
    
    cm.Point_CDG[:,0] = shipinputs.iloc[:,9]
    cm.Point_CDG[:,1] = shipinputs.iloc[:,10]
    cm.Point_CDG[:,2] = shipinputs.iloc[:,11]

    cm.Point_sk[:,0] = shipinputs.iloc[:,20]
    cm.Point_sk[:,1] = shipinputs.iloc[:,21]
    cm.Point_sk[:,2] = shipinputs.iloc[:,22]
    

## Fn equal to zero
elif count2 == 0:
    cm.resize_AM_D(0)
    cm.resize_K(count1)
    cm.resize_forces(0)
    cm.resize_other(0,nships)
    cm.resize_RAO(count1)
    cm.resize_seakeeping(count1)

    cm.Point_CDG[:,0] = shipinputs.iloc[:,9]
    cm.Point_CDG[:,1] = shipinputs.iloc[:,10]
    cm.Point_CDG[:,2] = shipinputs.iloc[:,11]

    cm.Point_sk[:,0] = shipinputs.iloc[:,20]
    cm.Point_sk[:,1] = shipinputs.iloc[:,21]
    cm.Point_sk[:,2] = shipinputs.iloc[:,22]

## cases with Fn = 0 and Fn >0
else:
    cm.resize_AM_D(2)
    cm.resize_K(nships)
    cm.resize_forces(2)
    cm.resize_other(2,nships)
    cm.resize_RAO(nships)
    cm.resize_seakeeping(nships)

    cm.Point_CDG[:,0] = shipinputs.iloc[:,9]
    cm.Point_CDG[:,1] = shipinputs.iloc[:,10]
    cm.Point_CDG[:,2] = shipinputs.iloc[:,11]

    cm.Point_sk[:,0] = shipinputs.iloc[:,20]
    cm.Point_sk[:,1] = shipinputs.iloc[:,21]
    cm.Point_sk[:,2] = shipinputs.iloc[:,22]

# displacement calculations
f.write("Ship displacement calculation... \n")
prd.displacements(shipinputs)

# hydrostatic restoring calculation
f.write("Hydrostatic restoring forces... \n")
prd.restoration_calc(shipinputs)

## Fn greater than zero
if count1 == 0:
    
    # convert full scale to dimensionless
    f.write("Scaling data... \n")
    si = prd.convert_dim(shipinputs2,2)

    # load ANN models
    f.write("Loading ANNs models... \n")
    models = prd.load_models_seakeeping2(nfolder)

    # make preditions with ANNS (dimensionless)
    f.write("Making predictions... \n")
    prd.prediction_seakeeping2(si,models)

    # full scale predictions
    f.write("Converting to full scale... \n")
    prd.full_scale_hydrodynamic_param2(si)

    # RAO calculations
    f.write("RAO curves computation... \n")
    sk.RAO_calculation2(si)

    # Wave spectrum calculation
    f.write("Wave spectrum computation... \n")
    sk.Wave_sp(nships)
    Wave_spectrum = cm.Wave_sp

    # Movement spectrum calculation
    f.write("Movement spectrum computation... \n")
    surge_rao = cm.RAO_11; sway_rao = cm.RAO_22; heave_rao = cm.RAO_33
    roll_rao = cm.RAO_44; pitch_rao = cm.RAO_55; yaw_rao = cm.RAO_66

elif count2 == 0:

    # load models for Fn equal to zero
    f.write("Loading ANNs models... \n")
    models = prd.load_models_seakeeping1(nfolder)

    # convert full scale to dimensionless
    f.write("Scaling data... \n")
    si = prd.convert_dim(shipinputs1,1)

    # make preditions with ANNS
    f.write("Making predictions... \n")
    prd.prediction_seakeeping1(si,models)

    # full scale predictions
    f.write("Converting to full scale... \n")
    prd.full_scale_hydrodynamic_param1(si)
    
    # RAO calculations
    f.write("RAO curves computation... \n")
    sk.RAO_calculation1(si)

    # Wave spectrum calculation
    f.write("Wave spectrum computation... \n")
    sk.Wave_sp(nships)
    Wave_spectrum = cm.Wave_sp

    # Movement spectrum calculation
    f.write("Movement spectrum computation... \n")
    surge_rao = cm.RAO_11; sway_rao = cm.RAO_22; heave_rao = cm.RAO_33
    roll_rao = cm.RAO_44; pitch_rao = cm.RAO_55; yaw_rao = cm.RAO_66

else:
    # convert full scale to dimensionless
    f.write("Scaling data... \n")
    si1 = prd.convert_dim(shipinputs1,1)
    si2 = prd.convert_dim(shipinputs2,2)

    f.write("Loading ANNs models... \n")
    # load models for Fn greater than zero
    models1 = prd.load_models_seakeeping1(nfolder)
    # load models for Fn greater than zero
    models2 = prd.load_models_seakeeping2(nfolder)

    # make preditions with ANNS
    f.write("Making predictions... \n")
    prd.prediction_seakeeping1(si1,models1)
    prd.prediction_seakeeping2(si2,models2)

    # full scale predictions
    f.write("Converting to full scale... \n")
    prd.full_scale_hydrodynamic_param1(si1)
    prd.full_scale_hydrodynamic_param2(si2)
    
    # RAO calculations
    f.write("RAO curves computation... \n")
    sk.RAO_calculation1(si1)
    sk.RAO_calculation2(si2)

    # Movement spectrum calculation
    f.write("Movement spectrum computation... \n")
    surge_rao = cm.RAO_11; sway_rao = cm.RAO_22; heave_rao = cm.RAO_33
    roll_rao = cm.RAO_44; pitch_rao = cm.RAO_55; yaw_rao = cm.RAO_66

    # Wave spectrum calculation
    f.write("Wave spectrum computation... \n")
    sk.Wave_sp(nships)
    Wave_spectrum = cm.Wave_sp

sm_surge = sk.spectrum_mov(Wave_spectrum,surge_rao,nheads,nships,nfreqs)
sm_sway = sk.spectrum_mov(Wave_spectrum,sway_rao,nheads,nships,nfreqs)
sm_heave = sk.spectrum_mov(Wave_spectrum,heave_rao,nheads,nships,nfreqs)
sm_roll = sk.spectrum_mov(Wave_spectrum,roll_rao,nheads,nships,nfreqs)
sm_pitch = sk.spectrum_mov(Wave_spectrum,pitch_rao,nheads,nships,nfreqs)
sm_yaw = sk.spectrum_mov(Wave_spectrum,yaw_rao,nheads,nships,nfreqs)

# Spectral moments computations
f.write("Spectral moments of zero order computation... \n")
cm.m0[0,:,:] = sk.spectral_momemts(0,sm_surge,nheads,nships,nfreqs)
cm.m0[1,:,:] = sk.spectral_momemts(0,sm_sway,nheads,nships,nfreqs)
cm.m0[2,:,:] = sk.spectral_momemts(0,sm_heave,nheads,nships,nfreqs)
cm.m0[3,:,:] = sk.spectral_momemts(0,sm_roll,nheads,nships,nfreqs)
cm.m0[4,:,:] = sk.spectral_momemts(0,sm_pitch,nheads,nships,nfreqs)
cm.m0[5,:,:] = sk.spectral_momemts(0,sm_yaw,nheads,nships,nfreqs)

f.write("Spectral moments of second order computation... \n")
cm.m2[0,:,:] = sk.spectral_momemts(2,sm_surge,nheads,nships,nfreqs)
cm.m2[1,:,:] = sk.spectral_momemts(2,sm_sway,nheads,nships,nfreqs)
cm.m2[2,:,:] = sk.spectral_momemts(2,sm_heave,nheads,nships,nfreqs)
cm.m2[3,:,:] = sk.spectral_momemts(2,sm_roll,nheads,nships,nfreqs)
cm.m2[4,:,:] = sk.spectral_momemts(2,sm_pitch,nheads,nships,nfreqs)
cm.m2[5,:,:] = sk.spectral_momemts(2,sm_yaw,nheads,nships,nfreqs)

f.write("Spectral moments of fourth order computation... \n")
cm.m4[0,:,:] = sk.spectral_momemts(4,sm_surge,nheads,nships,nfreqs)
cm.m4[1,:,:] = sk.spectral_momemts(4,sm_sway,nheads,nships,nfreqs)
cm.m4[2,:,:] = sk.spectral_momemts(4,sm_heave,nheads,nships,nfreqs)
cm.m4[3,:,:] = sk.spectral_momemts(4,sm_roll,nheads,nships,nfreqs)
cm.m4[4,:,:] = sk.spectral_momemts(4,sm_pitch,nheads,nships,nfreqs)
cm.m4[5,:,:] = sk.spectral_momemts(4,sm_yaw,nheads,nships,nfreqs)

# Seakeeping magnitudes computations: acceleration, movements, SMI & SM, at CDG.
f.write("Significant and maximum movement computation at CDG... \n")
sk.mov_max_sig(nships,nheads,cm.m0,cm.m2)

f.write("RMS acceleration computation at CDG... \n")
sk.acceleration_RMS(nheads,nships,cm.m4)

f.write("Subjetive magnitude computation at CDG... \n")
sk.calculate_SM(nheads,nships,cm.m4,cm.m2)

f.write("Motion Sickness incidence computation at CDG... \n")
sk.calculate_MSI(nheads,nships,cm.m2,cm.m4)

# save results
f.write("Saving results... \n")
if os.path.exists(resfolder):
    print("Old results file are removing \n")
    shutil.rmtree(resfolder)
    os.mkdir(resfolder)
else:
    os.mkdir(resfolder)

sv.s_AM(resfolder,nships) # added masses
sv.s_D(resfolder,nships) # dampings
sv.s_FE(resfolder,nships) # forces
sv.s_K(resfolder,nships) # restoring hydrostatic forces
sv.s_RAO(resfolder,nships) # RAO curves
sv.s_spectral_wave(resfolder,nships) # wave spectrums
sv.s_spectral_moments(resfolder,nships) # spectral moments
sv.s_seakeeping(resfolder,nships) # seakeeping magnitudes

# store results in a folder
#sv.zipfiles(resfolder,workdir) !!!!!!!!!!!!!!!!!!!!
f.write("End calculation\n")
end_dt = datetime.datetime.now() # current date
f.write(str(end_dt))
f.close()