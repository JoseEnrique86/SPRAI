import numpy as np
import common as cm
import math
#import scipy
#from scipy.stats import norm

###############################################################################################
#                                    SEAKEEPING CALCULATION                                   #
###############################################################################################

## matrix traslation to one point to another
def traslate_matrix(point_a,point_b,Matrix):
    # point_a: point where the matrix is calculated. (X,Y,Z)
    # point_b: point where the matrix will be traslated. (X,Y,Z)
    # Matrix: 6X6 to be traslated
    # | aoo  ao1 |
    # | a1o  a11 |

    sz = tuple((3,3))
    Rot = np.zeros(sz)
    szz = tuple((6,6))
    M_tras = np.zeros(szz)
    aoo = Matrix[:3,:3]; ao1 = Matrix[:3,3:]
    a1o = Matrix[3:,:3]; a11 = Matrix[3:,3:]

    rx = point_b[0] - point_a[0]; ry = point_b[1] - point_a[1]; rz = point_b[2] - point_a[2]
    #|0   rz   -ry|
    #|-rz  0    rx|
    #|ry  -rx   0 |
    Rot[0,1] = rz;  Rot[0,2] = -ry
    Rot[1,0] = -rz; Rot[1,2] = rx
    Rot[2,0] = ry;  Rot[2,1] = -rx

    #|  Aoo          Ao1-AooR       |
    #| A1o+RAoo  a11-A1oR+RAo1-RAooR|
    M_tras[:3,:3] = aoo
    M_tras[:3,3:] = ao1 - np.matmul(aoo,Rot)
    M_tras[3:,:3] = a1o + np.matmul(Rot,aoo)
    #M_tras[3:,:3] = a1o - np.matmul(Rot,aoo)
    a = np.matmul(Rot,aoo)

    M_tras[3:,3:] = a11 - np.matmul(a1o,Rot) + np.matmul(Rot,ao1) - np.matmul(a,Rot)
    #M_tras[3:,3:] = a11 + np.matmul(a1o,Rot) - np.matmul(Rot,ao1) + np.matmul(a,Rot)

    return M_tras

## traslation forces to point where RAO will be computed
def traslate_forces(point_a,point_b,force):
    # traslation forces |Fo         |
    #                   |Mo + Fo X R!
    # force: vector (6X1)

    rx = point_b[0] - point_a[0]; ry = point_b[1] - point_a[1]; rz = point_b[2] - point_a[2]
    fx = force[0,0]; fy = force[1,0]; fz = force[2,0]
    mx = force[3,0]; my = force[4,0]; mz = force[5,0]
    
    force[3,0] = mx - (fy*rz - fz*ry); force[4,0] = my - (fz*rx - fx*rz); force[5,0] = mz - (fx*ry - fy*rx)

    return force

## traslation RAOS to another point
#def traslate_RAOS(point_a,point_b,RAO):
#    
#    rx = point_b[0] - point_a[0]; ry = point_b[1] - point_a[1]; rz = point_b[2] - point_a[2]
#   alpha = np.cos(RAO[3,0]); beta = np.cos(RAO[4,0]); gamma = np.cos(RAO[5,0])
#
#    RAO[0,0] += (beta*rz - gamma*ry); RAO[1,0] += (gamma*rx - alpha*rz); RAO[2,0] += (alpha*ry - beta*rx)
#
#    return RAO

def traslate_RAOS_complex(point_a,point_b,RAO):
    # displacement vector
    rx = point_b[0] - point_a[0]; ry = point_b[1] - point_a[1]; rz = point_b[2] - point_a[2]
   
    rao_tans = RAO[0:3,0]
    rao_rot = RAO[3:6,0]
   
    r = np.array([[rx], [ry], [rz]])
 
    rotational = np.cross(rao_rot.flatten(),r.flatten())
 
    rao_traslated = rao_tans + rotational
    rao_new = np.zeros_like(RAO, dtype=complex)
 
    rao_new[0:3,0] = rao_traslated
    rao_new[3:6,0] = rao_rot
    return rao_new

def traslate_RAOS(point_a, point_b, RAO):
    r = np.array(point_b) - np.array(point_a)
    
    rao_trans = RAO[0:3, 0]  # surge, sway, heave
    rao_rot = RAO[3:6, 0]    # roll, pitch, yaw

    rotacional = np.cross(rao_rot, r)

    rao_translated = rao_trans + rotacional
    
    RAO_new = np.copy(RAO)
    RAO_new[0:3, 0] = rao_translated

    return RAO_new


###############################################################################################

### RAO CALCULATION WHEN FN IS EQUAL TO ZERO
def RAO_calculation1(ipt):

    # ship inputs for RAO calculation
    grav = cm.sea_var.grav
    Displaz = cm.Displaz ## all ships
    fq = cm.fq1 ## specific frequencies for Fn equal to zero
    scale = cm.scale1

    angles = cm.sea_var.angles
    nheads = len(angles)
    frequencies = cm.sea_var.frequencies
    nfreqs = len(frequencies)
    ns = np.shape(ipt)
    nships = ns[0]

    sz = tuple((6,6))
    vc = tuple((6,1))
    glob = tuple((12,12))
    fglob = tuple((12,1))

    # matrix definitions
    Mass = np.zeros(sz)
    Added_Mass = np.zeros(sz)
    Damp = np.zeros(sz)
    Stiffness = np.zeros(sz)
    Global = np.zeros(glob)
    forces_global = np.zeros(fglob)
    f_s = np.zeros(vc)
    f_c = np.zeros(vc)
    Global_1 = np.zeros(sz)
    Global_2 = np.zeros(sz)
    Ampl_s = np.zeros(vc)
    Ampl_c = np.zeros(vc)
    global_ampl = np.zeros(fglob)
    Amplitude = np.zeros(vc)

    #  assemble matrices and calculation
    for b in range(nships):
        # list contains position of entries with fn = 0
        index = cm.sea_var.list_fn_0[b]
        # reset variables
        Added_Mass.fill(0)
        Damp.fill(0)
        Stiffness.fill(0)
        Mass.fill(0)
        ## Ship caracteristic full scale
        scale_i = scale[b]
        Lenght = scale_i*ipt[b,0]
        Beam = scale_i*ipt[b,1]
        Draught = scale_i*ipt[b,2]
        Delta = Displaz[index]
        Xbi = scale_i*ipt[b,7]
        Zbi = Draught*ipt[b,8]

        I_xx = Delta*np.power((Beam/4),2)
        I_yy = Delta*np.power((Lenght/4),2)
        I_zz = Delta*np.power(0.3*np.sqrt(np.power(Beam,2) + np.power(Lenght,2)),2)

        # ships inertias
        Mass[0,0] = Delta; Mass[1,1] = Delta; Mass[2,2] = Delta
        Mass[3,3] = I_xx;  Mass[4,4] = I_yy;  Mass[5,5] = I_zz

        for i in range(nheads):
            for f in range(nfreqs):
                Added_Mass.fill(0)
                Damp.fill(0)
                Stiffness.fill(0)
                # Added masses matrix for each frequency
                Added_Mass[0,0] = cm.AM11_fs[f,i,index] ; Added_Mass[0,2] = cm.AM13_fs[f,i,index] ; Added_Mass[0,4] = cm.AM15_fs[f,i,index]
                Added_Mass[1,1] = cm.AM22_fs[f,i,index] ; Added_Mass[1,3] = cm.AM24_fs[f,i,index] ; Added_Mass[1,5] = cm.AM26_fs[f,i,index]
                Added_Mass[2,0] = cm.AM31_fs[f,i,index] ; Added_Mass[2,2] = cm.AM33_fs[f,i,index] ; Added_Mass[2,4] = cm.AM35_fs[f,i,index]
                Added_Mass[3,1] = cm.AM42_fs[f,i,index] ; Added_Mass[3,3] = cm.AM44_fs[f,i,index] ; Added_Mass[3,5] = cm.AM46_fs[f,i,index]
                Added_Mass[4,0] = cm.AM51_fs[f,i,index] ; Added_Mass[4,2] = cm.AM53_fs[f,i,index] ; Added_Mass[4,4] = cm.AM55_fs[f,i,index]
                Added_Mass[5,1] = cm.AM62_fs[f,i,index] ; Added_Mass[5,3] = cm.AM64_fs[f,i,index] ; Added_Mass[5,5] = cm.AM66_fs[f,i,index]
                
                # Damping matrix for each frequency
                Damp[0,0] = cm.D11_fs[f,i,index] ; Damp[0,2] = cm.D13_fs[f,i,index] ; Damp[0,4] = cm.D15_fs[f,i,index]
                Damp[1,1] = cm.D22_fs[f,i,index] ; Damp[1,3] = cm.D24_fs[f,i,index] ; Damp[1,5] = cm.D26_fs[f,i,index]
                Damp[2,0] = cm.D31_fs[f,i,index] ; Damp[2,2] = cm.D33_fs[f,i,index] ; Damp[2,4] = cm.D35_fs[f,i,index]
                Damp[3,1] = cm.D42_fs[f,i,index] ; Damp[3,3] = cm.D44_fs[f,i,index] ; Damp[3,5] = cm.D46_fs[f,i,index]
                Damp[4,0] = cm.D51_fs[f,i,index] ; Damp[4,2] = cm.D53_fs[f,i,index] ; Damp[4,4] = cm.D55_fs[f,i,index]
                Damp[5,1] = cm.D62_fs[f,i,index] ; Damp[5,3] = cm.D64_fs[f,i,index] ; Damp[5,5] = cm.D66_fs[f,i,index]
                
                # ships hydrostatic restoration
                Stiffness[2,2] = cm.K33_fs[f,i,index] ; Stiffness[3,3] = cm.K44_fs[f,i,index] ; Stiffness[4,4] = cm.K55_fs[f,i,index]
                Stiffness[2,4] = cm.K35_fs[f,i,index] ; Stiffness[3,5] = cm.K46_fs[f,i,index]
                Stiffness[4,2] = cm.K53_fs[f,i,index] ; Stiffness[5,3] = cm.K64_fs[f,i,index]
                
                Added_Mass_i = np.zeros((6,6))
                Added_Mass_i = Added_Mass

                Damp_i = np.zeros((6,6))
                Damp_i = Damp
              
                ## traslation from center of gravity to center of buoyancy
                point_a = [0.0,0.0,0.0]
                point_a[0] = cm.Point_CDG[index,0]
                point_a[1] = cm.Point_CDG[index,1]
                point_a[2] = cm.Point_CDG[index,2]
                #point_b = [Xbi-0.5*Lenght,0.0,Zbi]
                point_b = [Xbi,0.0,Zbi]
                
                Stiffness_i = np.zeros((6,6))
                Stiffness_i = traslate_matrix(point_a,point_b,Stiffness)
                Mass_i = np.zeros((6,6))
                Mass_i = traslate_matrix(point_a,point_b,Mass)

                ## composing vector with force component
                f_s[0] = cm.FEX_sin_fs[f,i,index]; f_s[1] = cm.FEY_sin_fs[f,i,index]; f_s[2] = cm.FEZ_sin_fs[f,i,index]; f_s[3] = cm.MEX_sin_fs[f,i,index]; f_s[4] = cm.MEY_sin_fs[f,i,index]; f_s[5] = cm.MEZ_sin_fs[f,i,index]
                f_c[0] = cm.FEX_cos_fs[f,i,index]; f_c[1] = cm.FEY_cos_fs[f,i,index]; f_c[2] = cm.FEZ_cos_fs[f,i,index]; f_c[3] = cm.MEX_cos_fs[f,i,index]; f_c[4] = cm.MEY_cos_fs[f,i,index]; f_c[5] = cm.MEZ_cos_fs[f,i,index]
                
                ## traslation from 0,0,0 (reference frame in the database calculation) to center of buoyancy
                point_a = [0.0,0.0,0.0]
                point_b = [Xbi-0.5*Lenght,0.0,Zbi]

                f_s = traslate_forces(point_a,point_b,f_s)
                f_c = traslate_forces(point_a,point_b,f_c)

                ## composing vector with total forces
                forces_global.fill(0) # reset variable
                forces_global[:6] = f_s
                forces_global[6:] = f_c

                Global_1.fill(0); Global_2.fill(0)

                Global_1 = -1.0*fq[f,b]*fq[f,b]*(Mass_i + Added_Mass_i) + Stiffness_i
                Global_2 =  fq[f,b]*Damp_i

                # | -w^2M+k    -wD  |
                # |    wD   -w^2M+k |
                Global.fill(0) # reset variable
                Global[:6,:6] = Global_1
                Global[6:,6:] = Global_1
                Global[:6,6:] = -1*Global_2
                Global[6:,:6] = Global_2
                    
                ## calculation of each component of amplitudes
                global_ampl.fill(0) # reset variable
                global_ampl = np.linalg.solve(Global,forces_global)
                #global_ampl = np.linalg.inv(Global).dot(forces_global)

                Ampl_s = global_ampl[:6]
                Ampl_c = global_ampl[6:]

                # Traslate RAOS from center of buoyancy to calculation point
                point_a = [Xbi,0.0,Zbi]
                point_b[0] = cm.Point_sk[index,0]
                point_b[1] = cm.Point_sk[index,1]
                point_b[2] = cm.Point_sk[index,2]

                Amplitude_complex = np.zeros(vc,dtype=complex)
                Amplitude_complex = Ampl_c + 1j*Ampl_s
                # Traslation
                Amplitude_complex_tras = traslate_RAOS_complex(point_a,point_b,Amplitude_complex)
                
                Amplitude.fill(0) # reset variable
                Amplitude = np.abs(Amplitude_complex_tras)
                Phase = np.angle(Amplitude_complex_tras)

                ## not used old function
                ## Ampl_s_tras = traslate_RAOS(point_a,point_b,Ampl_s)
                ## Ampl_c_tras = traslate_RAOS(point_a,point_b,Ampl_c)
                ## Phase = np.arctan(Ampl_s_tras/Ampl_c_tras)
                ## Amplitude = np.sqrt(np.power(Ampl_s_tras,2) + np.power(Ampl_c_tras,2))

                wavenumber = fq[f,b]*fq[f,b]/grav

                ## RAOs and phases at XP, YP, ZP
                cm.RAO_11[f,i,index] = Amplitude[0]; cm.RAO_phase_11[f,i,index] = Phase[0]
                cm.RAO_22[f,i,index] = Amplitude[1]; cm.RAO_phase_22[f,i,index] = Phase[1]
                cm.RAO_33[f,i,index] = Amplitude[2]; cm.RAO_phase_33[f,i,index] = Phase[2]
                cm.RAO_44[f,i,index] = Amplitude[3]; cm.RAO_phase_44[f,i,index] = Phase[3]
                cm.RAO_55[f,i,index] = Amplitude[4]; cm.RAO_phase_55[f,i,index] = Phase[4]
                cm.RAO_66[f,i,index] = Amplitude[5]; cm.RAO_phase_66[f,i,index] = Phase[5]

                cm.RAO_44_rep[f,i,index] = Amplitude[3]/wavenumber
                cm.RAO_55_rep[f,i,index] = Amplitude[4]/wavenumber
                cm.RAO_66_rep[f,i,index] = Amplitude[5]/wavenumber

                ## store the matrices traslated from its origin of coordinates to 0,0,0
                point_a = [Xbi-0.5*Lenght,0.0,Zbi]
                point_b = [0.0,0.0,0.0]
                point_c = [0.0,0.0,0.0]
                point_c[0] = cm.Point_CDG[index,0]
                point_c[1] = cm.Point_CDG[index,1]
                point_c[2] = cm.Point_CDG[index,2]

                Stiffness = traslate_matrix(point_c,point_b,Stiffness)
                Added_Mass = traslate_matrix(point_a,point_b,Added_Mass)
                Damp = traslate_matrix(point_a,point_b,Damp)

                cm.AM11_fs[f,i,index] = Added_Mass[0,0] ; cm.AM13_fs[f,i,index] = Added_Mass[0,2] ; cm.AM15_fs[f,i,index] = Added_Mass[0,4] 
                cm.AM22_fs[f,i,index] = Added_Mass[1,1] ; cm.AM24_fs[f,i,index] = Added_Mass[1,3] ; cm.AM26_fs[f,i,index] = Added_Mass[1,5] 
                cm.AM31_fs[f,i,index] = Added_Mass[2,0] ; cm.AM33_fs[f,i,index] = Added_Mass[2,2] ; cm.AM35_fs[f,i,index] = Added_Mass[2,4]
                cm.AM42_fs[f,i,index] = Added_Mass[3,1] ; cm.AM44_fs[f,i,index] = Added_Mass[3,3] ; cm.AM46_fs[f,i,index] = Added_Mass[3,5]
                cm.AM51_fs[f,i,index] = Added_Mass[4,0] ; cm.AM53_fs[f,i,index] = Added_Mass[4,2] ; cm.AM55_fs[f,i,index] = Added_Mass[4,4]
                cm.AM62_fs[f,i,index] = Added_Mass[5,1] ; cm.AM64_fs[f,i,index] = Added_Mass[5,3] ; cm.AM66_fs[f,i,index] = Added_Mass[5,5]
                
                cm.D11_fs[f,i,index] = Damp[0,0] ; cm.D13_fs[f,i,index] = Damp[0,2] ; cm.D15_fs[f,i,index] = Damp[0,4]
                cm.D22_fs[f,i,index] = Damp[1,1] ; cm.D24_fs[f,i,index] = Damp[1,3] ; cm.D26_fs[f,i,index] = Damp[1,5]
                cm.D31_fs[f,i,index] = Damp[2,0] ; cm.D33_fs[f,i,index] = Damp[2,2] ; cm.D35_fs[f,i,index] = Damp[2,4]
                cm.D42_fs[f,i,index] = Damp[3,1] ; cm.D44_fs[f,i,index] = Damp[3,3] ; cm.D46_fs[f,i,index] = Damp[3,5]
                cm.D51_fs[f,i,index] = Damp[4,0] ; cm.D53_fs[f,i,index] = Damp[4,2] ; cm.D55_fs[f,i,index] = Damp[4,4]
                cm.D62_fs[f,i,index] = Damp[5,1] ; cm.D64_fs[f,i,index] = Damp[5,3] ; cm.D66_fs[f,i,index] = Damp[5,5]
                
                cm.K33_fs[f,i,index] = Stiffness[2,2] ; cm.K44_fs[f,i,index] = Stiffness[3,3]; cm.K55_fs[f,i,index] = Stiffness[4,4]
                cm.K35_fs[f,i,index] = Stiffness[2,4] ; cm.K46_fs[f,i,index] = Stiffness[3,5]
                cm.K53_fs[f,i,index] = Stiffness[4,2] ; cm.K64_fs[f,i,index] = Stiffness[5,3]

## RAO CALCULATION WHEN FN IS GREATER THAN ZERO
def RAO_calculation2(ipt):
    # ship inputs for RAO calculation
    grav = cm.sea_var.grav
    Displaz = cm.Displaz
    fq = cm.fq2
    scale = cm.scale2

    angles = cm.sea_var.angles
    nheads = len(angles)
    frequencies = cm.sea_var.frequencies2
    nfreqs = len(frequencies)
    ns = np.shape(ipt)
    nships = ns[0]

    sz = tuple((6,6))
    vc = tuple((6,1))
    glob = tuple((12,12))
    fglob = tuple((12,1))

    # matrix definitions
    Mass = np.zeros(sz)
    Added_Mass = np.zeros(sz)
    Damp = np.zeros(sz)
    Stiffness = np.zeros(sz)
    Global = np.zeros(glob)
    forces_global = np.zeros(fglob)
    f_s = np.zeros(vc)
    f_c = np.zeros(vc)
    Global_1 = np.zeros(sz)
    Global_2 = np.zeros(sz)
    Ampl_s = np.zeros(vc)
    Ampl_c = np.zeros(vc)
    global_ampl = np.zeros(fglob)
    Amplitude = np.zeros(vc)

    #  assemble matrices and calculation
    for b in range(nships):
       # list contains position of entries with fn = 0
        index = cm.sea_var.list_fn[b]
        # reset variables
        Added_Mass.fill(0)
        Damp.fill(0)
        Stiffness.fill(0)
        Mass.fill(0)
        ## Ship caracteristic full scale
        scale_i = scale[b]
        Lenght = scale_i*ipt[b,0]
        Beam = scale_i*ipt[b,1]
        Draught = scale_i*ipt[b,2]
        Delta = Displaz[index]
        Xbi = scale_i*ipt[b,7]
        Zbi = Draught*ipt[b,8]

        I_xx = Delta*np.power((Beam/4),2)
        I_yy = Delta*np.power((Lenght/4),2)
        I_zz = Delta*np.power(0.3*np.sqrt(np.power(Beam,2) + np.power(Lenght,2)),2)

        # ships inertias
        Mass[0,0] = Delta; Mass[1,1] = Delta; Mass[2,2] = Delta
        Mass[3,3] = I_xx;  Mass[4,4] = I_yy;  Mass[5,5] = I_zz

        for i in range(nheads):
            for f in range(nfreqs):
                
                if angles[i] == 0:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_0[f,index] ; Added_Mass[0,2] = cm.AM13_fs_0[f,index] ; Added_Mass[0,4] = cm.AM15_fs_0[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_0[f,index] ; Added_Mass[1,3] = cm.AM24_fs_0[f,index] ; Added_Mass[1,5] = cm.AM26_fs_0[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_0[f,index] ; Added_Mass[2,2] = cm.AM33_fs_0[f,index] ; Added_Mass[2,4] = cm.AM35_fs_0[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_0[f,index] ; Added_Mass[3,3] = cm.AM44_fs_0[f,index] ; Added_Mass[3,5] = cm.AM46_fs_0[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_0[f,index] ; Added_Mass[4,2] = cm.AM53_fs_0[f,index] ; Added_Mass[4,4] = cm.AM55_fs_0[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_0[f,index] ; Added_Mass[5,3] = cm.AM64_fs_0[f,index] ; Added_Mass[5,5] = cm.AM66_fs_0[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_0[f,index] ; Damp[0,2] = cm.D13_fs_0[f,index] ; Damp[0,4] = cm.D15_fs_0[f,index]
                    Damp[1,1] = cm.D22_fs_0[f,index] ; Damp[1,3] = cm.D24_fs_0[f,index] ; Damp[1,5] = cm.D26_fs_0[f,index]
                    Damp[2,0] = cm.D31_fs_0[f,index] ; Damp[2,2] = cm.D33_fs_0[f,index] ; Damp[2,4] = cm.D35_fs_0[f,index]
                    Damp[3,1] = cm.D42_fs_0[f,index] ; Damp[3,3] = cm.D44_fs_0[f,index] ; Damp[3,5] = cm.D46_fs_0[f,index]
                    Damp[4,0] = cm.D51_fs_0[f,index] ; Damp[4,2] = cm.D53_fs_0[f,index] ; Damp[4,4] = cm.D55_fs_0[f,index]
                    Damp[5,1] = cm.D62_fs_0[f,index] ; Damp[5,3] = cm.D64_fs_0[f,index] ; Damp[5,5] = cm.D66_fs_0[f,index]
                
                if angles[i] == 30:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_30[f,index] ; Added_Mass[0,2] = cm.AM13_fs_30[f,index] ; Added_Mass[0,4] = cm.AM15_fs_30[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_30[f,index] ; Added_Mass[1,3] = cm.AM24_fs_30[f,index] ; Added_Mass[1,5] = cm.AM26_fs_30[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_30[f,index] ; Added_Mass[2,2] = cm.AM33_fs_30[f,index] ; Added_Mass[2,4] = cm.AM35_fs_30[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_30[f,index] ; Added_Mass[3,3] = cm.AM44_fs_30[f,index] ; Added_Mass[3,5] = cm.AM46_fs_30[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_30[f,index] ; Added_Mass[4,2] = cm.AM53_fs_30[f,index] ; Added_Mass[4,4] = cm.AM55_fs_30[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_30[f,index] ; Added_Mass[5,3] = cm.AM64_fs_30[f,index] ; Added_Mass[5,5] = cm.AM66_fs_30[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_30[f,index] ; Damp[0,2] = cm.D13_fs_30[f,index] ; Damp[0,4] = cm.D15_fs_30[f,index]
                    Damp[1,1] = cm.D22_fs_30[f,index] ; Damp[1,3] = cm.D24_fs_30[f,index] ; Damp[1,5] = cm.D26_fs_30[f,index]
                    Damp[2,0] = cm.D31_fs_30[f,index] ; Damp[2,2] = cm.D33_fs_30[f,index] ; Damp[2,4] = cm.D35_fs_30[f,index]
                    Damp[3,1] = cm.D42_fs_30[f,index] ; Damp[3,3] = cm.D44_fs_30[f,index] ; Damp[3,5] = cm.D46_fs_30[f,index]
                    Damp[4,0] = cm.D51_fs_30[f,index] ; Damp[4,2] = cm.D53_fs_30[f,index] ; Damp[4,4] = cm.D55_fs_30[f,index]
                    Damp[5,1] = cm.D62_fs_30[f,index] ; Damp[5,3] = cm.D64_fs_30[f,index] ; Damp[5,5] = cm.D66_fs_30[f,index]
                
                if angles[i] == 60:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_60[f,index] ; Added_Mass[0,2] = cm.AM13_fs_60[f,index] ; Added_Mass[0,4] = cm.AM15_fs_60[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_60[f,index] ; Added_Mass[1,3] = cm.AM24_fs_60[f,index] ; Added_Mass[1,5] = cm.AM26_fs_60[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_60[f,index] ; Added_Mass[2,2] = cm.AM33_fs_60[f,index] ; Added_Mass[2,4] = cm.AM35_fs_60[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_60[f,index] ; Added_Mass[3,3] = cm.AM44_fs_60[f,index] ; Added_Mass[3,5] = cm.AM46_fs_60[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_60[f,index] ; Added_Mass[4,2] = cm.AM53_fs_60[f,index] ; Added_Mass[4,4] = cm.AM55_fs_60[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_60[f,index] ; Added_Mass[5,3] = cm.AM64_fs_60[f,index] ; Added_Mass[5,5] = cm.AM66_fs_60[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_60[f,index] ; Damp[0,2] = cm.D13_fs_60[f,index] ; Damp[0,4] = cm.D15_fs_60[f,index]
                    Damp[1,1] = cm.D22_fs_60[f,index] ; Damp[1,3] = cm.D24_fs_60[f,index] ; Damp[1,5] = cm.D26_fs_60[f,index]
                    Damp[2,0] = cm.D31_fs_60[f,index] ; Damp[2,2] = cm.D33_fs_60[f,index] ; Damp[2,4] = cm.D35_fs_60[f,index]
                    Damp[3,1] = cm.D42_fs_60[f,index] ; Damp[3,3] = cm.D44_fs_60[f,index] ; Damp[3,5] = cm.D46_fs_60[f,index]
                    Damp[4,0] = cm.D51_fs_60[f,index] ; Damp[4,2] = cm.D53_fs_60[f,index] ; Damp[4,4] = cm.D55_fs_60[f,index]
                    Damp[5,1] = cm.D62_fs_60[f,index] ; Damp[5,3] = cm.D64_fs_60[f,index] ; Damp[5,5] = cm.D66_fs_60[f,index]

                if angles[i] == 90:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_90[f,index] ; Added_Mass[0,2] = cm.AM13_fs_90[f,index] ; Added_Mass[0,4] = cm.AM15_fs_90[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_90[f,index] ; Added_Mass[1,3] = cm.AM24_fs_90[f,index] ; Added_Mass[1,5] = cm.AM26_fs_90[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_90[f,index] ; Added_Mass[2,2] = cm.AM33_fs_90[f,index] ; Added_Mass[2,4] = cm.AM35_fs_90[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_90[f,index] ; Added_Mass[3,3] = cm.AM44_fs_90[f,index] ; Added_Mass[3,5] = cm.AM46_fs_90[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_90[f,index] ; Added_Mass[4,2] = cm.AM53_fs_90[f,index] ; Added_Mass[4,4] = cm.AM55_fs_90[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_90[f,index] ; Added_Mass[5,3] = cm.AM64_fs_90[f,index] ; Added_Mass[5,5] = cm.AM66_fs_90[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_90[f,index] ; Damp[0,2] = cm.D13_fs_90[f,index] ; Damp[0,4] = cm.D15_fs_90[f,index]
                    Damp[1,1] = cm.D22_fs_90[f,index] ; Damp[1,3] = cm.D24_fs_90[f,index] ; Damp[1,5] = cm.D26_fs_90[f,index]
                    Damp[2,0] = cm.D31_fs_90[f,index] ; Damp[2,2] = cm.D33_fs_90[f,index] ; Damp[2,4] = cm.D35_fs_90[f,index]
                    Damp[3,1] = cm.D42_fs_90[f,index] ; Damp[3,3] = cm.D44_fs_90[f,index] ; Damp[3,5] = cm.D46_fs_90[f,index]
                    Damp[4,0] = cm.D51_fs_90[f,index] ; Damp[4,2] = cm.D53_fs_90[f,index] ; Damp[4,4] = cm.D55_fs_90[f,index]
                    Damp[5,1] = cm.D62_fs_90[f,index] ; Damp[5,3] = cm.D64_fs_90[f,index] ; Damp[5,5] = cm.D66_fs_90[f,index]

                if angles[i] == 120:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_120[f,index] ; Added_Mass[0,2] = cm.AM13_fs_120[f,index] ; Added_Mass[0,4] = cm.AM15_fs_120[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_120[f,index] ; Added_Mass[1,3] = cm.AM24_fs_120[f,index] ; Added_Mass[1,5] = cm.AM26_fs_120[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_120[f,index] ; Added_Mass[2,2] = cm.AM33_fs_120[f,index] ; Added_Mass[2,4] = cm.AM35_fs_120[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_120[f,index] ; Added_Mass[3,3] = cm.AM44_fs_120[f,index] ; Added_Mass[3,5] = cm.AM46_fs_120[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_120[f,index] ; Added_Mass[4,2] = cm.AM53_fs_120[f,index] ; Added_Mass[4,4] = cm.AM55_fs_120[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_120[f,index] ; Added_Mass[5,3] = cm.AM64_fs_120[f,index] ; Added_Mass[5,5] = cm.AM66_fs_120[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_120[f,index] ; Damp[0,2] = cm.D13_fs_120[f,index] ; Damp[0,4] = cm.D15_fs_120[f,index]
                    Damp[1,1] = cm.D22_fs_120[f,index] ; Damp[1,3] = cm.D24_fs_120[f,index] ; Damp[1,5] = cm.D26_fs_120[f,index]
                    Damp[2,0] = cm.D31_fs_120[f,index] ; Damp[2,2] = cm.D33_fs_120[f,index] ; Damp[2,4] = cm.D35_fs_120[f,index]
                    Damp[3,1] = cm.D42_fs_120[f,index] ; Damp[3,3] = cm.D44_fs_120[f,index] ; Damp[3,5] = cm.D46_fs_120[f,index]
                    Damp[4,0] = cm.D51_fs_120[f,index] ; Damp[4,2] = cm.D53_fs_120[f,index] ; Damp[4,4] = cm.D55_fs_120[f,index]
                    Damp[5,1] = cm.D62_fs_120[f,index] ; Damp[5,3] = cm.D64_fs_120[f,index] ; Damp[5,5] = cm.D66_fs_120[f,index]
                
                if angles[i] == 150:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_150[f,index] ; Added_Mass[0,2] = cm.AM13_fs_150[f,index] ; Added_Mass[0,4] = cm.AM15_fs_150[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_150[f,index] ; Added_Mass[1,3] = cm.AM24_fs_150[f,index] ; Added_Mass[1,5] = cm.AM26_fs_150[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_150[f,index] ; Added_Mass[2,2] = cm.AM33_fs_150[f,index] ; Added_Mass[2,4] = cm.AM35_fs_150[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_150[f,index] ; Added_Mass[3,3] = cm.AM44_fs_150[f,index] ; Added_Mass[3,5] = cm.AM46_fs_150[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_150[f,index] ; Added_Mass[4,2] = cm.AM53_fs_150[f,index] ; Added_Mass[4,4] = cm.AM55_fs_150[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_150[f,index] ; Added_Mass[5,3] = cm.AM64_fs_150[f,index] ; Added_Mass[5,5] = cm.AM66_fs_150[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_150[f,index] ; Damp[0,2] = cm.D13_fs_150[f,index] ; Damp[0,4] = cm.D15_fs_150[f,index]
                    Damp[1,1] = cm.D22_fs_150[f,index] ; Damp[1,3] = cm.D24_fs_150[f,index] ; Damp[1,5] = cm.D26_fs_150[f,index]
                    Damp[2,0] = cm.D31_fs_150[f,index] ; Damp[2,2] = cm.D33_fs_150[f,index] ; Damp[2,4] = cm.D35_fs_150[f,index]
                    Damp[3,1] = cm.D42_fs_150[f,index] ; Damp[3,3] = cm.D44_fs_150[f,index] ; Damp[3,5] = cm.D46_fs_150[f,index]
                    Damp[4,0] = cm.D51_fs_150[f,index] ; Damp[4,2] = cm.D53_fs_150[f,index] ; Damp[4,4] = cm.D55_fs_150[f,index]
                    Damp[5,1] = cm.D62_fs_150[f,index] ; Damp[5,3] = cm.D64_fs_150[f,index] ; Damp[5,5] = cm.D66_fs_150[f,index]

                if angles[i] == 180:
                    # Added masses matrix for each frequency
                    Added_Mass[0,0] = cm.AM11_fs_180[f,index] ; Added_Mass[0,2] = cm.AM13_fs_180[f,index] ; Added_Mass[0,4] = cm.AM15_fs_180[f,index]
                    Added_Mass[1,1] = cm.AM22_fs_180[f,index] ; Added_Mass[1,3] = cm.AM24_fs_180[f,index] ; Added_Mass[1,5] = cm.AM26_fs_180[f,index]
                    Added_Mass[2,0] = cm.AM31_fs_180[f,index] ; Added_Mass[2,2] = cm.AM33_fs_180[f,index] ; Added_Mass[2,4] = cm.AM35_fs_180[f,index]
                    Added_Mass[3,1] = cm.AM42_fs_180[f,index] ; Added_Mass[3,3] = cm.AM44_fs_180[f,index] ; Added_Mass[3,5] = cm.AM46_fs_180[f,index]
                    Added_Mass[4,0] = cm.AM51_fs_180[f,index] ; Added_Mass[4,2] = cm.AM53_fs_180[f,index] ; Added_Mass[4,4] = cm.AM55_fs_180[f,index]
                    Added_Mass[5,1] = cm.AM62_fs_180[f,index] ; Added_Mass[5,3] = cm.AM64_fs_180[f,index] ; Added_Mass[5,5] = cm.AM66_fs_180[f,index]
                            
                    # Damping matrix for each frequency
                    Damp[0,0] = cm.D11_fs_180[f,index] ; Damp[0,2] = cm.D13_fs_180[f,index] ; Damp[0,4] = cm.D15_fs_180[f,index]
                    Damp[1,1] = cm.D22_fs_180[f,index] ; Damp[1,3] = cm.D24_fs_180[f,index] ; Damp[1,5] = cm.D26_fs_180[f,index]
                    Damp[2,0] = cm.D31_fs_180[f,index] ; Damp[2,2] = cm.D33_fs_180[f,index] ; Damp[2,4] = cm.D35_fs_180[f,index]
                    Damp[3,1] = cm.D42_fs_180[f,index] ; Damp[3,3] = cm.D44_fs_180[f,index] ; Damp[3,5] = cm.D46_fs_180[f,index]
                    Damp[4,0] = cm.D51_fs_180[f,index] ; Damp[4,2] = cm.D53_fs_180[f,index] ; Damp[4,4] = cm.D55_fs_180[f,index]
                    Damp[5,1] = cm.D62_fs_180[f,index] ; Damp[5,3] = cm.D64_fs_180[f,index] ; Damp[5,5] = cm.D66_fs_180[f,index]

                # ships hydrostatic restoration
                Stiffness[2,2] = cm.K33_fs[f,i,index] ; Stiffness[3,3] = cm.K44_fs[f,i,index] ; Stiffness[4,4] = cm.K55_fs[f,i,index]
                Stiffness[2,4] = cm.K35_fs[f,i,index] ; Stiffness[3,5] = cm.K46_fs[f,i,index]
                Stiffness[4,2] = cm.K53_fs[f,i,index] ; Stiffness[5,3] = cm.K64_fs[f,i,index]
                
                Added_Mass_i = np.zeros((6,6))
                Added_Mass_i = Added_Mass

                Damp_i = np.zeros((6,6))
                Damp_i = Damp
              
                ## traslation from center of gravity to center of buoyancy
                point_a = [0.0,0.0,0.0]
                point_a[0] = cm.Point_CDG[index,0]
                point_a[1] = cm.Point_CDG[index,1]
                point_a[2] = cm.Point_CDG[index,2]
                # bouyancy center
                point_b = [Xbi,0.0,Zbi]

                Stiffness_i = np.zeros((6,6))
                Stiffness_i = traslate_matrix(point_a,point_b,Stiffness)
                Mass_i = np.zeros((6,6))
                Mass_i = traslate_matrix(point_a,point_b,Mass)
                
                ## composing vector with force component
                f_s[0] = cm.FEX_sin_fs_din[f,i,index]; f_s[1] = cm.FEY_sin_fs_din[f,i,index]; f_s[2] = cm.FEZ_sin_fs_din[f,i,index]; f_s[3] = cm.MEX_sin_fs_din[f,i,index]; f_s[4] = cm.MEY_sin_fs_din[f,i,index]; f_s[5] = cm.MEZ_sin_fs_din[f,i,index]
                f_c[0] = cm.FEX_cos_fs_din[f,i,index]; f_c[1] = cm.FEY_cos_fs_din[f,i,index]; f_c[2] = cm.FEZ_cos_fs_din[f,i,index]; f_c[3] = cm.MEX_cos_fs_din[f,i,index]; f_c[4] = cm.MEY_cos_fs_din[f,i,index]; f_c[5] = cm.MEZ_cos_fs_din[f,i,index]

                ## traslation from 0,0,0 to center of buoyancy
                point_a = [0.0,0.0,0.0]
                point_b = [Xbi-0.5*Lenght,0.0,Zbi]
                f_s = traslate_forces(point_a,point_b,f_s)
                f_c = traslate_forces(point_a,point_b,f_c)

                ## composing vector with total forces
                forces_global.fill(0) # reset 
                forces_global[:6] = f_s
                forces_global[6:] = f_c

                #fn = ipt[b,14] # Froude number
                #beta = angles[i]
                #v = fn*np.sqrt(grav)
                #w = fq[f,b]
                #we = w*(1-(v*w/grav)*np.cos(beta*2*np.pi/360))
                
                Global_1.fill(0); Global_2.fill(0) # reset variable
                Global_1 = -1.0*w*w*(Mass_i + Added_Mass_i) + Stiffness_i
                Global_2 =  w*Damp_i

                # | -w^2M+k    -wD  |
                # |    wD   -w^2M+k |
                Global.fill(0) # reset variable
                Global[:6,:6] = Global_1
                Global[6:,6:] = Global_1
                Global[:6,6:] = -1*Global_2
                Global[6:,:6] = Global_2

                ## calculation of each component of amplitudes
                global_ampl.fill(0) # reset variable
                global_ampl = np.linalg.solve(Global,forces_global)

                Ampl_s = global_ampl[:6]
                Ampl_c = global_ampl[6:]

                # Traslate RAOS from center of buoyancy to calculation point
                point_a = [Xbi,0.0,Zbi]
                point_b[0] = cm.Point_sk[index,0]
                point_b[1] = cm.Point_sk[index,1]
                point_b[2] = cm.Point_sk[index,2]

                Amplitude_complex = np.zeros(vc,dtype=complex)
                Amplitude_complex = Ampl_c + 1j*Ampl_s
                
                # Traslation
                Amplitude_complex_tras = traslate_RAOS_complex(point_a,point_b,Amplitude_complex)
                Amplitude.fill(0) # reset variable
                Amplitude = np.abs(Amplitude_complex_tras)
                Phase = np.angle(Amplitude_complex_tras)

                ## not used old function
                ## Ampl_s_tras = traslate_RAOS(point_a,point_b,Ampl_s)
                ## Ampl_c_tras = traslate_RAOS(point_a,point_b,Ampl_c)
                ## Phase = np.arctan(Ampl_s_tras/Ampl_c_tras)
                ## Amplitude = np.sqrt(np.power(Ampl_c,2)+ np.power(Ampl_s,2))
                
                wavenumber = w*w/grav

                cm.RAO_11[f,i,index] = Amplitude[0]; cm.RAO_phase_11[f,i,index] = Phase[0]
                cm.RAO_22[f,i,index] = Amplitude[1]; cm.RAO_phase_22[f,i,index] = Phase[1]
                cm.RAO_33[f,i,index] = Amplitude[2]; cm.RAO_phase_33[f,i,index] = Phase[2]
                cm.RAO_44[f,i,index] = Amplitude[3]; cm.RAO_phase_44[f,i,index] = Phase[3]
                cm.RAO_55[f,i,index] = Amplitude[4]; cm.RAO_phase_55[f,i,index] = Phase[4]
                cm.RAO_66[f,i,index] = Amplitude[5]; cm.RAO_phase_66[f,i,index] = Phase[5]

                cm.RAO_44_rep[f,i,index] = Amplitude[3]/wavenumber
                cm.RAO_55_rep[f,i,index] = Amplitude[4]/wavenumber
                cm.RAO_66_rep[f,i,index] = Amplitude[5]/wavenumber

                ## store the matrices traslated from its origin of coordinates to 0,0,0
                point_a = [Xbi-0.5*Lenght,0.0,Zbi]
                point_b = [0.0,0.0,0.0]
                point_b[0] = cm.Point_sk[index,0]
                point_b[1] = cm.Point_sk[index,1]
                point_b[2] = cm.Point_sk[index,2]
                #point_c = [0.0,0.0,0.0]
                #point_c[0] = cm.Point_CDG[index,0]
                #point_c[1] = cm.Point_CDG[index,1]
                #point_c[2] = cm.Point_CDG[index,2]

                #Stiffness = traslate_matrix(point_c,point_b,Stiffness)
                Added_Mass = traslate_matrix(point_a,point_b,Added_Mass)
                Damp = traslate_matrix(point_a,point_b,Damp)

                ## composing vector with force component
                f_s[0] = cm.FEX_sin_fs_din[f,i,index]; f_s[1] = cm.FEY_sin_fs_din[f,i,index]; f_s[2] = cm.FEZ_sin_fs_din[f,i,index]; f_s[3] = cm.MEX_sin_fs_din[f,i,index]; f_s[4] = cm.MEY_sin_fs_din[f,i,index]; f_s[5] = cm.MEZ_sin_fs_din[f,i,index]
                f_c[0] = cm.FEX_cos_fs_din[f,i,index]; f_c[1] = cm.FEY_cos_fs_din[f,i,index]; f_c[2] = cm.FEZ_cos_fs_din[f,i,index]; f_c[3] = cm.MEX_cos_fs_din[f,i,index]; f_c[4] = cm.MEY_cos_fs_din[f,i,index]; f_c[5] = cm.MEZ_cos_fs_din[f,i,index]
                ## traslate forces from 0,0,0 to calculation point
                point_a = [0,0,0]
                f_s = traslate_forces(point_a,point_b,f_s)
                f_c = traslate_forces(point_a,point_b,f_c)

                # store the module of traslated forces
                cm.FEX_fs_din[f,i,index] = np.sqrt(np.power(f_s[0],2)+ np.power(f_c[0],2))
                cm.FEY_fs_din[f,i,index] = np.sqrt(np.power(f_s[1],2)+ np.power(f_c[1],2))
                cm.FEZ_fs_din[f,i,index] = np.sqrt(np.power(f_s[2],2)+ np.power(f_c[2],2))
                cm.MEX_fs_din[f,i,index] = np.sqrt(np.power(f_s[3],2)+ np.power(f_c[3],2))
                cm.MEY_fs_din[f,i,index] = np.sqrt(np.power(f_s[4],2)+ np.power(f_c[4],2))
                cm.MEZ_fs_din[f,i,index] = np.sqrt(np.power(f_s[5],2)+ np.power(f_c[5],2))

                if angles[i] == 0:
                    cm.AM11_fs_0[f,index] = Added_Mass[0,0]; cm.AM13_fs_0[f,index] = Added_Mass[0,2]; cm.AM15_fs_0[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_0[f,index] = Added_Mass[1,1]; cm.AM24_fs_0[f,index] = Added_Mass[1,3]; cm.AM26_fs_0[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_0[f,index] = Added_Mass[2,0]; cm.AM33_fs_0[f,index] = Added_Mass[2,2]; cm.AM35_fs_0[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_0[f,index] = Added_Mass[3,1]; cm.AM44_fs_0[f,index] = Added_Mass[3,3]; cm.AM46_fs_0[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_0[f,index] = Added_Mass[4,0]; cm.AM53_fs_0[f,index] = Added_Mass[4,2]; cm.AM55_fs_0[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_0[f,index] = Damp[0,0]; cm.D13_fs_0[f,index] = Damp[0,2]; cm.D15_fs_0[f,index] = Damp[0,4]
                    cm.D22_fs_0[f,index] = Damp[1,1]; cm.D24_fs_0[f,index] = Damp[1,3]; cm.D26_fs_0[f,index] = Damp[0,4]
                    cm.D31_fs_0[f,index] = Damp[2,0]; cm.D33_fs_0[f,index] = Damp[2,2]; cm.D35_fs_0[f,index] = Damp[2,4]
                    cm.D42_fs_0[f,index] = Damp[3,1]; cm.D44_fs_0[f,index] = Damp[3,3]; cm.D46_fs_0[f,index] = Damp[3,5]
                    cm.D51_fs_0[f,index] = Damp[4,0]; cm.D53_fs_0[f,index] = Damp[4,2]; cm.D55_fs_0[f,index] = Damp[4,4]
                    cm.D62_fs_0[f,index] = Damp[5,1]; cm.D64_fs_0[f,index] = Damp[5,3]; cm.D66_fs_0[f,index] = Damp[5,5]
                
                if angles[i] == 30:
                    cm.AM11_fs_30[f,index] = Added_Mass[0,0]; cm.AM13_fs_30[f,index] = Added_Mass[0,2]; cm.AM15_fs_30[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_30[f,index] = Added_Mass[1,1]; cm.AM24_fs_30[f,index] = Added_Mass[1,3]; cm.AM26_fs_30[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_30[f,index] = Added_Mass[2,0]; cm.AM33_fs_30[f,index] = Added_Mass[2,2]; cm.AM35_fs_30[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_30[f,index] = Added_Mass[3,1]; cm.AM44_fs_30[f,index] = Added_Mass[3,3]; cm.AM46_fs_30[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_30[f,index] = Added_Mass[4,0]; cm.AM53_fs_30[f,index] = Added_Mass[4,2]; cm.AM55_fs_30[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_30[f,index] = Damp[0,0]; cm.D13_fs_30[f,index] = Damp[0,2]; cm.D15_fs_30[f,index] = Damp[0,4]
                    cm.D22_fs_30[f,index] = Damp[1,1]; cm.D24_fs_30[f,index] = Damp[1,3]; cm.D26_fs_30[f,index] = Damp[0,4]
                    cm.D31_fs_30[f,index] = Damp[2,0]; cm.D33_fs_30[f,index] = Damp[2,2]; cm.D35_fs_30[f,index] = Damp[2,4]
                    cm.D42_fs_30[f,index] = Damp[3,1]; cm.D44_fs_30[f,index] = Damp[3,3]; cm.D46_fs_30[f,index] = Damp[3,5]
                    cm.D51_fs_30[f,index] = Damp[4,0]; cm.D53_fs_30[f,index] = Damp[4,2]; cm.D55_fs_30[f,index] = Damp[4,4]
                    cm.D62_fs_30[f,index] = Damp[5,1]; cm.D64_fs_30[f,index] = Damp[5,3]; cm.D66_fs_30[f,index] = Damp[5,5]
                
                if angles[i] == 60:
                    cm.AM11_fs_60[f,index] = Added_Mass[0,0]; cm.AM13_fs_60[f,index] = Added_Mass[0,2]; cm.AM15_fs_60[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_60[f,index] = Added_Mass[1,1]; cm.AM24_fs_60[f,index] = Added_Mass[1,3]; cm.AM26_fs_60[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_60[f,index] = Added_Mass[2,0]; cm.AM33_fs_60[f,index] = Added_Mass[2,2]; cm.AM35_fs_60[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_60[f,index] = Added_Mass[3,1]; cm.AM44_fs_60[f,index] = Added_Mass[3,3]; cm.AM46_fs_60[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_60[f,index] = Added_Mass[4,0]; cm.AM53_fs_60[f,index] = Added_Mass[4,2]; cm.AM55_fs_60[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_60[f,index] = Damp[0,0]; cm.D13_fs_60[f,index] = Damp[0,2]; cm.D15_fs_60[f,index] = Damp[0,4]
                    cm.D22_fs_60[f,index] = Damp[1,1]; cm.D24_fs_60[f,index] = Damp[1,3]; cm.D26_fs_60[f,index] = Damp[0,4]
                    cm.D31_fs_60[f,index] = Damp[2,0]; cm.D33_fs_60[f,index] = Damp[2,2]; cm.D35_fs_60[f,index] = Damp[2,4]
                    cm.D42_fs_60[f,index] = Damp[3,1]; cm.D44_fs_60[f,index] = Damp[3,3]; cm.D46_fs_60[f,index] = Damp[3,5]
                    cm.D51_fs_60[f,index] = Damp[4,0]; cm.D53_fs_60[f,index] = Damp[4,2]; cm.D55_fs_60[f,index] = Damp[4,4]
                    cm.D62_fs_60[f,index] = Damp[5,1]; cm.D64_fs_60[f,index] = Damp[5,3]; cm.D66_fs_60[f,index] = Damp[5,5]

                if angles[i] == 90:
                    cm.AM11_fs_90[f,index] = Added_Mass[0,0]; cm.AM13_fs_90[f,index] = Added_Mass[0,2]; cm.AM15_fs_90[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_90[f,index] = Added_Mass[1,1]; cm.AM24_fs_90[f,index] = Added_Mass[1,3]; cm.AM26_fs_90[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_90[f,index] = Added_Mass[2,0]; cm.AM33_fs_90[f,index] = Added_Mass[2,2]; cm.AM35_fs_90[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_90[f,index] = Added_Mass[3,1]; cm.AM44_fs_90[f,index] = Added_Mass[3,3]; cm.AM46_fs_90[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_90[f,index] = Added_Mass[4,0]; cm.AM53_fs_90[f,index] = Added_Mass[4,2]; cm.AM55_fs_90[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_90[f,index] = Damp[0,0]; cm.D13_fs_90[f,index] = Damp[0,2]; cm.D15_fs_90[f,index] = Damp[0,4]
                    cm.D22_fs_90[f,index] = Damp[1,1]; cm.D24_fs_90[f,index] = Damp[1,3]; cm.D26_fs_90[f,index] = Damp[0,4]
                    cm.D31_fs_90[f,index] = Damp[2,0]; cm.D33_fs_90[f,index] = Damp[2,2]; cm.D35_fs_90[f,index] = Damp[2,4]
                    cm.D42_fs_90[f,index] = Damp[3,1]; cm.D44_fs_90[f,index] = Damp[3,3]; cm.D46_fs_90[f,index] = Damp[3,5]
                    cm.D51_fs_90[f,index] = Damp[4,0]; cm.D53_fs_90[f,index] = Damp[4,2]; cm.D55_fs_90[f,index] = Damp[4,4]
                    cm.D62_fs_90[f,index] = Damp[5,1]; cm.D64_fs_90[f,index] = Damp[5,3]; cm.D66_fs_90[f,index] = Damp[5,5]

                if angles[i] == 120:
                    cm.AM11_fs_120[f,index] = Added_Mass[0,0]; cm.AM13_fs_120[f,index] = Added_Mass[0,2]; cm.AM15_fs_120[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_120[f,index] = Added_Mass[1,1]; cm.AM24_fs_120[f,index] = Added_Mass[1,3]; cm.AM26_fs_120[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_120[f,index] = Added_Mass[2,0]; cm.AM33_fs_120[f,index] = Added_Mass[2,2]; cm.AM35_fs_120[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_120[f,index] = Added_Mass[3,1]; cm.AM44_fs_120[f,index] = Added_Mass[3,3]; cm.AM46_fs_120[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_120[f,index] = Added_Mass[4,0]; cm.AM53_fs_120[f,index] = Added_Mass[4,2]; cm.AM55_fs_120[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_120[f,index] = Damp[0,0]; cm.D13_fs_120[f,index] = Damp[0,2]; cm.D15_fs_120[f,index] = Damp[0,4]
                    cm.D22_fs_120[f,index] = Damp[1,1]; cm.D24_fs_120[f,index] = Damp[1,3]; cm.D26_fs_120[f,index] = Damp[0,4]
                    cm.D31_fs_120[f,index] = Damp[2,0]; cm.D33_fs_120[f,index] = Damp[2,2]; cm.D35_fs_120[f,index] = Damp[2,4]
                    cm.D42_fs_120[f,index] = Damp[3,1]; cm.D44_fs_120[f,index] = Damp[3,3]; cm.D46_fs_120[f,index] = Damp[3,5]
                    cm.D51_fs_120[f,index] = Damp[4,0]; cm.D53_fs_120[f,index] = Damp[4,2]; cm.D55_fs_120[f,index] = Damp[4,4]
                    cm.D62_fs_120[f,index] = Damp[5,1]; cm.D64_fs_120[f,index] = Damp[5,3]; cm.D66_fs_120[f,index] = Damp[5,5]
                
                if angles[i] == 150:
                    cm.AM11_fs_150[f,index] = Added_Mass[0,0]; cm.AM13_fs_150[f,index] = Added_Mass[0,2]; cm.AM15_fs_150[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_150[f,index] = Added_Mass[1,1]; cm.AM24_fs_150[f,index] = Added_Mass[1,3]; cm.AM26_fs_150[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_150[f,index] = Added_Mass[2,0]; cm.AM33_fs_150[f,index] = Added_Mass[2,2]; cm.AM35_fs_150[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_150[f,index] = Added_Mass[3,1]; cm.AM44_fs_150[f,index] = Added_Mass[3,3]; cm.AM46_fs_150[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_150[f,index] = Added_Mass[4,0]; cm.AM53_fs_150[f,index] = Added_Mass[4,2]; cm.AM55_fs_150[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_150[f,index] = Damp[0,0]; cm.D13_fs_150[f,index] = Damp[0,2]; cm.D15_fs_150[f,index] = Damp[0,4]
                    cm.D22_fs_150[f,index] = Damp[1,1]; cm.D24_fs_150[f,index] = Damp[1,3]; cm.D26_fs_150[f,index] = Damp[0,4]
                    cm.D31_fs_150[f,index] = Damp[2,0]; cm.D33_fs_150[f,index] = Damp[2,2]; cm.D35_fs_150[f,index] = Damp[2,4]
                    cm.D42_fs_150[f,index] = Damp[3,1]; cm.D44_fs_150[f,index] = Damp[3,3]; cm.D46_fs_150[f,index] = Damp[3,5]
                    cm.D51_fs_150[f,index] = Damp[4,0]; cm.D53_fs_150[f,index] = Damp[4,2]; cm.D55_fs_150[f,index] = Damp[4,4]
                    cm.D62_fs_150[f,index] = Damp[5,1]; cm.D64_fs_150[f,index] = Damp[5,3]; cm.D66_fs_150[f,index] = Damp[5,5]

                if angles[i] == 180:
                    cm.AM11_fs_180[f,index] = Added_Mass[0,0]; cm.AM13_fs_180[f,index] = Added_Mass[0,2]; cm.AM15_fs_180[f,index] = Added_Mass[0,4]
                    cm.AM22_fs_180[f,index] = Added_Mass[1,1]; cm.AM24_fs_180[f,index] = Added_Mass[1,3]; cm.AM26_fs_180[f,index] = Added_Mass[1,5]
                    cm.AM31_fs_180[f,index] = Added_Mass[2,0]; cm.AM33_fs_180[f,index] = Added_Mass[2,2]; cm.AM35_fs_180[f,index] = Added_Mass[2,4]
                    cm.AM42_fs_180[f,index] = Added_Mass[3,1]; cm.AM44_fs_180[f,index] = Added_Mass[3,3]; cm.AM46_fs_180[f,index] = Added_Mass[3,5]
                    cm.AM51_fs_180[f,index] = Added_Mass[4,0]; cm.AM53_fs_180[f,index] = Added_Mass[4,2]; cm.AM55_fs_180[f,index] = Added_Mass[4,4]
                            
                    cm.D11_fs_180[f,index] = Damp[0,0]; cm.D13_fs_180[f,index] = Damp[0,2]; cm.D15_fs_180[f,index] = Damp[0,4]
                    cm.D22_fs_180[f,index] = Damp[1,1]; cm.D24_fs_180[f,index] = Damp[1,3]; cm.D26_fs_180[f,index] = Damp[0,4]
                    cm.D31_fs_180[f,index] = Damp[2,0]; cm.D33_fs_180[f,index] = Damp[2,2]; cm.D35_fs_180[f,index] = Damp[2,4]
                    cm.D42_fs_180[f,index] = Damp[3,1]; cm.D44_fs_180[f,index] = Damp[3,3]; cm.D46_fs_180[f,index] = Damp[3,5]
                    cm.D51_fs_180[f,index] = Damp[4,0]; cm.D53_fs_180[f,index] = Damp[4,2]; cm.D55_fs_180[f,index] = Damp[4,4]
                    cm.D62_fs_180[f,index] = Damp[5,1]; cm.D64_fs_180[f,index] = Damp[5,3]; cm.D66_fs_180[f,index] = Damp[5,5]
                
                cm.K33_fs[f,index] = Stiffness[2,2] ; cm.K44_fs[f,index] = Stiffness[3,3]; cm.K55_fs[f,index] = Stiffness[4,4]
                cm.K35_fs[f,index] = Stiffness[2,4] ; cm.K46_fs[f,index] = Stiffness[3,5]
                cm.K53_fs[f,index] = Stiffness[4,2] ; cm.K64_fs[f,index] = Stiffness[5,3]

## Wave spectrum calculation 2D & 3D
def Wave_sp(nships):
    
    # Mod: type of spectrum: 1 Pierson Mostkowiz; 2 JONSWAP
    # Dir: mean wave direction if = -1, no computation of directional spectrum
    # if dir != -1 computation of directional spectrum according to OMI

    nfreqs = len(cm.sea_var.frequencies)
    nheads = len(cm.sea_var.angles)
    angles = cm.sea_var.angles

    # definition of peak parameter
    peak_param = 3.3

    s = tuple((nheads,nships))
    t = tuple((nfreqs,nships))
    # directional function
    d_function = np.zeros(s)

    #create a local matrix
    pm_sp_i = np.zeros(t)
    expo = np.zeros(t)
    jwp_sp_i = np.zeros(t)

    # calculate directional function
    for b in range(nships):
        if cm.Dir[b] != -1:

            for i in range(nheads):
                if abs(cm.Dir[b]-angles[i]) <= (np.pi/2):
                    d_function[i,b] = (2/np.pi)*np.cos(cm.Dir[b]-angles[i])*np.cos(cm.Dir[b]-angles[i])
                else:
                    d_function[i,b] = 0.0

    A_gamma = 1 - 0.287*np.log(peak_param)
    count1 = 0
    count2 = 0
    # calculate spectrum
    for b in range(nships):
        
        # Peak frequency
        wp = 2.0*np.pi/cm.Tp[b]

        # determine what frequency to use
        if b in cm.sea_var.list_fn_0:
            freqs = cm.fq1 # full scale frequencies for each ship with Fn = 0
            t = count1 # auxiliar variable
            count1 += 1
        if b in cm.sea_var.list_fn:
            freqs = cm.fq2 # full scale frequencies for each ship with Fn > 0
            t = count2
            count2 += 1

        for i in range(nheads):
            for f in range(nfreqs):
                
                if freqs[f,t] <= wp:
                    sigma = 0.07
                else:
                    sigma = 0.09

                pm_sp_i[f,b] = (5.0/16)*(cm.Hs[b]**2)*(wp**4)*(freqs[f,t]**(-5))*np.exp((-5.0/4)*(freqs[f,t]/wp)**(-4))
                expo[f,b] = np.exp(-0.5*((freqs[f,t]-wp)/(sigma*wp))**2)
                jwp_sp_i[f,b] = A_gamma*pm_sp_i[f,b]*peak_param**(expo[f,b])

                if cm.Dir[b] != -1:
                    if cm.Mod[b] == 1:
                        cm.Wave_sp[f,i,b] = jwp_sp_i[f,b]*d_function[i,b]
                    else:
                        cm.Wave_sp[f,i,b] = pm_sp_i[f,b]*d_function[i,b]
                else:
                    if cm.Mod[b] == 1:
                        cm.Wave_sp[f,i,b] = jwp_sp_i[f,b]
                    else:
                        cm.Wave_sp[f,i,b] = pm_sp_i[f,b]

## computation of movements spectrum
def spectrum_mov(SPEC,RAO,nheads,nships,nfreqs):

    # SPEC: wave spectrum
    # RAO: response amplitude operator
    # SP: movement spectrum of ships in all headings

    SP = np.zeros((nfreqs,nheads,nships))


    for b in range(nships):
        for i in range(nheads):
            for f in range(nfreqs):
                SP[f,i,b] = RAO[f,i,b]*RAO[f,i,b]*SPEC[f,i,b]
    
    return SP

## computation spectral moments 6 degrees of freedon
def spectral_momemts(order,SP,nheads,nships,nfreqs):
    
    # i moment to calculate
    # SP spectral moment of a ships
    # SP = [nfreqs, nheads, nships] 

    #create a local matrix
    M = np.zeros((nfreqs,nheads,nships))
    sm = np.zeros((nheads,nships))

    count1 = 0
    count2 = 0
    for b in range(nships):

        # determine what frequency to use
        if b in cm.sea_var.list_fn_0:
            freq = cm.fq1 # full scale frequencies for each ship with Fn = 0
            t = count1 # auxiliar variable
            count1 += 1
        if b in cm.sea_var.list_fn:
            freq = cm.fq2 # full scale frequencies for each ship with Fn > 0
            t = count2
            count2 += 1
        
        for i in range(nheads):
            for f in range(nfreqs):

                # multiply each value plus freq^i
                M[f,i,b] = SP[f,i,b]*np.power(freq[f,t],order)

            #initialize inner variable  
            m_i = 0 

            for f in range(nfreqs-1):
                # trapezoid rule to integrate spectralm moment curve 
                m_i += 0.5*abs(M[f,i,b] + M[f+1,i,b])*abs(freq[f,t]-freq[f+1,t])
            
            sm[i,b] = m_i
    
    return sm

## computation of maximum and significant movements
def mov_max_sig(nships,nheads,m0,m2):

    for i in range(nheads):
        for b in range(nships):
            cm.mov_11_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[0,i,b]/m2[0,i,b]))))*np.sqrt(m0[0,i,b])
            cm.mov_22_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[1,i,b]/m2[1,i,b]))))*np.sqrt(m0[1,i,b])
            cm.mov_33_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[2,i,b]/m2[2,i,b]))))*np.sqrt(m0[2,i,b])
            cm.mov_44_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[3,i,b]/m2[3,i,b]))))*np.sqrt(m0[3,i,b])
            cm.mov_55_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[4,i,b]/m2[4,i,b]))))*np.sqrt(m0[4,i,b])
            cm.mov_66_max[i,b] = np.sqrt(2*np.log(10800/(2*np.pi*np.sqrt(m0[5,i,b]/m2[5,i,b]))))*np.sqrt(m0[5,i,b])
            
            cm.mov_11_sig[i,b] = 2*np.sqrt(m0[0,i,b])
            cm.mov_22_sig[i,b] = 2*np.sqrt(m0[1,i,b])
            cm.mov_33_sig[i,b] = 2*np.sqrt(m0[2,i,b])
            cm.mov_44_sig[i,b] = 2*np.sqrt(m0[3,i,b])
            cm.mov_55_sig[i,b] = 2*np.sqrt(m0[4,i,b])
            cm.mov_66_sig[i,b] = 2*np.sqrt(m0[5,i,b])

## funtion for acceleration Root Mean Square 
def acceleration_RMS(nheads,nships,m4):

    for i in range(nheads):
        for b in range(nships):
            cm.acc_RMS_11[i,b] = np.sqrt(m4[0,i,b])
            cm.acc_RMS_22[i,b] = np.sqrt(m4[1,i,b])
            cm.acc_RMS_33[i,b] = np.sqrt(m4[2,i,b])
            cm.acc_RMS_44[i,b] = np.sqrt(m4[3,i,b])
            cm.acc_RMS_55[i,b] = np.sqrt(m4[4,i,b])
            cm.acc_RMS_66[i,b] = np.sqrt(m4[5,i,b])

## funtion for SM (Subjetive Magnitude Parameter)(heave acceleration)
def calculate_SM(nheads,nships,m4,m2):
    grav = cm.sea_var.grav 
    
    for i in range(nheads):
        for b in range(nships):
            we = np.sqrt(m4[2,i,b]/m2[2,i,b])
            Amplitude = (1-(np.exp(-1.65*np.power(we,2))))*(75.6-49.6*np.log(we)+13.5*(np.power(np.log(we),2)))
            acc = m4[2,i,b]
            acc_30 = 2*np.sqrt(m4[2,i,b])
            cm.Significative_mag[i,b] = Amplitude*np.power((acc_30/grav),1.43)

## funtion for MSI after 2 hour (heave acceleration)
def calculate_MSI(nheads,nships,m2,m4):

    grav = cm.sea_var.grav

    for i in range(nheads):
        for b in range(nships):
            we = np.sqrt(m4[2,i,b]/m2[2,i,b])
            fe = we/(2*np.pi)
            mu_MSI = 0.654+3.697*np.log10(fe)+2.32*np.power(np.log10(fe),2)
            S3= 0.798*np.sqrt(m4[2,i,b])

            x=(np.log10(S3/grav)-mu_MSI)/0.4
            # cumulative normal distribution with mean equal to zero, and std equal to one
            #cm.Motion_sickness[i,b] = 100*(0.5+math.erf(x/np.sqrt(2)))
            cm.Motion_sickness[i,b] = 100*phi(x)

## cumulative normal distribution
## Abramowitz and Stegun. Handbook of mathematical functions.
def phi(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)/math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)

    return 0.5*(1.0 + sign*y)

## function for calculate probabiity of water on deck
def water_on_deck(m0,freeboard):
    ## m0: zero order of vertical movement spectrum of the bow
    Pwod = np.exp(-0.5*freeboard*freeboard/m0)

    return Pwod

def number_of_ocurrences_water(m0,m2,Pwod):

    Tp = 1/(2*np.pi)*np.sqrt(m2/m0)
    Now = 3600*Pwod/Tp
    return Now

## fucntion for calculate slamming
def slamming(draught,velocity,m0,m2):
    ## m0: zero order of vertical movement spectrum of the bow
    ## m2: second order of vertical movement spectrum of the bow
    ## draught: ship draught at bow
    ## velocity: thresshold velocity for slamming
    Pos = np.exp(-0.5*draught*draught/m0)*np.exp(-0.5*velocity*velocity/m2)

    return Pos

def number_of_slamming_ocurrences(m0,m2,Pos):

    Tp = 1/(2*np.pi)*np.sqrt(m2/m0)
    Nos = 3600*Pos/Tp
    return Nos

## function for calculate propeller emersion
def propeller_emersion(m0,depth):
    ## m0: zero order of vertical movement spectrum of the propeller
    ## depth: propeller depth

    Pem = np.exp(-0.5*depth*depth/m0)

    return Pem

def number_of_propeller_emergences(m0,m2,Pem):

    Tp = 1/(2*np.pi)*np.sqrt(m2/m0)
    Npem = 3600*Pem/Tp
    return Npem


