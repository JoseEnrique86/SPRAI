
import shutil
import os
import datetime
import pandas as pd
import numpy as np

################################################################################################
#                                     CLASSES DEFINTION                                        #
################################################################################################

class sea_var():
    'Class with global variables to predict ships'
    ## variables:
    rho = 1025.0 # fluid density
    grav = 9.8065 # gravity acceleration 
    angles = [0,30,60,90,120,150,180] # list of headings to be predicted
    ## frequencies for Fn = 0
    frequencies=[35.11069577,30.4150108,26.82716115,23.99646339,21.70611818,19.81488428,18.22680012,
                16.87438526,15.70880383,14.69384131,13.8020746,13.01235689,12.30811929,11.67619565,
                11.10599167,10.58888616,10.11779215,9.686830152,9.291081454,8.926399616,8.589264609,
                8.276668835,7.986027149,7.715105209,7.461961917,7.224902844,7.002442252,6.793271924,
                6.596235422,6.410306694]
    
    ## frequencies for Fn > 0
    #frequencies2=[4.532771333, 4.965402214, 5.551488443, 5.934790783, 6.410306694, 7.022139153,
    #             7.850990247, 8.275670356, 8.777673942, 9.383728155, 10.13558483, 11.10297689,
    #             12.41350553, 14.33388152, 17.55534788]

     ## frequencies for Fn > 0
    frequencies2 = [4.532771333, 4.749086774, 4.965402214, 5.258445329, 5.551488443, 5.743139613,
                    5.934790783, 6.172548739, 6.410306694, 6.716222924, 7.022139153, 7.4365647, 
                    7.850990247, 8.063330302, 8.275670356, 8.526672149, 8.777673942, 9.080701049, 
                    9.383728155, 9.759656493, 10.13558483, 10.61928086, 11.10297689, 11.75824121, 
                    12.41350553, 13.37369353, 14.33388152, 15.9446147, 16.74998129, 17.55534788]

    nheads = len(angles) # number of headind directions
    nfreqs = len(frequencies) # number of frequencies to be predicted Fn = 0
    nfreqs2 = len(frequencies2) # number of frequencies to be predicted Fn > 0

    list_fn_0 = []
    list_fn = []

class workfiles():
    'Class with atributte files'
    workdirectory = os.getcwd()
    lfile = os.path.join(workdirectory,"logfile.log") # log file
    ifile = os.path.join(workdirectory,"input.csv") # ship input file for prediction
    ## B [m] = 0, T [m] = 1, CB [-] = 2, CF [-] = 3, CM [-] = 4, CC [-] = 5, XB [-] = 6, ZB [-] = 7
    ## lambda [-] = 8, GMT_ad [m] = 9, GML_ad [m] = 10, Ixx [m4] = 11 , Iyy [m4] = 12, Izz = 13 [m4]
    wfile = os.path.join(workdirectory,"wave.csv") # wave input file
    ## Hs [m] = 0, Tm [s] = 1, Type of spectrum = 2
    nfolder = os.path.join(workdirectory,"ANN") ## path where the ANN are
    resfolder = os.path.join(workdirectory,"results") # path where the res files are placed

################################################################################################
#                                     FUNCTION DEFINTION                                       #
################################################################################################

def read_inputs(ifile,lfile):
    # reading inputfile with data for ML prediction
    if os.path.exists(ifile) == False:
        print("Input file with ship data does not exists \n")
        print("please create a input file \n")
        f = open(lfile, "a")
        f.write("Fail >> Input input file does not exist. \n")
        f.close()
        os.kill
        exit
    else:
        # reading file
        print("Reading input file... \n")
        f = open(lfile, 'a')
        f.write("Reading input file \n")
        shipinputs = pd.read_csv(ifile) # reading file using pandas functions
        #nships = np.shape(shipinputs) # length of array equal to number of ship to be predicted
        #nships = nships[0]
        f.close()
        return shipinputs

# Create log file
def logfile(filepath):
    f = open(filepath, "w")
    f.write(str(datetime.datetime.now())) # print date
    f.write("\n")
    f.write("Logfile \n")
    f.close()

# Overwrite log file
def oldlogfile(filepath):
    f = open(filepath, "a")
    f.write(str(datetime.datetime.now())) # print date
    f.write("\n")
    f.write("Logfile \n")
    f.close()

# log file to write execution
def log_file(lfile, resfolder):
    if os.path.exists(lfile):
        os.remove(lfile)
        if os.path.exists(resfolder):
            print("There is another case, all folder containing results will be removed")
            shutil.rmtree(resfolder)
    else:
        logfile(lfile)

def resize_AM_D(cof):

    nfreqs = len(sea_var.frequencies)
    nheads = len(sea_var.angles)

    if cof == 0 or cof == 2:
        nships = len(sea_var.list_fn_0)

        ##  resize numpy arrays added mass and damping Fn = 0
        global Added_M, Damp

        t=tuple((nfreqs,18,nships))
        Added_M = np.resize(Added_M, t)
        Damp = np.resize(Damp, t)

        global AM11_fs, AM22_fs, AM33_fs, AM44_fs, AM55_fs, AM66_fs, AM13_fs, AM15_fs, AM62_fs
        global AM24_fs, AM26_fs, AM31_fs, AM35_fs, AM42_fs, AM46_fs, AM51_fs, AM53_fs, AM64_fs
        global D11_fs, D22_fs, D33_fs, D44_fs, D55_fs, D66_fs, D13_fs, D15_fs, D62_fs
        global D24_fs, D26_fs, D31_fs, D35_fs, D42_fs, D46_fs, D51_fs, D53_fs, D64_fs

        #s = tuple((nfreqs,nships))
        s = tuple((nfreqs,nheads,nships))
        AM11_fs = np.resize(AM11_fs, s); D11_fs = np.resize(D11_fs, s)
        AM22_fs = np.resize(AM22_fs, s); D22_fs = np.resize(D22_fs, s)
        AM33_fs = np.resize(AM33_fs, s); D33_fs = np.resize(D33_fs, s)
        AM44_fs = np.resize(AM44_fs, s); D44_fs = np.resize(D44_fs, s)
        AM55_fs = np.resize(AM55_fs, s); D55_fs = np.resize(D55_fs, s)
        AM66_fs = np.resize(AM66_fs, s); D66_fs = np.resize(D66_fs, s)
        AM13_fs = np.resize(AM13_fs, s); D13_fs = np.resize(D13_fs, s)
        AM15_fs = np.resize(AM15_fs, s); D15_fs = np.resize(D15_fs, s)
        AM24_fs = np.resize(AM24_fs, s); D24_fs = np.resize(D24_fs, s)
        AM26_fs = np.resize(AM26_fs, s); D26_fs = np.resize(D26_fs, s)
        AM31_fs = np.resize(AM31_fs, s); D31_fs = np.resize(D31_fs, s)
        AM35_fs = np.resize(AM35_fs, s); D35_fs = np.resize(D35_fs, s)
        AM42_fs = np.resize(AM42_fs, s); D42_fs = np.resize(D42_fs, s)
        AM46_fs = np.resize(AM46_fs, s); D46_fs = np.resize(D46_fs, s)
        AM51_fs = np.resize(AM51_fs, s); D51_fs = np.resize(D51_fs, s)
        AM53_fs = np.resize(AM53_fs, s); D53_fs = np.resize(D53_fs, s)
        AM62_fs = np.resize(AM62_fs, s); D62_fs = np.resize(D62_fs, s)
        AM64_fs = np.resize(AM64_fs, s); D64_fs = np.resize(D64_fs, s)

    if cof == 1 or cof == 2:
        nships = len(sea_var.list_fn)

        ##  resize numpy arrays added mass and damping Fn > 0
        t=tuple((nfreqs,nheads,18,nships))
        s = tuple((nfreqs,nships))
        global Added_M_0_180, Damp_0_180, Added_M_0_180_fs, Damp_0_180_fs

        Added_M_0_180 = np.resize(Added_M_0_180, t)
        Damp_0_180 = np.resize(Damp_0_180, t)
        Added_M_0_180_fs = np.resize(Added_M_0_180_fs, t)
        Damp_0_180_fs = np.resize(Damp_0_180_fs, t)
        
        global AM11_fs_0, AM22_fs_0, AM33_fs_0, AM44_fs_0, AM55_fs_0, AM66_fs_0, AM13_fs_0, AM15_fs_0, AM24_fs_0
        global AM26_fs_0, AM31_fs_0, AM35_fs_0, AM42_fs_0, AM46_fs_0, AM51_fs_0, AM53_fs_0, AM62_fs_0, AM64_fs_0

        global AM11_fs_30, AM22_fs_30, AM33_fs_30, AM44_fs_30, AM55_fs_30, AM66_fs_30, AM13_fs_30, AM15_fs_30, AM24_fs_30
        global AM26_fs_30, AM31_fs_30, AM35_fs_30, AM42_fs_30, AM46_fs_30, AM51_fs_30, AM53_fs_30, AM62_fs_30, AM64_fs_30

        global AM11_fs_60, AM22_fs_60, AM33_fs_60, AM44_fs_60, AM55_fs_60, AM66_fs_60, AM13_fs_60, AM15_fs_60, AM24_fs_60
        global AM26_fs_60, AM31_fs_60, AM35_fs_60, AM42_fs_60, AM46_fs_60, AM51_fs_60, AM53_fs_60, AM62_fs_60, AM64_fs_60

        global AM11_fs_90, AM22_fs_90, AM33_fs_90, AM44_fs_90, AM55_fs_90, AM66_fs_90, AM13_fs_90, AM15_fs_90, AM24_fs_90
        global AM26_fs_90, AM31_fs_90, AM35_fs_90, AM42_fs_90, AM46_fs_90, AM51_fs_90, AM53_fs_90, AM62_fs_90, AM64_fs_90

        global AM11_fs_120, AM22_fs_120, AM33_fs_120, AM44_fs_120, AM55_fs_120, AM66_fs_120, AM13_fs_120, AM15_fs_120, AM24_fs_120
        global AM26_fs_120, AM31_fs_120, AM35_fs_120, AM42_fs_120, AM46_fs_120, AM51_fs_120, AM53_fs_120, AM62_fs_120, AM64_fs_120

        global AM11_fs_150, AM22_fs_150, AM33_fs_150, AM44_fs_150, AM55_fs_150, AM66_fs_150, AM13_fs_150, AM15_fs_150, AM24_fs_150
        global AM26_fs_150, AM31_fs_150, AM35_fs_150, AM42_fs_150, AM46_fs_150, AM51_fs_150, AM53_fs_150, AM62_fs_150, AM64_fs_150

        global AM11_fs_180, AM22_fs_180, AM33_fs_180, AM44_fs_180, AM55_fs_180, AM66_fs_180, AM13_fs_180, AM15_fs_180, AM24_fs_180
        global AM26_fs_180, AM31_fs_180, AM35_fs_180, AM42_fs_180, AM46_fs_180, AM51_fs_180, AM53_fs_180, AM62_fs_180, AM64_fs_180

        global D11_fs_0, D22_fs_0, D33_fs_0, D44_fs_0, D55_fs_0, D66_fs_0, D13_fs_0, D15_fs_0, D24_fs_0
        global D26_fs_0, D31_fs_0, D35_fs_0, D42_fs_0, D46_fs_0, D51_fs_0, D53_fs_0, D62_fs_0, D64_fs_0

        global D11_fs_30, D22_fs_30, D33_fs_30, D44_fs_30, D55_fs_30, D66_fs_30, D13_fs_30, D15_fs_30, D24_fs_30
        global D26_fs_30, D31_fs_30, D35_fs_30, D42_fs_30, D46_fs_30, D51_fs_30, D53_fs_30, D62_fs_30, D64_fs_30

        global D11_fs_60, D22_fs_60, D33_fs_60, D44_fs_60, D55_fs_60, D66_fs_60, D13_fs_60, D15_fs_60, D24_fs_60
        global D26_fs_60, D31_fs_60, D35_fs_60, D42_fs_60, D46_fs_60, D51_fs_60, D53_fs_60, D62_fs_60, D64_fs_60

        global D11_fs_90, D22_fs_90, D33_fs_90, D44_fs_90, D55_fs_90, D66_fs_90, D13_fs_90, D15_fs_90, D24_fs_90
        global D26_fs_90, D31_fs_90, D35_fs_90, D42_fs_90, D46_fs_90, D51_fs_90, D53_fs_90, D62_fs_90, D64_fs_90

        global D11_fs_120, D22_fs_120, D33_fs_120, D44_fs_120, D55_fs_120, D66_fs_120, D13_fs_120, D15_fs_120, D24_fs_120
        global D26_fs_120, D31_fs_120, D35_fs_120, D42_fs_120, D46_fs_120, D51_fs_120, D53_fs_120, D62_fs_120, D64_fs_120

        global D11_fs_150, D22_fs_150, D33_fs_150, D44_fs_150, D55_fs_150, D66_fs_150, D13_fs_150, D15_fs_150, D24_fs_150
        global D26_fs_150, D31_fs_150, D35_fs_150, D42_fs_150, D46_fs_150, D51_fs_150, D53_fs_150, D62_fs_150, D64_fs_150

        global D11_fs_180, D22_fs_180, D33_fs_180, D44_fs_180, D55_fs_180, D66_fs_180, D13_fs_180, D15_fs_180, D24_fs_180
        global D26_fs_180, D31_fs_180, D35_fs_180, D42_fs_180, D46_fs_180, D51_fs_180, D53_fs_180, D62_fs_180, D64_fs_180

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_0 = np.resize(AM11_fs_0, s); D11_fs_0 = np.resize(D11_fs_0, s)
        AM22_fs_0 = np.resize(AM22_fs_0, s); D22_fs_0 = np.resize(D22_fs_0, s)
        AM33_fs_0 = np.resize(AM33_fs_0, s); D33_fs_0 = np.resize(D33_fs_0, s)
        AM44_fs_0 = np.resize(AM44_fs_0, s); D44_fs_0 = np.resize(D44_fs_0, s)
        AM55_fs_0 = np.resize(AM55_fs_0, s); D55_fs_0 = np.resize(D55_fs_0, s)
        AM66_fs_0 = np.resize(AM66_fs_0, s); D66_fs_0 = np.resize(D66_fs_0, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_0 = np.resize(AM13_fs_0, s); D13_fs_0 = np.resize(D13_fs_0, s)
        AM15_fs_0 = np.resize(AM15_fs_0, s); D15_fs_0 = np.resize(D15_fs_0, s)
        AM24_fs_0 = np.resize(AM24_fs_0, s); D24_fs_0 = np.resize(D24_fs_0, s)
        AM26_fs_0 = np.resize(AM26_fs_0, s); D26_fs_0 = np.resize(D26_fs_0, s)
        AM31_fs_0 = np.resize(AM31_fs_0, s); D31_fs_0 = np.resize(D31_fs_0, s)
        AM35_fs_0 = np.resize(AM35_fs_0, s); D35_fs_0 = np.resize(D35_fs_0, s)
        AM42_fs_0 = np.resize(AM42_fs_0, s); D42_fs_0 = np.resize(D42_fs_0, s)
        AM46_fs_0 = np.resize(AM46_fs_0, s); D46_fs_0 = np.resize(D46_fs_0, s)
        AM51_fs_0 = np.resize(AM51_fs_0, s); D51_fs_0 = np.resize(D51_fs_0, s)
        AM53_fs_0 = np.resize(AM53_fs_0, s); D53_fs_0 = np.resize(D53_fs_0, s)
        AM62_fs_0 = np.resize(AM62_fs_0, s); D62_fs_0 = np.resize(D62_fs_0, s)
        AM64_fs_0 = np.resize(AM64_fs_0, s); D64_fs_0 = np.resize(D64_fs_0, s)

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_30 = np.resize(AM11_fs_30, s); D11_fs_30 = np.resize(D11_fs_30, s)
        AM22_fs_30 = np.resize(AM22_fs_30, s); D22_fs_30 = np.resize(D22_fs_30, s)
        AM33_fs_30 = np.resize(AM33_fs_30, s); D33_fs_30 = np.resize(D33_fs_30, s)
        AM44_fs_30 = np.resize(AM44_fs_30, s); D44_fs_30 = np.resize(D44_fs_30, s)
        AM55_fs_30 = np.resize(AM55_fs_30, s); D55_fs_30 = np.resize(D55_fs_30, s)
        AM66_fs_30 = np.resize(AM66_fs_30, s); D66_fs_30 = np.resize(D66_fs_30, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_30 = np.resize(AM13_fs_30, s); D13_fs_30 = np.resize(D13_fs_30, s)
        AM15_fs_30 = np.resize(AM15_fs_30, s); D15_fs_30 = np.resize(D15_fs_30, s)
        AM24_fs_30 = np.resize(AM24_fs_30, s); D24_fs_30 = np.resize(D24_fs_30, s)
        AM26_fs_30 = np.resize(AM26_fs_30, s); D26_fs_30 = np.resize(D26_fs_30, s)
        AM31_fs_30 = np.resize(AM31_fs_30, s); D31_fs_30 = np.resize(D31_fs_30, s)
        AM35_fs_30 = np.resize(AM35_fs_30, s); D35_fs_30 = np.resize(D35_fs_30, s)
        AM42_fs_30 = np.resize(AM42_fs_30, s); D42_fs_30 = np.resize(D42_fs_30, s)
        AM46_fs_30 = np.resize(AM46_fs_30, s); D46_fs_30 = np.resize(D46_fs_30, s)
        AM51_fs_30 = np.resize(AM51_fs_30, s); D51_fs_30 = np.resize(D51_fs_30, s)
        AM53_fs_30 = np.resize(AM53_fs_30, s); D53_fs_30 = np.resize(D53_fs_30, s)
        AM62_fs_30 = np.resize(AM62_fs_30, s); D62_fs_30 = np.resize(D62_fs_30, s)
        AM64_fs_30 = np.resize(AM64_fs_30, s); D64_fs_30 = np.resize(D64_fs_30, s)

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_60 = np.resize(AM11_fs_60, s); D11_fs_60 = np.resize(D11_fs_60, s)
        AM22_fs_60 = np.resize(AM22_fs_60, s); D22_fs_60 = np.resize(D22_fs_60, s)
        AM33_fs_60 = np.resize(AM33_fs_60, s); D33_fs_60 = np.resize(D33_fs_60, s)
        AM44_fs_60 = np.resize(AM44_fs_60, s); D44_fs_60 = np.resize(D44_fs_60, s)
        AM55_fs_60 = np.resize(AM55_fs_60, s); D55_fs_60 = np.resize(D55_fs_60, s)
        AM66_fs_60 = np.resize(AM66_fs_60, s); D66_fs_60 = np.resize(D66_fs_60, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_60 = np.resize(AM13_fs_60, s); D13_fs_60 = np.resize(D13_fs_60, s)
        AM15_fs_60 = np.resize(AM15_fs_60, s); D15_fs_60 = np.resize(D15_fs_60, s)
        AM24_fs_60 = np.resize(AM24_fs_60, s); D24_fs_60 = np.resize(D24_fs_60, s)
        AM26_fs_60 = np.resize(AM26_fs_60, s); D26_fs_60 = np.resize(D26_fs_60, s)
        AM31_fs_60 = np.resize(AM31_fs_60, s); D31_fs_60 = np.resize(D31_fs_60, s)
        AM35_fs_60 = np.resize(AM35_fs_60, s); D35_fs_60 = np.resize(D35_fs_60, s)
        AM42_fs_60 = np.resize(AM42_fs_60, s); D42_fs_60 = np.resize(D42_fs_60, s)
        AM46_fs_60 = np.resize(AM46_fs_60, s); D46_fs_60 = np.resize(D46_fs_60, s)
        AM51_fs_60 = np.resize(AM51_fs_60, s); D51_fs_60 = np.resize(D51_fs_60, s)
        AM53_fs_60 = np.resize(AM53_fs_60, s); D53_fs_60 = np.resize(D53_fs_60, s)
        AM62_fs_60 = np.resize(AM62_fs_60, s); D62_fs_60 = np.resize(D62_fs_60, s)
        AM64_fs_60 = np.resize(AM64_fs_60, s); D64_fs_60 = np.resize(D64_fs_60, s)

    ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_90 = np.resize(AM11_fs_90, s); D11_fs_90 = np.resize(D11_fs_90, s)
        AM22_fs_90 = np.resize(AM22_fs_90, s); D22_fs_90 = np.resize(D22_fs_90, s)
        AM33_fs_90 = np.resize(AM33_fs_90, s); D33_fs_90 = np.resize(D33_fs_90, s)
        AM44_fs_90 = np.resize(AM44_fs_90, s); D44_fs_90 = np.resize(D44_fs_90, s)
        AM55_fs_90 = np.resize(AM55_fs_90, s); D55_fs_90 = np.resize(D55_fs_90, s)
        AM66_fs_90 = np.resize(AM66_fs_90, s); D66_fs_90 = np.resize(D66_fs_90, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_90 = np.resize(AM13_fs_90, s); D13_fs_90 = np.resize(D13_fs_90, s)
        AM15_fs_90 = np.resize(AM15_fs_90, s); D15_fs_90 = np.resize(D15_fs_90, s)
        AM24_fs_90 = np.resize(AM24_fs_90, s); D24_fs_90 = np.resize(D24_fs_90, s)
        AM26_fs_90 = np.resize(AM26_fs_90, s); D26_fs_90 = np.resize(D26_fs_90, s)
        AM31_fs_90 = np.resize(AM31_fs_90, s); D31_fs_90 = np.resize(D31_fs_90, s)
        AM35_fs_90 = np.resize(AM35_fs_90, s); D35_fs_90 = np.resize(D35_fs_90, s)
        AM42_fs_90 = np.resize(AM42_fs_90, s); D42_fs_90 = np.resize(D42_fs_90, s)
        AM46_fs_90 = np.resize(AM46_fs_90, s); D46_fs_90 = np.resize(D46_fs_90, s)
        AM51_fs_90 = np.resize(AM51_fs_90, s); D51_fs_90 = np.resize(D51_fs_90, s)
        AM53_fs_90 = np.resize(AM53_fs_90, s); D53_fs_90 = np.resize(D53_fs_90, s)
        AM62_fs_90 = np.resize(AM62_fs_90, s); D62_fs_90 = np.resize(D62_fs_90, s)
        AM64_fs_90 = np.resize(AM64_fs_90, s); D64_fs_90 = np.resize(D64_fs_90, s)

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_120 = np.resize(AM11_fs_120, s); D11_fs_120 = np.resize(D11_fs_120, s)
        AM22_fs_120 = np.resize(AM22_fs_120, s); D22_fs_120 = np.resize(D22_fs_120, s)
        AM33_fs_120 = np.resize(AM33_fs_120, s); D33_fs_120 = np.resize(D33_fs_120, s)
        AM44_fs_120 = np.resize(AM44_fs_120, s); D44_fs_120 = np.resize(D44_fs_120, s)
        AM55_fs_120 = np.resize(AM55_fs_120, s); D55_fs_120 = np.resize(D55_fs_120, s)
        AM66_fs_120 = np.resize(AM66_fs_120, s); D66_fs_120 = np.resize(D66_fs_120, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_120 = np.resize(AM13_fs_120, s); D13_fs_120 = np.resize(D13_fs_120, s)
        AM15_fs_120 = np.resize(AM15_fs_120, s); D15_fs_120 = np.resize(D15_fs_120, s)
        AM24_fs_120 = np.resize(AM24_fs_120, s); D24_fs_120 = np.resize(D24_fs_120, s)
        AM26_fs_120 = np.resize(AM26_fs_120, s); D26_fs_120 = np.resize(D26_fs_120, s)
        AM31_fs_120 = np.resize(AM31_fs_120, s); D31_fs_120 = np.resize(D31_fs_120, s)
        AM35_fs_120 = np.resize(AM35_fs_120, s); D35_fs_120 = np.resize(D35_fs_120, s)
        AM42_fs_120 = np.resize(AM42_fs_120, s); D42_fs_120 = np.resize(D42_fs_120, s)
        AM46_fs_120 = np.resize(AM46_fs_120, s); D46_fs_120 = np.resize(D46_fs_120, s)
        AM51_fs_120 = np.resize(AM51_fs_120, s); D51_fs_120 = np.resize(D51_fs_120, s)
        AM53_fs_120 = np.resize(AM53_fs_120, s); D53_fs_120 = np.resize(D53_fs_120, s)
        AM62_fs_120 = np.resize(AM62_fs_120, s); D62_fs_120 = np.resize(D62_fs_120, s)
        AM64_fs_120 = np.resize(AM64_fs_120, s); D64_fs_120 = np.resize(D64_fs_120, s)

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_150 = np.resize(AM11_fs_150, s); D11_fs_150 = np.resize(D11_fs_150, s)
        AM22_fs_150 = np.resize(AM22_fs_150, s); D22_fs_150 = np.resize(D22_fs_150, s)
        AM33_fs_150 = np.resize(AM33_fs_150, s); D33_fs_150 = np.resize(D33_fs_150, s)
        AM44_fs_150 = np.resize(AM44_fs_150, s); D44_fs_150 = np.resize(D44_fs_150, s)
        AM55_fs_150 = np.resize(AM55_fs_150, s); D55_fs_150 = np.resize(D55_fs_150, s)
        AM66_fs_150 = np.resize(AM66_fs_150, s); D66_fs_150 = np.resize(D66_fs_150, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_150 = np.resize(AM13_fs_150, s); D13_fs_150 = np.resize(D13_fs_150, s)
        AM15_fs_150 = np.resize(AM15_fs_150, s); D15_fs_150 = np.resize(D15_fs_150, s)
        AM24_fs_150 = np.resize(AM24_fs_150, s); D24_fs_150 = np.resize(D24_fs_150, s)
        AM26_fs_150 = np.resize(AM26_fs_150, s); D26_fs_150 = np.resize(D26_fs_150, s)
        AM31_fs_150 = np.resize(AM31_fs_150, s); D31_fs_150 = np.resize(D31_fs_150, s)
        AM35_fs_150 = np.resize(AM35_fs_150, s); D35_fs_150 = np.resize(D35_fs_150, s)
        AM42_fs_150 = np.resize(AM42_fs_150, s); D42_fs_150 = np.resize(D42_fs_150, s)
        AM46_fs_150 = np.resize(AM46_fs_150, s); D46_fs_150 = np.resize(D46_fs_150, s)
        AM51_fs_150 = np.resize(AM51_fs_150, s); D51_fs_150 = np.resize(D51_fs_150, s)
        AM53_fs_150 = np.resize(AM53_fs_150, s); D53_fs_150 = np.resize(D53_fs_150, s)
        AM62_fs_150 = np.resize(AM62_fs_150, s); D62_fs_150 = np.resize(D62_fs_150, s)
        AM64_fs_150 = np.resize(AM64_fs_150, s); D64_fs_150 = np.resize(D64_fs_150, s)

        ## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
        AM11_fs_180 = np.resize(AM11_fs_180, s); D11_fs_180 = np.resize(D11_fs_180, s)
        AM22_fs_180 = np.resize(AM22_fs_180, s); D22_fs_180 = np.resize(D22_fs_180, s)
        AM33_fs_180 = np.resize(AM33_fs_180, s); D33_fs_180 = np.resize(D33_fs_180, s)
        AM44_fs_180 = np.resize(AM44_fs_180, s); D44_fs_180 = np.resize(D44_fs_180, s)
        AM55_fs_180 = np.resize(AM55_fs_180, s); D55_fs_180 = np.resize(D55_fs_180, s)
        AM66_fs_180 = np.resize(AM66_fs_180, s); D66_fs_180 = np.resize(D66_fs_180, s)
        ## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
        AM13_fs_180 = np.resize(AM13_fs_180, s); D13_fs_180 = np.resize(D13_fs_180, s)
        AM15_fs_180 = np.resize(AM15_fs_180, s); D15_fs_180 = np.resize(D15_fs_180, s)
        AM24_fs_180 = np.resize(AM24_fs_180, s); D24_fs_180 = np.resize(D24_fs_180, s)
        AM26_fs_180 = np.resize(AM26_fs_180, s); D26_fs_180 = np.resize(D26_fs_180, s)
        AM31_fs_180 = np.resize(AM31_fs_180, s); D31_fs_180 = np.resize(D31_fs_180, s)
        AM35_fs_180 = np.resize(AM35_fs_180, s); D35_fs_180 = np.resize(D35_fs_180, s)
        AM42_fs_180 = np.resize(AM42_fs_180, s); D42_fs_180 = np.resize(D42_fs_180, s)
        AM46_fs_180 = np.resize(AM46_fs_180, s); D46_fs_180 = np.resize(D46_fs_180, s)
        AM51_fs_180 = np.resize(AM51_fs_180, s); D51_fs_180 = np.resize(D51_fs_180, s)
        AM53_fs_180 = np.resize(AM53_fs_180, s); D53_fs_180 = np.resize(D53_fs_180, s)
        AM62_fs_180 = np.resize(AM62_fs_180, s); D62_fs_180 = np.resize(D62_fs_180, s)
        AM64_fs_180 = np.resize(AM64_fs_180, s); D64_fs_180 = np.resize(D64_fs_180, s)

def resize_K(nships):

    global K33_fs, K34_fs, K35_fs, K43_fs, K44_fs, K45_fs, K46_fs, K53_fs, K54_fs, K55_fs, K56_fs, K64_fs
    
    nfreqs = len(sea_var.frequencies)
    nheads = len(sea_var.angles)
    s = tuple((nfreqs,nheads,nships))

    K33_fs = np.resize(K33_fs, s); K34_fs = np.resize(K34_fs, s); K35_fs = np.resize(K35_fs, s)
    K43_fs = np.resize(K43_fs, s); K44_fs = np.resize(K44_fs, s); K45_fs = np.resize(K45_fs, s); K46_fs = np.resize(K46_fs, s)
    K53_fs = np.resize(K53_fs, s); K54_fs = np.resize(K54_fs, s); K55_fs = np.resize(K55_fs, s); K56_fs = np.resize(K56_fs, s)
    K64_fs = np.resize(K64_fs, s)

def resize_forces(cof):
    
    nfreqs = len(sea_var.frequencies)

    if cof == 0 or cof == 2:
        nships = len(sea_var.list_fn_0)

        ## MODEL SCALE VARIABLES
        global FEX_cos0_60, FEX_cos90, FEX_cos120_180, FEX_sin0_60, FEX_sin90, FEX_sin120_180
        global FEY_sin30_150, FEY_cos30_150, FEZ_sin, FEZ_cos
        global MEX_sin30_150, MEX_cos30_150, MEY_sin, MEY_cos
        global MEZ_sin30_150, MEZ_cos30_150
        global Phase_FEX0_60, Phase_FEX90, Phase_FEX120_180, Phase_FEY30_150, Phase_FEZ
        global Phase_MEX30_150, Phase_MEY, Phase_MEZ30_150

        ## FULL SCALE VARIABLES
        global FEX_cos_fs, FEX_sin_fs, FEY_cos_fs, FEY_sin_fs, FEZ_cos_fs, FEZ_sin_fs
        global MEX_cos_fs, MEX_sin_fs, MEY_cos_fs, MEY_sin_fs, MEZ_cos_fs, MEZ_sin_fs
        global FEX_fs, FEY_fs, FEZ_fs, MEX_fs, MEY_fs, MEZ_fs
        global Phase_FEX_fs, Phase_FEY_fs, Phase_FEZ_fs, Phase_MEX_fs, Phase_MEY_fs, Phase_MEZ_fs
        
        ## TUPLES 
        p = (nfreqs,1,nships); t = (nfreqs,3,nships); s = (nfreqs,5,nships); q = (nfreqs,7,nships)
        
        ## RESIZE MODEL SCALE VARIABLES
        FEX_cos0_60 = np.resize(FEX_cos0_60, t)    ; FEX_cos90 = np.resize(FEX_cos90, p); FEX_cos120_180 = np.resize(FEX_cos120_180, t)
        FEX_sin0_60 = np.resize(FEX_sin0_60, t)    ; FEX_sin90 = np.resize(FEX_sin90, p); FEX_sin120_180 = np.resize(FEX_sin120_180, t)
        FEY_sin30_150 = np.resize(FEY_sin30_150,s) ; FEY_cos30_150 = np.resize(FEY_cos30_150, s)
        FEZ_sin = np.resize(FEZ_sin, q)            ; FEZ_cos= np.resize(FEZ_cos, q)

        MEX_sin30_150 = np.resize(MEX_sin30_150,s)  ; MEX_cos30_150 = np.resize(MEX_cos30_150, s)
        MEY_sin = np.resize(MEY_sin, q)             ; MEY_cos = np.resize(MEY_cos, q)
        MEZ_sin30_150 = np.resize(MEZ_sin30_150, s) ; MEZ_cos30_150 = np.resize(MEZ_cos30_150, s)

        Phase_FEX0_60 = np.resize(Phase_FEX0_60, t); Phase_FEX90 = np.resize(Phase_FEX90, p); Phase_FEX120_180 = np.resize(Phase_FEX120_180, t)
        Phase_FEY30_150 = np.resize(Phase_FEY30_150, s)
        Phase_FEZ = np.resize(Phase_FEZ, q)

        Phase_MEX30_150 = np.resize(Phase_MEX30_150, s); Phase_MEY = np.resize(Phase_MEY, q)
        Phase_MEZ30_150 = np.resize(Phase_MEZ30_150, s)

        ## RESIZE FULL SCALE VARIABLES
        FEX_fs = np.resize(FEX_fs, q)
        Phase_FEX_fs = np.resize(Phase_FEX_fs, q)
        FEX_cos_fs = np.resize(FEX_cos_fs, q); FEX_sin_fs = np.resize(FEX_sin_fs, q)
        
        FEY_fs = np.resize(FEY_fs, q)
        Phase_FEY_fs = np.resize(Phase_FEY_fs, q)
        FEY_cos_fs = np.resize(FEY_cos_fs, q); FEY_sin_fs = np.resize(FEY_sin_fs, q)

        FEZ_fs = np.resize(FEZ_fs, q)
        Phase_FEZ_fs = np.resize(Phase_FEZ_fs, q)
        FEZ_cos_fs = np.resize(FEZ_cos_fs, q); FEZ_sin_fs = np.resize(FEZ_sin_fs, q)

        MEX_fs = np.resize(MEX_fs, q)
        Phase_MEX_fs = np.resize(Phase_MEX_fs, q)
        MEX_cos_fs = np.resize(MEX_cos_fs, q); MEX_sin_fs = np.resize(MEX_sin_fs, q)

        MEY_fs = np.resize(MEY_fs, q)
        Phase_MEY_fs = np.resize(Phase_MEY_fs, q)
        MEY_cos_fs = np.resize(MEY_cos_fs, q); MEY_sin_fs = np.resize(MEY_sin_fs, q)

        MEZ_fs = np.resize(MEZ_fs, q)
        Phase_MEZ_fs = np.resize(Phase_MEZ_fs, q)
        MEZ_cos_fs = np.resize(MEZ_cos_fs, q); MEZ_sin_fs = np.resize(MEZ_sin_fs, q)

    if cof == 1 or cof == 2:

        nships = len(sea_var.list_fn)

        ## MODEL SCALE VARIABLES
        global FEX_cos0_60_din, FEX_cos90_din, FEX_cos120_180_din, FEX_sin0_60_din, FEX_sin90_din, FEX_sin120_180_din
        global FEY_sin30_150_din, FEY_cos30_150_din, FEZ_sin_din, FEZ_cos_din
        global MEX_sin30_150_din, MEX_cos30_150_din, MEY_sin_din, MEY_cos_din
        global MEZ_sin30_150_din, MEZ_cos30_150_din
        global Phase_FEX0_60_din, Phase_FEX90_din, Phase_FEX120_180_din, Phase_FEY30_150_din, Phase_FEZ_din
        global Phase_MEX30_150_din, Phase_MEY_din, Phase_MEZ30_150_din
        
        ## FULL SCALE VARIABLES
        global FEX_cos_fs_din, FEX_sin_fs_din, FEY_cos_fs_din, FEY_sin_fs_din, FEZ_cos_fs_din, FEZ_sin_fs_din
        global MEX_cos_fs_din, MEX_sin_fs_din, MEY_cos_fs_din, MEY_sin_fs_din, MEZ_cos_fs_din, MEZ_sin_fs_din
        global FEX_fs_din, FEY_fs_din, FEZ_fs_din, MEX_fs_din, MEY_fs_din, MEZ_fs_din
        global Phase_FEX_fs_din, Phase_FEY_fs_din, Phase_FEZ_fs_din, Phase_MEX_fs_din, Phase_MEY_fs_din, Phase_MEZ_fs_din
        
        ## TUPLES 
        p = (nfreqs,1,nships); t = (nfreqs,3,nships); s = (nfreqs,5,nships); q = (nfreqs,7,nships)
        
        ## RESIZE MODEL SCALE VARIABLES
        FEX_cos0_60_din = np.resize(FEX_cos0_60_din, t)    ; FEX_cos90_din = np.resize(FEX_cos90_din, p); FEX_cos120_180_din = np.resize(FEX_cos120_180_din, t)
        FEX_sin0_60_din = np.resize(FEX_sin0_60_din, t)    ; FEX_sin90_din = np.resize(FEX_sin90_din, p); FEX_sin120_180_din = np.resize(FEX_sin120_180_din, t)
        FEY_sin30_150_din = np.resize(FEY_sin30_150_din,s) ; FEY_cos30_150_din = np.resize(FEY_cos30_150_din, s)
        FEZ_sin_din = np.resize(FEZ_sin_din, q)            ; FEZ_cos_din = np.resize(FEZ_cos_din, q)

        MEX_sin30_150_din = np.resize(MEX_sin30_150_din, s) ; MEX_cos30_150_din = np.resize(MEX_cos30_150_din, s)
        MEY_sin_din = np.resize(MEY_sin_din, q)             ; MEY_cos_din = np.resize(MEY_cos_din, q)
        MEZ_sin30_150_din = np.resize(MEZ_sin30_150_din, s) ; MEZ_cos30_150_din = np.resize(MEZ_cos30_150_din, s)

        Phase_FEX0_60_din = np.resize(Phase_FEX0_60_din, t); Phase_FEX90_din = np.resize(Phase_FEX90_din, p); Phase_FEX120_180_din = np.resize(Phase_FEX120_180_din, t)
        Phase_FEY30_150_din = np.resize(Phase_FEY30_150_din, s)
        Phase_FEZ_din = np.resize(Phase_FEZ_din, q)

        Phase_MEX30_150_din = np.resize(Phase_MEX30_150_din, s); Phase_MEY_din = np.resize(Phase_MEY_din, q)
        Phase_MEZ30_150_din = np.resize(Phase_MEZ30_150_din, s)

        ## RESIZE FULL SCALE VARIABLES
        FEX_fs_din = np.resize(FEX_fs_din, q)
        Phase_FEX_fs_din = np.resize(Phase_FEX_fs_din, q)
        FEX_cos_fs_din = np.resize(FEX_cos_fs_din, q); FEX_sin_fs_din = np.resize(FEX_sin_fs_din, q)
        
        FEY_fs_din = np.resize(FEY_fs_din, q)
        Phase_FEY_fs_din = np.resize(Phase_FEY_fs_din, q)
        FEY_cos_fs_din = np.resize(FEY_cos_fs_din, q); FEY_sin_fs_din = np.resize(FEY_sin_fs_din, q)

        FEZ_fs_din = np.resize(FEZ_fs_din, q)
        Phase_FEZ_fs_din = np.resize(Phase_FEZ_fs_din, q)
        FEZ_cos_fs_din = np.resize(FEZ_cos_fs_din, q); FEZ_sin_fs_din = np.resize(FEZ_sin_fs_din, q)

        MEX_fs_din = np.resize(MEX_fs_din, q)
        Phase_MEX_fs_din = np.resize(Phase_MEX_fs_din, q)
        MEX_cos_fs_din = np.resize(MEX_cos_fs_din, q); MEX_sin_fs_din = np.resize(MEX_sin_fs_din, q)

        MEY_fs_din = np.resize(MEY_fs_din, q)
        Phase_MEY_fs_din = np.resize(Phase_MEY_fs_din, q)
        MEY_cos_fs_din = np.resize(MEY_cos_fs_din, q); MEY_sin_fs_din = np.resize(MEY_sin_fs_din, q)

        MEZ_fs_din = np.resize(MEZ_fs_din, q)
        Phase_MEZ_fs_din = np.resize(Phase_MEZ_fs_din, q)
        MEZ_cos_fs_din = np.resize(MEZ_cos_fs_din, q); MEZ_sin_fs_din = np.resize(MEZ_sin_fs_din, q)

def resize_RAO(nships):

    nfreqs = len(sea_var.frequencies)
    nheads = len(sea_var.angles)

    global RAO_11, RAO_22, RAO_33, RAO_44, RAO_55, RAO_66
    global RAO_44_rep, RAO_55_rep, RAO_66_rep
    global RAO_phase_11, RAO_phase_22, RAO_phase_33, RAO_phase_44, RAO_phase_55, RAO_phase_66

    global RAO_11_traslated, RAO_phase_11_traslated, RAO_22_traslated,RAO_phase_22_traslated, RAO_33_traslated,RAO_phase_33_traslated
    global RAO_44_traslated, RAO_phase_44_traslated, RAO_55_traslated,RAO_phase_55_traslated, RAO_66_traslated,RAO_phase_66_traslated

    t = (nfreqs,nheads,nships)

    RAO_11 = np.resize(RAO_11, t);         RAO_22 = np.resize(RAO_22, t);         RAO_33 = np.resize(RAO_33, t)
    RAO_44 = np.resize(RAO_44, t);         RAO_55 = np.resize(RAO_55, t);         RAO_66 = np.resize(RAO_66, t)
    RAO_44_rep = np.resize(RAO_44_rep, t); RAO_55_rep = np.resize(RAO_55_rep, t); RAO_66_rep = np.resize(RAO_66_rep, t)

    RAO_phase_11 = np.resize(RAO_phase_11, t); RAO_phase_22 = np.resize(RAO_phase_22, t)
    RAO_phase_33 = np.resize(RAO_phase_33, t); RAO_phase_44 = np.resize(RAO_phase_44, t)
    RAO_phase_55 = np.resize(RAO_phase_55, t); RAO_phase_66 = np.resize(RAO_phase_66, t)

    RAO_11_traslated = np.resize(RAO_11_traslated, t); RAO_22_traslated = np.resize(RAO_22_traslated, t); RAO_33_traslated = np.resize(RAO_33_traslated, t)
    RAO_44_traslated = np.resize(RAO_44_traslated, t); RAO_55_traslated = np.resize(RAO_55_traslated, t); RAO_66_traslated = np.resize(RAO_66_traslated, t)

    RAO_phase_11_traslated = np.resize(RAO_phase_11_traslated, t); RAO_phase_22_traslated = np.resize(RAO_phase_22_traslated, t)
    RAO_phase_33_traslated = np.resize(RAO_phase_33_traslated, t); RAO_phase_44_traslated = np.resize(RAO_phase_44_traslated, t)
    RAO_phase_55_traslated = np.resize(RAO_phase_55_traslated, t); RAO_phase_66_traslated = np.resize(RAO_phase_66_traslated, t)
    
def resize_other(cof,nships):
    global Displaz, scale1, scale2, fq1, fq2
    
    nfreqs = len(sea_var.frequencies)
    Displaz = np.resize(Displaz, nships)
    
    if cof == 0:
        nships = len(sea_var.list_fn_0)
        s = (nfreqs, nships)
        fq1 = np.resize(fq1, s)
        scale1 = np.resize(scale1, nships)
    
    elif cof == 1:
        nships = len(sea_var.list_fn)
        s = (nfreqs, nships)
        fq2 = np.resize(fq2, s)
        scale2 = np.resize(scale2, nships)
    
    else:
        nships1 = len(sea_var.list_fn_0)
        nships2 = len(sea_var.list_fn)

        s = (nfreqs, nships1)
        fq1 = np.resize(fq1, s)
        scale1 = np.resize(scale1, nships1)

        s = (nfreqs, nships2)
        fq2 = np.resize(fq2, s)
        scale2 = np.resize(scale2, nships2)

def resize_seakeeping(nships):

    nfreqs = len(sea_var.frequencies)
    nheads = len(sea_var.angles)

    global Mod; global Dir; global Tp; global Hs
    global Wave_sp
    global Point_sk
    global Point_CDG

    global mov_11_max ; global mov_22_max ; global mov_33_max ; global mov_44_max ; global mov_55_max ; global mov_66_max
    global mov_11_sig ; global mov_22_sig ; global mov_33_sig ; global mov_44_sig ; global mov_55_sig ; global mov_66_sig
    
    global acc_RMS_11 ; global acc_RMS_22 ; global acc_RMS_33 ; global acc_RMS_44 ; global acc_RMS_55 ; global acc_RMS_66

    global m0; global m2; global m4
    
    global Significative_mag; global Motion_sickness

    r = (nships); t = (nfreqs,nheads,nships); s = (nheads,nships)

    Mod = np.resize(Mod, r); Dir = np.resize(Dir, r); Tp = np.resize(Tp, r); Hs = np.resize(Hs, r)
    Wave_sp = np.resize(Wave_sp, t)

    mov_11_max = np.resize(mov_11_max, s) ; mov_11_sig = np.resize(mov_11_sig, s)
    mov_22_max = np.resize(mov_22_max, s) ; mov_22_sig = np.resize(mov_22_sig, s)
    mov_33_max = np.resize(mov_33_max, s) ; mov_33_sig = np.resize(mov_33_sig, s)
    mov_44_max = np.resize(mov_44_max, s) ; mov_44_sig = np.resize(mov_44_sig, s)
    mov_55_max = np.resize(mov_55_max, s) ; mov_55_sig = np.resize(mov_55_sig, s)
    mov_66_max = np.resize(mov_66_max, s) ; mov_66_sig = np.resize(mov_66_sig, s)

    acc_RMS_11 = np.resize(acc_RMS_11, s); acc_RMS_44 = np.resize(acc_RMS_44, s)
    acc_RMS_22 = np.resize(acc_RMS_22, s); acc_RMS_55 = np.resize(acc_RMS_55, s)
    acc_RMS_33 = np.resize(acc_RMS_33, s); acc_RMS_66 = np.resize(acc_RMS_66, s)

    q = (6,nheads,nships)
    m0 = np.resize(m0, q); m2 = np.resize(m2, q); m4 = np.resize(m4, q)

    Significative_mag = np.resize(Significative_mag, s); Motion_sickness = np.resize(Motion_sickness, s)

    w = (nships,3)
    Point_sk = np.resize(Point_sk, w)
    Point_CDG = np.resize(Point_CDG, w)


################################################################################################
#                                VARIABLES DEFINITION                                          #
################################################################################################

#tuples with matrix dimension
s=tuple((1,1))
t=tuple((1,1,1))
z=tuple((1,1,1,1))

Displaz = np.zeros(1)
scale1 = np.zeros(1)
scale2 = np.zeros(1)

fq1 = np.zeros(s) ## frequencies for Fn equal to zero
fq2 = np.zeros(s) ## frequencies for Fn greater than zero

## dimmensionless added masses and dampings: diagonal terms
Added_M = np.zeros(t)
Damp = np.zeros(t)

Added_M_0_180 = np.zeros(z); Damp_0_180 = np.zeros(z)
Added_M_0_180_fs = np.zeros(z); Damp_0_180_fs = np.zeros(z)

## forces, moments & phases: SCALE MODEL (FN = 0)
FEX_cos0_60 = np.zeros(t) ; FEX_cos90 = np.zeros(t); FEX_cos120_180 = np.zeros(t)
FEX_sin0_60 = np.zeros(t) ; FEX_sin90 = np.zeros(t); FEX_sin120_180 = np.zeros(t)

FEY_sin30_150 = np.zeros(t) ; FEY_cos30_150 = np.zeros(t)
FEZ_sin = np.zeros(t)       ; FEZ_cos= np.zeros(t)

MEX_sin30_150 = np.zeros(t) ; MEX_cos30_150 = np.zeros(t)
MEY_sin = np.zeros(t)       ; MEY_cos =np.zeros(t)
MEZ_sin30_150 = np.zeros(t) ; MEZ_cos30_150 = np.zeros(t)

Phase_FEX0_60 = np.zeros(t) ; Phase_FEX90 = np.zeros(t); Phase_FEX120_180 = np.zeros(t)
Phase_FEY30_150 = np.zeros(t)
Phase_FEZ = np.zeros(t)

Phase_MEX30_150 = np.zeros(t)
Phase_MEY = np.zeros(t)
Phase_MEZ30_150 = np.zeros(t)

## forces, moments & phases: SCALE MODEL (FN > 0)
FEX_cos0_60_din = np.zeros(t) ; FEX_cos90_din = np.zeros(t); FEX_cos120_180_din = np.zeros(t)
FEX_sin0_60_din = np.zeros(t) ; FEX_sin90_din = np.zeros(t); FEX_sin120_180_din = np.zeros(t)

FEY_sin30_150_din = np.zeros(t) ; FEY_cos30_150_din = np.zeros(t)
FEZ_sin_din = np.zeros(t)       ; FEZ_cos_din = np.zeros(t)

MEX_sin30_150_din = np.zeros(t) ; MEX_cos30_150_din = np.zeros(t)
MEY_sin_din = np.zeros(t)       ; MEY_cos_din =np.zeros(t)
MEZ_sin30_150_din = np.zeros(t) ; MEZ_cos30_150_din = np.zeros(t)

Phase_FEX0_60_din = np.zeros(t) ; Phase_FEX90_din = np.zeros(t); Phase_FEX120_180_din = np.zeros(t)
Phase_FEY30_150_din = np.zeros(t)
Phase_FEZ_din = np.zeros(t)

Phase_MEX30_150_din = np.zeros(t)
Phase_MEY_din = np.zeros(t)
Phase_MEZ30_150_din = np.zeros(t)

## excitation forces and moment & phases: FULL SCALE
## Component sin and cos
FEX_cos_fs = np.zeros(t); FEX_sin_fs = np.zeros(t)
FEY_cos_fs = np.zeros(t); FEY_sin_fs = np.zeros(t)
FEZ_cos_fs = np.zeros(t); FEZ_sin_fs = np.zeros(t)

MEX_cos_fs = np.zeros(t); MEX_sin_fs = np.zeros(t)
MEY_cos_fs = np.zeros(t); MEY_sin_fs = np.zeros(t)
MEZ_cos_fs = np.zeros(t); MEZ_sin_fs = np.zeros(t)

# force module
FEX_fs = np.zeros(t); FEY_fs = np.zeros(t); FEZ_fs = np.zeros(t)
MEX_fs = np.zeros(t); MEY_fs = np.zeros(t); MEZ_fs = np.zeros(t)

## Phases of forces and moments
Phase_FEX_fs = np.zeros(t); Phase_FEY_fs = np.zeros(t); Phase_FEZ_fs = np.zeros(t)
Phase_MEX_fs = np.zeros(t); Phase_MEY_fs = np.zeros(t); Phase_MEZ_fs = np.zeros(t)

## excitation forces and moment & phases: FULL SCALE (FN>0)
## Component sin and cos
FEX_cos_fs_din = np.zeros(t); FEX_sin_fs_din = np.zeros(t)
FEY_cos_fs_din = np.zeros(t); FEY_sin_fs_din = np.zeros(t)
FEZ_cos_fs_din = np.zeros(t); FEZ_sin_fs_din = np.zeros(t)

MEX_cos_fs_din = np.zeros(t); MEX_sin_fs_din = np.zeros(t)
MEY_cos_fs_din = np.zeros(t); MEY_sin_fs_din = np.zeros(t)
MEZ_cos_fs_din = np.zeros(t); MEZ_sin_fs_din = np.zeros(t)

# force module
FEX_fs_din = np.zeros(t); FEY_fs_din = np.zeros(t); FEZ_fs_din = np.zeros(t)
MEX_fs_din = np.zeros(t); MEY_fs_din = np.zeros(t); MEZ_fs_din = np.zeros(t)

## Phases of forces and moments
Phase_FEX_fs_din = np.zeros(t); Phase_FEY_fs_din = np.zeros(t); Phase_FEZ_fs_din = np.zeros(t)
Phase_MEX_fs_din = np.zeros(t); Phase_MEY_fs_din = np.zeros(t); Phase_MEZ_fs_din = np.zeros(t)

## Restoration terms
K33_fs = np.zeros(t); K34_fs = np.zeros(t); K35_fs = np.zeros(t)
K43_fs = np.zeros(t); K44_fs = np.zeros(s); K45_fs = np.zeros(t); K46_fs = np.zeros(t)
K53_fs = np.zeros(t); K54_fs = np.zeros(s); K55_fs = np.zeros(s); K56_fs = np.zeros(t)
K64_fs = np.zeros(t)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs = np.zeros(t); D11_fs = np.zeros(t)
AM22_fs = np.zeros(t); D22_fs = np.zeros(t)
AM33_fs = np.zeros(t); D33_fs = np.zeros(t)
AM44_fs = np.zeros(t); D44_fs = np.zeros(t)
AM55_fs = np.zeros(t); D55_fs = np.zeros(t)
AM66_fs = np.zeros(t); D66_fs = np.zeros(t)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs = np.zeros(t); D13_fs = np.zeros(t)
AM15_fs = np.zeros(t); D15_fs = np.zeros(t)
AM24_fs = np.zeros(t); D24_fs = np.zeros(t)
AM26_fs = np.zeros(t); D26_fs = np.zeros(t)
AM31_fs = np.zeros(t); D31_fs = np.zeros(t)
AM35_fs = np.zeros(t); D35_fs = np.zeros(t)
AM42_fs = np.zeros(t); D42_fs = np.zeros(t)
AM46_fs = np.zeros(t); D46_fs = np.zeros(t)
AM51_fs = np.zeros(t); D51_fs = np.zeros(t)
AM53_fs = np.zeros(t); D53_fs = np.zeros(t)
AM62_fs = np.zeros(t); D62_fs = np.zeros(t)
AM64_fs = np.zeros(t); D64_fs = np.zeros(t)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_0 = np.zeros(s); D11_fs_0 = np.zeros(s)
AM22_fs_0 = np.zeros(s); D22_fs_0 = np.zeros(s)
AM33_fs_0 = np.zeros(s); D33_fs_0 = np.zeros(s)
AM44_fs_0 = np.zeros(s); D44_fs_0 = np.zeros(s)
AM55_fs_0 = np.zeros(s); D55_fs_0 = np.zeros(s)
AM66_fs_0 = np.zeros(s); D66_fs_0 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_0 = np.zeros(s); D13_fs_0 = np.zeros(s)
AM15_fs_0 = np.zeros(s); D15_fs_0 = np.zeros(s)
AM24_fs_0 = np.zeros(s); D24_fs_0 = np.zeros(s)
AM26_fs_0 = np.zeros(s); D26_fs_0 = np.zeros(s)
AM31_fs_0 = np.zeros(s); D31_fs_0 = np.zeros(s)
AM35_fs_0 = np.zeros(s); D35_fs_0 = np.zeros(s)
AM42_fs_0 = np.zeros(s); D42_fs_0 = np.zeros(s)
AM46_fs_0 = np.zeros(s); D46_fs_0 = np.zeros(s)
AM51_fs_0 = np.zeros(s); D51_fs_0 = np.zeros(s)
AM53_fs_0 = np.zeros(s); D53_fs_0 = np.zeros(s)
AM62_fs_0 = np.zeros(s); D62_fs_0 = np.zeros(s)
AM64_fs_0 = np.zeros(s); D64_fs_0 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_30 = np.zeros(s); D11_fs_30 = np.zeros(s)
AM22_fs_30 = np.zeros(s); D22_fs_30 = np.zeros(s)
AM33_fs_30 = np.zeros(s); D33_fs_30 = np.zeros(s)
AM44_fs_30 = np.zeros(s); D44_fs_30 = np.zeros(s)
AM55_fs_30 = np.zeros(s); D55_fs_30 = np.zeros(s)
AM66_fs_30 = np.zeros(s); D66_fs_30 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_30 = np.zeros(s); D13_fs_30 = np.zeros(s)
AM15_fs_30 = np.zeros(s); D15_fs_30 = np.zeros(s)
AM24_fs_30 = np.zeros(s); D24_fs_30 = np.zeros(s)
AM26_fs_30 = np.zeros(s); D26_fs_30 = np.zeros(s)
AM31_fs_30 = np.zeros(s); D31_fs_30 = np.zeros(s)
AM35_fs_30 = np.zeros(s); D35_fs_30 = np.zeros(s)
AM42_fs_30 = np.zeros(s); D42_fs_30 = np.zeros(s)
AM46_fs_30 = np.zeros(s); D46_fs_30 = np.zeros(s)
AM51_fs_30 = np.zeros(s); D51_fs_30 = np.zeros(s)
AM53_fs_30 = np.zeros(s); D53_fs_30 = np.zeros(s)
AM62_fs_30 = np.zeros(s); D62_fs_30 = np.zeros(s)
AM64_fs_30 = np.zeros(s); D64_fs_30 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_60 = np.zeros(s); D11_fs_60 = np.zeros(s)
AM22_fs_60 = np.zeros(s); D22_fs_60 = np.zeros(s)
AM33_fs_60 = np.zeros(s); D33_fs_60 = np.zeros(s)
AM44_fs_60 = np.zeros(s); D44_fs_60 = np.zeros(s)
AM55_fs_60 = np.zeros(s); D55_fs_60 = np.zeros(s)
AM66_fs_60 = np.zeros(s); D66_fs_60 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_60 = np.zeros(s); D13_fs_60 = np.zeros(s)
AM15_fs_60 = np.zeros(s); D15_fs_60 = np.zeros(s)
AM24_fs_60 = np.zeros(s); D24_fs_60 = np.zeros(s)
AM26_fs_60 = np.zeros(s); D26_fs_60 = np.zeros(s)
AM31_fs_60 = np.zeros(s); D31_fs_60 = np.zeros(s)
AM35_fs_60 = np.zeros(s); D35_fs_60 = np.zeros(s)
AM42_fs_60 = np.zeros(s); D42_fs_60 = np.zeros(s)
AM46_fs_60 = np.zeros(s); D46_fs_60 = np.zeros(s)
AM51_fs_60 = np.zeros(s); D51_fs_60 = np.zeros(s)
AM53_fs_60 = np.zeros(s); D53_fs_60 = np.zeros(s)
AM62_fs_60 = np.zeros(s); D62_fs_60 = np.zeros(s)
AM64_fs_60 = np.zeros(s); D64_fs_60 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_90 = np.zeros(s); D11_fs_90 = np.zeros(s)
AM22_fs_90 = np.zeros(s); D22_fs_90 = np.zeros(s)
AM33_fs_90 = np.zeros(s); D33_fs_90 = np.zeros(s)
AM44_fs_90 = np.zeros(s); D44_fs_90 = np.zeros(s)
AM55_fs_90 = np.zeros(s); D55_fs_90 = np.zeros(s)
AM66_fs_90 = np.zeros(s); D66_fs_90 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_90 = np.zeros(s); D13_fs_90 = np.zeros(s)
AM15_fs_90 = np.zeros(s); D15_fs_90 = np.zeros(s)
AM24_fs_90 = np.zeros(s); D24_fs_90 = np.zeros(s)
AM26_fs_90 = np.zeros(s); D26_fs_90 = np.zeros(s)
AM31_fs_90 = np.zeros(s); D31_fs_90 = np.zeros(s)
AM35_fs_90 = np.zeros(s); D35_fs_90 = np.zeros(s)
AM42_fs_90 = np.zeros(s); D42_fs_90 = np.zeros(s)
AM46_fs_90 = np.zeros(s); D46_fs_90 = np.zeros(s)
AM51_fs_90 = np.zeros(s); D51_fs_90 = np.zeros(s)
AM53_fs_90 = np.zeros(s); D53_fs_90 = np.zeros(s)
AM62_fs_90 = np.zeros(s); D62_fs_90 = np.zeros(s)
AM64_fs_90 = np.zeros(s); D64_fs_90 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_120 = np.zeros(s); D11_fs_120 = np.zeros(s)
AM22_fs_120 = np.zeros(s); D22_fs_120 = np.zeros(s)
AM33_fs_120 = np.zeros(s); D33_fs_120 = np.zeros(s)
AM44_fs_120 = np.zeros(s); D44_fs_120 = np.zeros(s)
AM55_fs_120 = np.zeros(s); D55_fs_120 = np.zeros(s)
AM66_fs_120 = np.zeros(s); D66_fs_120 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_120 = np.zeros(s); D13_fs_120 = np.zeros(s)
AM15_fs_120 = np.zeros(s); D15_fs_120 = np.zeros(s)
AM24_fs_120 = np.zeros(s); D24_fs_120 = np.zeros(s)
AM26_fs_120 = np.zeros(s); D26_fs_120 = np.zeros(s)
AM31_fs_120 = np.zeros(s); D31_fs_120 = np.zeros(s)
AM35_fs_120 = np.zeros(s); D35_fs_120 = np.zeros(s)
AM42_fs_120 = np.zeros(s); D42_fs_120 = np.zeros(s)
AM46_fs_120 = np.zeros(s); D46_fs_120 = np.zeros(s)
AM51_fs_120 = np.zeros(s); D51_fs_120 = np.zeros(s)
AM53_fs_120 = np.zeros(s); D53_fs_120 = np.zeros(s)
AM62_fs_120 = np.zeros(s); D62_fs_120 = np.zeros(s)
AM64_fs_120 = np.zeros(s); D64_fs_120 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_150 = np.zeros(s); D11_fs_150 = np.zeros(s)
AM22_fs_150 = np.zeros(s); D22_fs_150 = np.zeros(s)
AM33_fs_150 = np.zeros(s); D33_fs_150 = np.zeros(s)
AM44_fs_150 = np.zeros(s); D44_fs_150 = np.zeros(s)
AM55_fs_150 = np.zeros(s); D55_fs_150 = np.zeros(s)
AM66_fs_150 = np.zeros(s); D66_fs_150 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_150 = np.zeros(s); D13_fs_150 = np.zeros(s)
AM15_fs_150 = np.zeros(s); D15_fs_150 = np.zeros(s)
AM24_fs_150 = np.zeros(s); D24_fs_150 = np.zeros(s)
AM26_fs_150 = np.zeros(s); D26_fs_150 = np.zeros(s)
AM31_fs_150 = np.zeros(s); D31_fs_150 = np.zeros(s)
AM35_fs_150 = np.zeros(s); D35_fs_150 = np.zeros(s)
AM42_fs_150 = np.zeros(s); D42_fs_150 = np.zeros(s)
AM46_fs_150 = np.zeros(s); D46_fs_150 = np.zeros(s)
AM51_fs_150 = np.zeros(s); D51_fs_150 = np.zeros(s)
AM53_fs_150 = np.zeros(s); D53_fs_150 = np.zeros(s)
AM62_fs_150 = np.zeros(s); D62_fs_150 = np.zeros(s)
AM64_fs_150 = np.zeros(s); D64_fs_150 = np.zeros(s)

## added masses and dampings: nships x nfreqs, diagonal terms FULL SCALE
AM11_fs_180 = np.zeros(s); D11_fs_180 = np.zeros(s)
AM22_fs_180 = np.zeros(s); D22_fs_180 = np.zeros(s)
AM33_fs_180 = np.zeros(s); D33_fs_180 = np.zeros(s)
AM44_fs_180 = np.zeros(s); D44_fs_180 = np.zeros(s)
AM55_fs_180 = np.zeros(s); D55_fs_180 = np.zeros(s)
AM66_fs_180 = np.zeros(s); D66_fs_180 = np.zeros(s)
## added masses and dampings: nships x nfreqs, out diagonal terms FULL SCALE
AM13_fs_180 = np.zeros(s); D13_fs_180 = np.zeros(s)
AM15_fs_180 = np.zeros(s); D15_fs_180 = np.zeros(s)
AM24_fs_180 = np.zeros(s); D24_fs_180 = np.zeros(s)
AM26_fs_180 = np.zeros(s); D26_fs_180 = np.zeros(s)
AM31_fs_180 = np.zeros(s); D31_fs_180 = np.zeros(s)
AM35_fs_180 = np.zeros(s); D35_fs_180 = np.zeros(s)
AM42_fs_180 = np.zeros(s); D42_fs_180 = np.zeros(s)
AM46_fs_180 = np.zeros(s); D46_fs_180 = np.zeros(s)
AM51_fs_180 = np.zeros(s); D51_fs_180 = np.zeros(s)
AM53_fs_180 = np.zeros(s); D53_fs_180 = np.zeros(s)
AM62_fs_180 = np.zeros(s); D62_fs_180 = np.zeros(s)
AM64_fs_180 = np.zeros(s); D64_fs_180 = np.zeros(s)

## arrays with RAOS Functions
RAO_11 = np.zeros(t); RAO_phase_11 = np.zeros(t)
RAO_22 = np.zeros(t); RAO_phase_22 = np.zeros(t)
RAO_33 = np.zeros(t); RAO_phase_33 = np.zeros(t)
RAO_44 = np.zeros(t); RAO_phase_44 = np.zeros(t)
RAO_55 = np.zeros(t); RAO_phase_55 = np.zeros(t)
RAO_66 = np.zeros(t); RAO_phase_66 = np.zeros(t)

## arrays with RAOS Functions
RAO_11_traslated = np.zeros(t); RAO_phase_11_traslated = np.zeros(t)
RAO_22_traslated = np.zeros(t); RAO_phase_22_traslated = np.zeros(t)
RAO_33_traslated = np.zeros(t); RAO_phase_33_traslated = np.zeros(t)
RAO_44_traslated = np.zeros(t); RAO_phase_44_traslated = np.zeros(t)
RAO_55_traslated = np.zeros(t); RAO_phase_55_traslated = np.zeros(t)
RAO_66_traslated = np.zeros(t); RAO_phase_66_traslated = np.zeros(t)

## arrays with RAOS Functions
RAO_44_rep = np.zeros(t)
RAO_55_rep = np.zeros(t)
RAO_66_rep = np.zeros(t)

############# other variables
Mod = np.zeros((1)); Dir = np.zeros((1)); Tp = np.zeros((1)); Hs= np.zeros((1)); Wave_sp = np.zeros(t)

mov_11_max = np.zeros(s); mov_22_max = np.zeros(s); mov_33_max = np.zeros(s)
mov_44_max = np.zeros(s); mov_55_max = np.zeros(s); mov_66_max = np.zeros(s)

mov_11_sig = np.zeros(s); mov_22_sig = np.zeros(s); mov_33_sig = np.zeros(s)
mov_44_sig = np.zeros(s); mov_55_sig = np.zeros(s); mov_66_sig = np.zeros(s)

acc_RMS_11 = np.zeros(s); acc_RMS_22 = np.zeros(s); acc_RMS_33 = np.zeros(s)
acc_RMS_44 = np.zeros(s); acc_RMS_55 = np.zeros(s); acc_RMS_66 = np.zeros(s)

m0 = np.zeros(t); m2 = np.zeros(t); m4 = np.zeros(t)

Significative_mag = np.zeros(s); Motion_sickness = np.zeros(s)

# point where RAOs are calculated
Point_sk = np.zeros(s)

# Center of gravity of ship
Point_CDG = np.zeros(s)
