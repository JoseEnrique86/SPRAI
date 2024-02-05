
import shutil
import os
import glob
import numpy as np
import pandas as pd
import common as cm

#wkd = cm.workfiles.workdirectory
nheads = cm.sea_var.nheads
angles = cm.sea_var.angles

def join_csv(wkd,name):

    os.chdir(wkd)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    folder = os.path.join(wkd,name)
    with open(folder, 'w') as outfile:
        for fname in all_filenames:
            with open(fname) as infile:
                outfile.write(infile.read())
            
            os.remove(fname)
    #combine all files in the list
    #combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], axis=1)
    #export to csv
    #combined_csv.to_csv(name, index=False)
    # encoding='utf-8-sig'

def s_AM(wkd,nships):

    # create folder to store all added masses
    folder = os.path.join(wkd,"Added Masses")
    os.mkdir(folder)
    
    index1 = 0
    index2 = 0
    freq = np.zeros((30,1))
    
    for i in range(nships):
        
        name = "added_mass_ship_"
        name = name + str(i)
        added_masses = os.path.join(folder,name)
        os.mkdir(added_masses)

        if i in cm.sea_var.list_fn_0:

            freq[:,0] = cm.fq1[:,index1]
            message = "Added Mass 11:\n"
            arraymatrix = np.stack((cm.AM11_fs[:,0,index1],cm.AM11_fs[:,1,index1],cm.AM11_fs[:,2,index1],cm.AM11_fs[:,3,index1],cm.AM11_fs[:,4,index1],cm.AM11_fs[:,5,index1],cm.AM11_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_11.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 22:\n"
            arraymatrix = np.stack((cm.AM22_fs[:,0,index1],cm.AM22_fs[:,1,index1],cm.AM22_fs[:,2,index1],cm.AM22_fs[:,3,index1],cm.AM22_fs[:,4,index1],cm.AM22_fs[:,5,index1],cm.AM22_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_22.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 33:\n"
            arraymatrix = np.stack((cm.AM33_fs[:,0,index1],cm.AM33_fs[:,1,index1],cm.AM33_fs[:,2,index1],cm.AM33_fs[:,3,index1],cm.AM33_fs[:,4,index1],cm.AM33_fs[:,5,index1],cm.AM33_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_33.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 44:\n"
            arraymatrix = np.stack((cm.AM44_fs[:,0,index1],cm.AM44_fs[:,1,index1],cm.AM44_fs[:,2,index1],cm.AM44_fs[:,3,index1],cm.AM44_fs[:,4,index1],cm.AM44_fs[:,5,index1],cm.AM44_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_44.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 55:\n"
            arraymatrix = np.stack((cm.AM55_fs[:,0,index1],cm.AM55_fs[:,1,index1],cm.AM55_fs[:,2,index1],cm.AM55_fs[:,3,index1],cm.AM55_fs[:,4,index1],cm.AM55_fs[:,5,index1],cm.AM55_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_55.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 66:\n"
            arraymatrix = np.stack((cm.AM66_fs[:,0,index1],cm.AM66_fs[:,1,index1],cm.AM66_fs[:,2,index1],cm.AM66_fs[:,3,index1],cm.AM66_fs[:,4,index1],cm.AM66_fs[:,5,index1],cm.AM66_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_66.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 13:\n"
            arraymatrix = np.stack((cm.AM13_fs[:,0,index1],cm.AM13_fs[:,1,index1],cm.AM13_fs[:,2,index1],cm.AM13_fs[:,3,index1],cm.AM13_fs[:,4,index1],cm.AM13_fs[:,5,index1],cm.AM13_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_13.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 15:\n"
            arraymatrix = np.stack((cm.AM15_fs[:,0,index1],cm.AM15_fs[:,1,index1],cm.AM15_fs[:,2,index1],cm.AM15_fs[:,3,index1],cm.AM15_fs[:,4,index1],cm.AM15_fs[:,5,index1],cm.AM15_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_15.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 24:\n"
            arraymatrix = np.stack((cm.AM24_fs[:,0,index1],cm.AM24_fs[:,1,index1],cm.AM24_fs[:,2,index1],cm.AM24_fs[:,3,index1],cm.AM24_fs[:,4,index1],cm.AM24_fs[:,5,index1],cm.AM24_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_24.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 26:\n"
            arraymatrix = np.stack((cm.AM26_fs[:,0,index1],cm.AM26_fs[:,1,index1],cm.AM26_fs[:,2,index1],cm.AM26_fs[:,3,index1],cm.AM26_fs[:,4,index1],cm.AM26_fs[:,5,index1],cm.AM26_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_26.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 31:\n"
            arraymatrix = np.stack((cm.AM31_fs[:,0,index1],cm.AM31_fs[:,1,index1],cm.AM31_fs[:,2,index1],cm.AM31_fs[:,3,index1],cm.AM31_fs[:,4,index1],cm.AM31_fs[:,5,index1],cm.AM31_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_31.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 35:\n"
            arraymatrix = np.stack((cm.AM35_fs[:,0,index1],cm.AM35_fs[:,1,index1],cm.AM35_fs[:,2,index1],cm.AM35_fs[:,3,index1],cm.AM35_fs[:,4,index1],cm.AM35_fs[:,5,index1],cm.AM35_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_35.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 42:\n"
            arraymatrix = np.stack((cm.AM42_fs[:,0,index1],cm.AM42_fs[:,1,index1],cm.AM42_fs[:,2,index1],cm.AM42_fs[:,3,index1],cm.AM42_fs[:,4,index1],cm.AM42_fs[:,5,index1],cm.AM42_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_42.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 46:\n"
            arraymatrix = np.stack((cm.AM46_fs[:,0,index1],cm.AM46_fs[:,1,index1],cm.AM46_fs[:,2,index1],cm.AM46_fs[:,3,index1],cm.AM46_fs[:,4,index1],cm.AM46_fs[:,5,index1],cm.AM46_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_46.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 51:\n"
            arraymatrix = np.stack((cm.AM51_fs[:,0,index1],cm.AM51_fs[:,1,index1],cm.AM51_fs[:,2,index1],cm.AM51_fs[:,3,index1],cm.AM51_fs[:,4,index1],cm.AM51_fs[:,5,index1],cm.AM51_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_51.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 53:\n"
            arraymatrix = np.stack((cm.AM53_fs[:,0,index1],cm.AM53_fs[:,1,index1],cm.AM53_fs[:,2,index1],cm.AM53_fs[:,3,index1],cm.AM53_fs[:,4,index1],cm.AM53_fs[:,5,index1],cm.AM53_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_53.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 62:\n"
            arraymatrix = np.stack((cm.AM62_fs[:,0,index1],cm.AM62_fs[:,1,index1],cm.AM62_fs[:,2,index1],cm.AM62_fs[:,3,index1],cm.AM62_fs[:,4,index1],cm.AM62_fs[:,5,index1],cm.AM62_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_62.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 64:\n"
            arraymatrix = np.stack((cm.AM64_fs[:,0,index1],cm.AM64_fs[:,1,index1],cm.AM64_fs[:,2,index1],cm.AM64_fs[:,3,index1],cm.AM64_fs[:,4,index1],cm.AM64_fs[:,5,index1],cm.AM64_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_64.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

            index1 += 1

            name = "added_masses_ship_"
            name_i = name + str(i) + '.csv'
            join_csv(added_masses,name_i)


        if i in cm.sea_var.list_fn:
            
            freq[:,0] = cm.fq2[:,index2]
            message = "Added Mass 11:\n"
            arraymatrix = np.stack((cm.AM11_fs_0[:,index2],cm.AM11_fs_30[:,index2],cm.AM11_fs_60[:,index2],cm.AM11_fs_90[:,index2],cm.AM11_fs_120[:,index2],cm.AM11_fs_150[:,index2],cm.AM11_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_11.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 22:\n"
            arraymatrix = np.stack((cm.AM22_fs_0[:,index2],cm.AM22_fs_30[:,index2],cm.AM22_fs_60[:,index2],cm.AM22_fs_90[:,index2],cm.AM22_fs_120[:,index2],cm.AM22_fs_150[:,index2],cm.AM22_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_22.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 33:\n"
            arraymatrix = np.stack((cm.AM33_fs_0[:,index2],cm.AM33_fs_30[:,index2],cm.AM33_fs_60[:,index2],cm.AM33_fs_90[:,index2],cm.AM33_fs_120[:,index2],cm.AM33_fs_150[:,index2],cm.AM33_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_33.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 44:\n"
            arraymatrix = np.stack((cm.AM44_fs_0[:,index2],cm.AM44_fs_30[:,index2],cm.AM44_fs_60[:,index2],cm.AM44_fs_90[:,index2],cm.AM44_fs_120[:,index2],cm.AM44_fs_150[:,index2],cm.AM44_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_44.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 55:\n"
            arraymatrix = np.stack((cm.AM55_fs_0[:,index2],cm.AM55_fs_30[:,index2],cm.AM55_fs_60[:,index2],cm.AM55_fs_90[:,index2],cm.AM55_fs_120[:,index2],cm.AM55_fs_150[:,index2],cm.AM55_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_55.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 66:\n"
            arraymatrix = np.stack((cm.AM66_fs_0[:,index2],cm.AM66_fs_30[:,index2],cm.AM66_fs_60[:,index2],cm.AM66_fs_90[:,index2],cm.AM66_fs_120[:,index2],cm.AM66_fs_150[:,index2],cm.AM66_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_66.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 13:\n"
            arraymatrix = np.stack((cm.AM13_fs_0[:,index2],cm.AM13_fs_30[:,index2],cm.AM13_fs_60[:,index2],cm.AM13_fs_90[:,index2],cm.AM13_fs_120[:,index2],cm.AM13_fs_150[:,index2],cm.AM13_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_13.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 15:\n"
            arraymatrix = np.stack((cm.AM15_fs_0[:,index2],cm.AM15_fs_30[:,index2],cm.AM15_fs_60[:,index2],cm.AM15_fs_90[:,index2],cm.AM15_fs_120[:,index2],cm.AM15_fs_150[:,index2],cm.AM15_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_15.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 24:\n"
            arraymatrix = np.stack((cm.AM24_fs_0[:,index2],cm.AM24_fs_30[:,index2],cm.AM24_fs_60[:,index2],cm.AM24_fs_90[:,index2],cm.AM24_fs_120[:,index2],cm.AM24_fs_150[:,index2],cm.AM24_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_24.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 26:\n"
            arraymatrix = np.stack((cm.AM26_fs_0[:,index2],cm.AM26_fs_30[:,index2],cm.AM26_fs_60[:,index2],cm.AM26_fs_90[:,index2],cm.AM26_fs_120[:,index2],cm.AM26_fs_150[:,index2],cm.AM26_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_26.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 31:\n"
            arraymatrix = np.stack((cm.AM31_fs_0[:,index2],cm.AM31_fs_30[:,index2],cm.AM31_fs_60[:,index2],cm.AM31_fs_90[:,index2],cm.AM31_fs_120[:,index2],cm.AM31_fs_150[:,index2],cm.AM31_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_31.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 35:\n"
            arraymatrix = np.stack((cm.AM35_fs_0[:,index2],cm.AM35_fs_30[:,index2],cm.AM35_fs_60[:,index2],cm.AM35_fs_90[:,index2],cm.AM35_fs_120[:,index2],cm.AM35_fs_150[:,index2],cm.AM35_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_35.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 42:\n"
            arraymatrix = np.stack((cm.AM42_fs_0[:,index2],cm.AM42_fs_30[:,index2],cm.AM42_fs_60[:,index2],cm.AM42_fs_90[:,index2],cm.AM42_fs_120[:,index2],cm.AM42_fs_150[:,index2],cm.AM42_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_42.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 46:\n"
            arraymatrix = np.stack((cm.AM46_fs_0[:,index2],cm.AM46_fs_30[:,index2],cm.AM46_fs_60[:,index2],cm.AM46_fs_90[:,index2],cm.AM46_fs_120[:,index2],cm.AM46_fs_150[:,index2],cm.AM46_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_46.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Added Mass 51:\n"
            arraymatrix = np.stack((cm.AM51_fs_0[:,index2],cm.AM51_fs_30[:,index2],cm.AM51_fs_60[:,index2],cm.AM51_fs_90[:,index2],cm.AM51_fs_120[:,index2],cm.AM51_fs_150[:,index2],cm.AM51_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_51.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 53:\n"
            arraymatrix = np.stack((cm.AM53_fs_0[:,index2],cm.AM53_fs_30[:,index2],cm.AM53_fs_60[:,index2],cm.AM53_fs_90[:,index2],cm.AM53_fs_120[:,index2],cm.AM53_fs_150[:,index2],cm.AM53_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_53.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 62:\n"
            arraymatrix = np.stack((cm.AM62_fs_0[:,index2],cm.AM62_fs_30[:,index2],cm.AM62_fs_60[:,index2],cm.AM62_fs_90[:,index2],cm.AM62_fs_120[:,index2],cm.AM62_fs_150[:,index2],cm.AM62_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_62.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Added Mass 64:\n"
            arraymatrix = np.stack((cm.AM64_fs_0[:,index2],cm.AM64_fs_30[:,index2],cm.AM64_fs_60[:,index2],cm.AM64_fs_90[:,index2],cm.AM64_fs_120[:,index2],cm.AM64_fs_150[:,index2],cm.AM64_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(added_masses),'Added_Mass_64.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

            index2 += 1

            name = "added_masses_ship_"
            name_i = name + str(i)+ '.csv'
            join_csv(added_masses,name_i)
        
def s_D(wkd,nships):

     # create folder to store all damping coefficients
    folder = os.path.join(wkd,"Damping")
    os.mkdir(folder)

    index1 = 0
    index2 = 0
    freq = np.zeros((30,1))
    
    for i in range(nships):

        name = "Damping_ship_"
        name_i = name + str(i)
        damping = os.path.join(folder,name_i)
        os.mkdir(damping)

        if i in cm.sea_var.list_fn_0:

            freq[:,0] = cm.fq1[:,index1]

            message = "Damping 11:\n"
            arraymatrix = np.stack((cm.D11_fs[:,0,index1],cm.D11_fs[:,1,index1],cm.D11_fs[:,2,index1],cm.D11_fs[:,3,index1],cm.D11_fs[:,4,index1],cm.D11_fs[:,5,index1],cm.D11_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_11.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 22:\n"
            arraymatrix = np.stack((cm.D22_fs[:,0,index1],cm.D22_fs[:,1,index1],cm.D22_fs[:,2,index1],cm.D22_fs[:,3,index1],cm.D22_fs[:,4,index1],cm.D22_fs[:,5,index1],cm.D22_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_22.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "DDping 33:\n"
            arraymatrix = np.stack((cm.D33_fs[:,0,index1],cm.D33_fs[:,1,index1],cm.D33_fs[:,2,index1],cm.D33_fs[:,3,index1],cm.D33_fs[:,4,index1],cm.D33_fs[:,5,index1],cm.D33_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_33.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 44:\n"
            arraymatrix = np.stack((cm.D44_fs[:,0,index1],cm.D44_fs[:,1,index1],cm.D44_fs[:,2,index1],cm.D44_fs[:,3,index1],cm.D44_fs[:,4,index1],cm.D44_fs[:,5,index1],cm.D44_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_44.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 55:\n"
            arraymatrix =np.stack((cm.D55_fs[:,0,index1],cm.D55_fs[:,1,index1],cm.D55_fs[:,2,index1],cm.D55_fs[:,3,index1],cm.D55_fs[:,4,index1],cm.D55_fs[:,5,index1],cm.D55_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_55.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 66:\n"
            arraymatrix = np.stack((cm.D66_fs[:,0,index1],cm.D66_fs[:,1,index1],cm.D66_fs[:,2,index1],cm.D66_fs[:,3,index1],cm.D66_fs[:,4,index1],cm.D66_fs[:,5,index1],cm.D66_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_66.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 13:\n"
            arraymatrix = np.stack((cm.D13_fs[:,0,index1],cm.D13_fs[:,1,index1],cm.D13_fs[:,2,index1],cm.D13_fs[:,3,index1],cm.D13_fs[:,4,index1],cm.D13_fs[:,5,index1],cm.D13_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_13.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 15:\n"
            arraymatrix = np.stack((cm.D15_fs[:,0,index1],cm.D15_fs[:,1,index1],cm.D15_fs[:,2,index1],cm.D15_fs[:,3,index1],cm.D15_fs[:,4,index1],cm.D15_fs[:,5,index1],cm.D15_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_15.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 24:\n"
            arraymatrix = np.stack((cm.D24_fs[:,0,index1],cm.D24_fs[:,1,index1],cm.D24_fs[:,2,index1],cm.D24_fs[:,3,index1],cm.D24_fs[:,4,index1],cm.D24_fs[:,5,index1],cm.D24_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_24.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 26:\n"
            arraymatrix = np.stack((cm.D26_fs[:,0,index1],cm.D26_fs[:,1,index1],cm.D26_fs[:,2,index1],cm.D26_fs[:,3,index1],cm.D26_fs[:,4,index1],cm.D26_fs[:,5,index1],cm.D26_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_26.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 31:\n"
            arraymatrix = np.stack((cm.D31_fs[:,0,index1],cm.D31_fs[:,1,index1],cm.D31_fs[:,2,index1],cm.D31_fs[:,3,index1],cm.D31_fs[:,4,index1],cm.D31_fs[:,5,index1],cm.D31_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_31.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 35:\n"
            arraymatrix = np.stack((cm.D35_fs[:,0,index1],cm.D35_fs[:,1,index1],cm.D35_fs[:,2,index1],cm.D35_fs[:,3,index1],cm.D35_fs[:,4,index1],cm.D35_fs[:,5,index1],cm.D35_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_35.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 42:\n"
            arraymatrix = np.stack((cm.D42_fs[:,0,index1],cm.D42_fs[:,1,index1],cm.D42_fs[:,2,index1],cm.D42_fs[:,3,index1],cm.D42_fs[:,4,index1],cm.D42_fs[:,5,index1],cm.D42_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_42.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 46:\n"
            arraymatrix = np.stack((cm.D46_fs[:,0,index1],cm.D46_fs[:,1,index1],cm.D46_fs[:,2,index1],cm.D46_fs[:,3,index1],cm.D46_fs[:,4,index1],cm.D46_fs[:,5,index1],cm.D46_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_46.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 51:\n"
            arraymatrix = np.stack((cm.D51_fs[:,0,index1],cm.D51_fs[:,1,index1],cm.D51_fs[:,2,index1],cm.D51_fs[:,3,index1],cm.D51_fs[:,4,index1],cm.D51_fs[:,5,index1],cm.D51_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_51.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 53:\n"
            arraymatrix = np.stack((cm.D53_fs[:,0,index1],cm.D53_fs[:,1,index1],cm.D53_fs[:,2,index1],cm.D53_fs[:,3,index1],cm.D53_fs[:,4,index1],cm.D53_fs[:,5,index1],cm.D53_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_53.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 62:\n"
            arraymatrix =np.stack((cm.D62_fs[:,0,index1],cm.D62_fs[:,1,index1],cm.D62_fs[:,2,index1],cm.D62_fs[:,3,index1],cm.D62_fs[:,4,index1],cm.D62_fs[:,5,index1],cm.D62_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_62.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 64:\n"
            arraymatrix = np.stack((cm.D64_fs[:,0,index1],cm.D64_fs[:,1,index1],cm.D64_fs[:,2,index1],cm.D64_fs[:,3,index1],cm.D64_fs[:,4,index1],cm.D64_fs[:,5,index1],cm.D64_fs[:,6,index1]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_64.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

            index1 += 1

            name = "damping_ship_"
            name_i = name + str(i) + '.csv'
            join_csv(damping,name_i)


        if i in cm.sea_var.list_fn:
            
            freq[:,0] = cm.fq2[:,index2]

            message = "Damping 11:\n"
            arraymatrix = np.stack((cm.D11_fs_0[:,index2],cm.D11_fs_30[:,index2],cm.D11_fs_60[:,index2],cm.D11_fs_90[:,index2],cm.D11_fs_120[:,index2],cm.D11_fs_150[:,index2],cm.D11_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_11.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 22:\n"
            arraymatrix = np.stack((cm.D22_fs_0[:,index2],cm.D22_fs_30[:,index2],cm.D22_fs_60[:,index2],cm.D22_fs_90[:,index2],cm.D22_fs_120[:,index2],cm.D22_fs_150[:,index2],cm.D22_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_22.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 33:\n"
            arraymatrix = np.stack((cm.D33_fs_0[:,index2],cm.D33_fs_30[:,index2],cm.D33_fs_60[:,index2],cm.D33_fs_90[:,index2],cm.D33_fs_120[:,index2],cm.D33_fs_150[:,index2],cm.D33_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_33.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 44:\n"
            arraymatrix = np.stack((cm.D44_fs_0[:,index2],cm.D44_fs_30[:,index2],cm.D44_fs_60[:,index2],cm.D44_fs_90[:,index2],cm.D44_fs_120[:,index2],cm.D44_fs_150[:,index2],cm.D44_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_44.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 55:\n"
            arraymatrix = np.stack((cm.D55_fs_0[:,index2],cm.D55_fs_30[:,index2],cm.D55_fs_60[:,index2],cm.D55_fs_90[:,index2],cm.D55_fs_120[:,index2],cm.D55_fs_150[:,index2],cm.D55_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_55.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 66:\n"
            arraymatrix = np.stack((cm.D66_fs_0[:,index2],cm.D66_fs_30[:,index2],cm.D66_fs_60[:,index2],cm.D66_fs_90[:,index2],cm.D66_fs_120[:,index2],cm.D66_fs_150[:,index2],cm.D66_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_66.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 13:\n"
            arraymatrix = np.stack((cm.D13_fs_0[:,index2],cm.D13_fs_30[:,index2],cm.D13_fs_60[:,index2],cm.D13_fs_90[:,index2],cm.D13_fs_120[:,index2],cm.D13_fs_150[:,index2],cm.D13_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_13.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 15:\n"
            arraymatrix = np.stack((cm.D15_fs_0[:,index2],cm.D15_fs_30[:,index2],cm.D15_fs_60[:,index2],cm.D15_fs_90[:,index2],cm.D15_fs_120[:,index2],cm.D15_fs_150[:,index2],cm.D15_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_15.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 24:\n"
            arraymatrix = np.stack((cm.D24_fs_0[:,index2],cm.D24_fs_30[:,index2],cm.D24_fs_60[:,index2],cm.D24_fs_90[:,index2],cm.D24_fs_120[:,index2],cm.D24_fs_150[:,index2],cm.D24_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_24.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 26:\n"
            arraymatrix = np.stack((cm.D26_fs_0[:,index2],cm.D26_fs_30[:,index2],cm.D26_fs_60[:,index2],cm.D26_fs_90[:,index2],cm.D26_fs_120[:,index2],cm.D26_fs_150[:,index2],cm.D26_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_26.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 31:\n"
            arraymatrix = np.stack((cm.D31_fs_0[:,index2],cm.D31_fs_30[:,index2],cm.D31_fs_60[:,index2],cm.D31_fs_90[:,index2],cm.D31_fs_120[:,index2],cm.D31_fs_150[:,index2],cm.D31_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_31.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 35:\n"
            arraymatrix = np.stack((cm.D35_fs_0[:,index2],cm.D35_fs_30[:,index2],cm.D35_fs_60[:,index2],cm.D35_fs_90[:,index2],cm.D35_fs_120[:,index2],cm.D35_fs_150[:,index2],cm.D35_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_35.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 42:\n"
            arraymatrix = np.stack((cm.D42_fs_0[:,index2],cm.D42_fs_30[:,index2],cm.D42_fs_60[:,index2],cm.D42_fs_90[:,index2],cm.D42_fs_120[:,index2],cm.D42_fs_150[:,index2],cm.D42_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_42.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 46:\n"
            arraymatrix = np.stack((cm.D46_fs_0[:,index2],cm.D46_fs_30[:,index2],cm.D46_fs_60[:,index2],cm.D46_fs_90[:,index2],cm.D46_fs_120[:,index2],cm.D46_fs_150[:,index2],cm.D46_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_46.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            message = "Damping 51:\n"
            arraymatrix = np.stack((cm.D51_fs_0[:,index2],cm.D51_fs_30[:,index2],cm.D51_fs_60[:,index2],cm.D51_fs_90[:,index2],cm.D51_fs_120[:,index2],cm.D51_fs_150[:,index2],cm.D51_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_51.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 53:\n"
            arraymatrix = np.stack((cm.D53_fs_0[:,index2],cm.D53_fs_30[:,index2],cm.D53_fs_60[:,index2],cm.D53_fs_90[:,index2],cm.D53_fs_120[:,index2],cm.D53_fs_150[:,index2],cm.D53_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_53.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 62:\n"
            arraymatrix = np.stack((cm.D62_fs_0[:,index2],cm.D62_fs_30[:,index2],cm.D62_fs_60[:,index2],cm.D62_fs_90[:,index2],cm.D62_fs_120[:,index2],cm.D62_fs_150[:,index2],cm.D62_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_62.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
            
            message = "Damping 64:\n"
            arraymatrix = np.stack((cm.D64_fs_0[:,index2],cm.D64_fs_30[:,index2],cm.D64_fs_60[:,index2],cm.D64_fs_90[:,index2],cm.D64_fs_120[:,index2],cm.D64_fs_150[:,index2],cm.D64_fs_180[:,index2]), axis=-1)
            a = np.append(freq,arraymatrix,axis=1)
            np.savetxt(os.path.join(str(damping),'Damping_64.csv'), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
           
            index2 += 1

            name = "damping_ship_"
            name_i = name + str(i) + '.csv'
            join_csv(damping,name_i)
            
def s_FE(wkd,nships):
    # create folder to store all forces
    forces = os.path.join(wkd,"forces")
    os.mkdir(forces)

    if os.path.exists(forces):

        index1 = 0
        index2 = 0
        freq = np.zeros((30,1))
        
        name_fx= 'FEX.csv'; name_fy= 'FEY.csv'; name_fz= 'FEZ.csv'
        name_mx= 'MEX.csv'; name_my= 'MEY.csv'; name_mz= 'MEZ.csv'

        for i in range(nships):
            
            name = "Forces_ship_"
            name_i = name + str(i)
            forces_i = os.path.join(forces,name_i)
            os.mkdir(forces_i)

            if i in cm.sea_var.list_fn_0:
                freq[:,0] = cm.fq1[:,index1]

                message = "Forces X:\n"
                arraymatrix = np.stack((cm.FEX_fs[:,0,index1],cm.FEX_fs[:,1,index1],cm.FEX_fs[:,2,index1],cm.FEX_fs[:,3,index1],cm.FEX_fs[:,4,index1],cm.FEX_fs[:,5,index1],cm.FEX_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fx)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Forces Y:\n"
                arraymatrix = np.stack((cm.FEY_fs[:,0,index1],cm.FEY_fs[:,1,index1],cm.FEY_fs[:,2,index1],cm.FEY_fs[:,3,index1],cm.FEY_fs[:,4,index1],cm.FEY_fs[:,5,index1],cm.FEY_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fy)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "Forces Z:\n"
                arraymatrix = np.stack((cm.FEZ_fs[:,0,index1],cm.FEZ_fs[:,1,index1],cm.FEZ_fs[:,2,index1],cm.FEZ_fs[:,3,index1],cm.FEZ_fs[:,4,index1],cm.FEZ_fs[:,5,index1],cm.FEZ_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fz)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment X:\n"
                arraymatrix = np.stack((cm.MEX_fs[:,0,index1],cm.MEX_fs[:,1,index1],cm.MEX_fs[:,2,index1],cm.MEX_fs[:,3,index1],cm.MEX_fs[:,4,index1],cm.MEX_fs[:,5,index1],cm.MEX_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_mx)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment Y:\n"
                arraymatrix = np.stack((cm.MEY_fs[:,0,index1],cm.MEY_fs[:,1,index1],cm.MEY_fs[:,2,index1],cm.MEY_fs[:,3,index1],cm.MEY_fs[:,4,index1],cm.MEY_fs[:,5,index1],cm.MEY_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_my)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment Z:\n"
                arraymatrix = np.stack((cm.MEZ_fs[:,0,index1],cm.MEZ_fs[:,1,index1],cm.MEZ_fs[:,2,index1],cm.MEZ_fs[:,3,index1],cm.MEZ_fs[:,4,index1],cm.MEZ_fs[:,5,index1],cm.MEZ_fs[:,6,index1]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_mz)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                index1 += 1

                name = "Forces_ship_"
                name_i = name + str(i) + '.csv'
                join_csv(forces_i,name_i)
            
            if i in cm.sea_var.list_fn:
                
                freq[:,0] = cm.fq2[:,index2]
                
                message = "Forces X:\n"
                arraymatrix = np.stack((cm.FEX_fs_din[:,0,index2],cm.FEX_fs_din[:,1,index2],cm.FEX_fs_din[:,2,index2],cm.FEX_fs_din[:,3,index2],cm.FEX_fs_din[:,4,index2],cm.FEX_fs_din[:,5,index2],cm.FEX_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fx)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Forces Y:\n"
                arraymatrix = np.stack((cm.FEY_fs_din[:,0,index2],cm.FEY_fs_din[:,1,index2],cm.FEY_fs_din[:,2,index2],cm.FEY_fs_din[:,3,index2],cm.FEY_fs_din[:,4,index2],cm.FEY_fs_din[:,5,index2],cm.FEY_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fy)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "Forces Z:\n"
                arraymatrix = np.stack((cm.FEZ_fs_din[:,0,index2],cm.FEZ_fs_din[:,1,index2],cm.FEZ_fs_din[:,2,index2],cm.FEZ_fs_din[:,3,index2],cm.FEZ_fs_din[:,4,index2],cm.FEZ_fs_din[:,5,index2],cm.FEZ_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_fz)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment X:\n"
                arraymatrix = np.stack((cm.MEX_fs_din[:,0,index2],cm.MEX_fs_din[:,1,index2],cm.MEX_fs_din[:,2,index2],cm.MEX_fs_din[:,3,index2],cm.MEX_fs_din[:,4,index2],cm.MEX_fs_din[:,5,index2],cm.MEX_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_mx)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment Y:\n"
                arraymatrix = np.stack((cm.MEY_fs_din[:,0,index2],cm.MEY_fs_din[:,1,index2],cm.MEY_fs_din[:,2,index2],cm.MEY_fs_din[:,3,index2],cm.MEY_fs_din[:,4,index2],cm.MEY_fs_din[:,5,index2],cm.MEY_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_my)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "Moment Z:\n"
                arraymatrix = np.stack((cm.MEZ_fs_din[:,0,index2],cm.MEZ_fs_din[:,1,index2],cm.MEZ_fs_din[:,2,index2],cm.MEZ_fs_din[:,3,index2],cm.MEZ_fs_din[:,4,index2],cm.MEZ_fs_din[:,5,index2],cm.MEZ_fs_din[:,6,index2]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(forces_i),str(name_mz)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                index2 += 1

                name = "Forces_ship_"
                name_i = name + str(i) + '.csv'
                join_csv(forces_i,name_i)

def s_K(wkd,nships):
    # create folder to store all restoring forces
    stiffness = os.path.join(wkd,"stiffness")
    os.mkdir(stiffness)
    
    index1 = 0
    index2 = 0
    freq = np.zeros((30,1))
    
    if os.path.exists(stiffness):

        for i in range(nships):
            
            name = "Stiffness_ship_"
            name_i = name + str(i) + '.csv'

            if i in cm.sea_var.list_fn_0:
                
                freq[:,0] = cm.fq1[:,index1]
                arraymatrix = np.stack((cm.K33_fs[:,0,i],cm.K34_fs[:,0,i],cm.K35_fs[:,0,i],cm.K44_fs[:,0,i],cm.K45_fs[:,0,i],cm.K46_fs[:,0,i],cm.K53_fs[:,0,i],cm.K55_fs[:,0,i],cm.K56_fs[:,0,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(stiffness),str(name_i)), a, delimiter=';', header="Freq[rad/s];K33;K34;K35;K44;K45;K46;K53;K55;K56")

                index1 +=1
            
            if i in cm.sea_var.list_fn:

                freq[:,0] = cm.fq2[:,index2]
                arraymatrix = np.stack((cm.K33_fs[:,0,i],cm.K34_fs[:,0,i],cm.K35_fs[:,0,i],cm.K44_fs[:,0,i],cm.K45_fs[:,0,i],cm.K46_fs[:,0,i],cm.K53_fs[:,0,i],cm.K55_fs[:,0,i],cm.K56_fs[:,0,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(stiffness),str(name_i)), a, delimiter=';', header="Freq[rad/s];K33;K34;K35;K44;K45;K46;K53;K55;K56")
                
                index2 +=1
        
def s_RAO(wkd,nships):
    # create folder to store all RAO curves
    raocurves = os.path.join(wkd,"RAOs")
    os.mkdir(raocurves)

    index1 = 0
    index2 = 0
    freq = np.zeros((30,1))
    
    if os.path.exists(raocurves):
        
        name_11= 'RAO_11.csv'; name_22= 'RAO_22.csv'; name_33= 'RAO_33.csv'
        name_44= 'RAO_44.csv'; name_55= 'RAO_55.csv'; name_66= 'RAO_66.csv'

        phase_11= 'Phase_RAO_11.csv'; phase_22= 'Phase_RAO_22.csv'; phase_33= 'Phase_RAO_33.csv'
        phase_44= 'Phase_RAO_44.csv'; phase_55= 'Phase_RAO_55.csv'; phase_66= 'Phase_RAO_66.csv'

        for i in range (nships):
            
            name = "RAOs_ship_"
            name_i = name + str(i)
            raos = os.path.join(raocurves,name_i)
            os.mkdir(raos)

            if i in cm.sea_var.list_fn_0:
                freq[:,0] = cm.fq1[:,index1]

                message = "RAO SURGE:\n"
                arraymatrix = np.stack((cm.RAO_11[:,0,i],cm.RAO_11[:,1,i],cm.RAO_11[:,2,i],cm.RAO_11[:,3,i],cm.RAO_11[:,4,i],cm.RAO_11[:,5,i],cm.RAO_11[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_11)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "PHASE SURGE:\n"
                arraymatrix = np.stack((cm.RAO_phase_11[:,0,i],cm.RAO_phase_11[:,1,i],cm.RAO_phase_11[:,2,i],cm.RAO_phase_11[:,3,i],cm.RAO_phase_11[:,4,i],cm.RAO_phase_11[:,5,i],cm.RAO_phase_11[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_11)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO SWAY:\n"
                arraymatrix = np.stack((cm.RAO_22[:,0,i],cm.RAO_22[:,1,i],cm.RAO_22[:,2,i],cm.RAO_22[:,3,i],cm.RAO_22[:,4,i],cm.RAO_22[:,5,i],cm.RAO_22[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_22)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE SWAY:\n"
                arraymatrix = np.stack((cm.RAO_phase_22[:,0,i],cm.RAO_phase_22[:,1,i],cm.RAO_phase_22[:,2,i],cm.RAO_phase_22[:,3,i],cm.RAO_phase_22[:,4,i],cm.RAO_phase_22[:,5,i],cm.RAO_phase_22[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_22)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO HEAVE:\n"
                arraymatrix = np.stack((cm.RAO_33[:,0,i],cm.RAO_33[:,1,i],cm.RAO_33[:,2,i],cm.RAO_33[:,3,i],cm.RAO_33[:,4,i],cm.RAO_33[:,5,i],cm.RAO_33[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_33)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE HEAVE:\n"
                arraymatrix = np.stack((cm.RAO_phase_33[:,0,i],cm.RAO_phase_33[:,1,i],cm.RAO_phase_33[:,2,i],cm.RAO_phase_33[:,3,i],cm.RAO_phase_33[:,4,i],cm.RAO_phase_33[:,5,i],cm.RAO_phase_33[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_33)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO ROLL:\n"
                arraymatrix = np.stack((cm.RAO_44[:,0,i],cm.RAO_44[:,1,i],cm.RAO_44[:,2,i],cm.RAO_44[:,3,i],cm.RAO_44[:,4,i],cm.RAO_44[:,5,i],cm.RAO_44[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_44)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE ROLL:\n"
                arraymatrix = np.stack((cm.RAO_phase_44[:,0,i],cm.RAO_phase_44[:,1,i],cm.RAO_phase_44[:,2,i],cm.RAO_phase_44[:,3,i],cm.RAO_phase_44[:,4,i],cm.RAO_phase_44[:,5,i],cm.RAO_phase_44[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_44)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "RAO PITCH:\n"
                arraymatrix = np.stack((cm.RAO_55[:,0,i],cm.RAO_55[:,1,i],cm.RAO_55[:,2,i],cm.RAO_55[:,3,i],cm.RAO_55[:,4,i],cm.RAO_55[:,5,i],cm.RAO_55[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_55)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "PHASE PITCH:\n"
                arraymatrix = np.stack((cm.RAO_phase_55[:,0,i],cm.RAO_phase_55[:,1,i],cm.RAO_phase_55[:,2,i],cm.RAO_phase_55[:,3,i],cm.RAO_phase_55[:,4,i],cm.RAO_phase_55[:,5,i],cm.RAO_phase_55[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_55)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "RAO YAW:\n"
                arraymatrix = np.stack((cm.RAO_66[:,0,i],cm.RAO_66[:,1,i],cm.RAO_66[:,2,i],cm.RAO_66[:,3,i],cm.RAO_66[:,4,i],cm.RAO_66[:,5,i],cm.RAO_66[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_66)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE YAW:\n"
                arraymatrix = np.stack((cm.RAO_phase_66[:,0,i],cm.RAO_phase_66[:,1,i],cm.RAO_phase_66[:,2,i],cm.RAO_phase_66[:,3,i],cm.RAO_phase_66[:,4,i],cm.RAO_phase_66[:,5,i],cm.RAO_phase_66[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_66)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                index1 +1

                name = "RAOS_ship_"
                name_i = name + str(i) + '.csv'
                join_csv(raos,name_i)
            
            if i in cm.sea_var.list_fn:
                freq[:,0] = cm.fq2[:,index2]
                
                message = "RAO SURGE:\n"
                arraymatrix = np.stack((cm.RAO_11[:,0,i],cm.RAO_11[:,1,i],cm.RAO_11[:,2,i],cm.RAO_11[:,3,i],cm.RAO_11[:,4,i],cm.RAO_11[:,5,i],cm.RAO_11[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_11)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "PHASE SURGE:\n"
                arraymatrix = np.stack((cm.RAO_phase_11[:,0,i],cm.RAO_phase_11[:,1,i],cm.RAO_phase_11[:,2,i],cm.RAO_phase_11[:,3,i],cm.RAO_phase_11[:,4,i],cm.RAO_phase_11[:,5,i],cm.RAO_phase_11[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_11)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO SWAY:\n"
                arraymatrix = np.stack((cm.RAO_22[:,0,i],cm.RAO_22[:,1,i],cm.RAO_22[:,2,i],cm.RAO_22[:,3,i],cm.RAO_22[:,4,i],cm.RAO_22[:,5,i],cm.RAO_22[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_22)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE SWAY:\n"
                arraymatrix = np.stack((cm.RAO_phase_22[:,0,i],cm.RAO_phase_22[:,1,i],cm.RAO_phase_22[:,2,i],cm.RAO_phase_22[:,3,i],cm.RAO_phase_22[:,4,i],cm.RAO_phase_22[:,5,i],cm.RAO_phase_22[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_22)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO HEAVE:\n"
                arraymatrix = np.stack((cm.RAO_33[:,0,i],cm.RAO_33[:,1,i],cm.RAO_33[:,2,i],cm.RAO_33[:,3,i],cm.RAO_33[:,4,i],cm.RAO_33[:,5,i],cm.RAO_33[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_33)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE HEAVE:\n"
                arraymatrix = np.stack((cm.RAO_phase_33[:,0,i],cm.RAO_phase_33[:,1,i],cm.RAO_phase_33[:,2,i],cm.RAO_phase_33[:,3,i],cm.RAO_phase_33[:,4,i],cm.RAO_phase_33[:,5,i],cm.RAO_phase_33[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_33)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "RAO ROLL:\n"
                arraymatrix = np.stack((cm.RAO_44[:,0,i],cm.RAO_44[:,1,i],cm.RAO_44[:,2,i],cm.RAO_44[:,3,i],cm.RAO_44[:,4,i],cm.RAO_44[:,5,i],cm.RAO_44[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_44)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE ROLL:\n"
                arraymatrix = np.stack((cm.RAO_phase_44[:,0,i],cm.RAO_phase_44[:,1,i],cm.RAO_phase_44[:,2,i],cm.RAO_phase_44[:,3,i],cm.RAO_phase_44[:,4,i],cm.RAO_phase_44[:,5,i],cm.RAO_phase_44[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_44)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "RAO PITCH:\n"
                arraymatrix = np.stack((cm.RAO_55[:,0,i],cm.RAO_55[:,1,i],cm.RAO_55[:,2,i],cm.RAO_55[:,3,i],cm.RAO_55[:,4,i],cm.RAO_55[:,5,i],cm.RAO_55[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_55)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "PHASE PITCH:\n"
                arraymatrix = np.stack((cm.RAO_phase_55[:,0,i],cm.RAO_phase_55[:,1,i],cm.RAO_phase_55[:,2,i],cm.RAO_phase_55[:,3,i],cm.RAO_phase_55[:,4,i],cm.RAO_phase_55[:,5,i],cm.RAO_phase_55[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_55)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)

                message = "RAO YAW:\n"
                arraymatrix = np.stack((cm.RAO_66[:,0,i],cm.RAO_66[:,1,i],cm.RAO_66[:,2,i],cm.RAO_66[:,3,i],cm.RAO_66[:,4,i],cm.RAO_66[:,5,i],cm.RAO_66[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(name_66)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                message = "PHASE YAW:\n"
                arraymatrix = np.stack((cm.RAO_phase_66[:,0,i],cm.RAO_phase_66[:,1,i],cm.RAO_phase_66[:,2,i],cm.RAO_phase_66[:,3,i],cm.RAO_phase_66[:,4,i],cm.RAO_phase_66[:,5,i],cm.RAO_phase_66[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(raos),str(phase_66)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180", comments=message)
                
                index2 +1

                name = "RAOS_ship_"
                name_i = name + str(i) + '.csv'
                join_csv(raos,name_i)

def s_spectral_wave(wkd,nships):
    # create folder to store all RAO curves
    spwave = os.path.join(wkd,"Wave Spectrum")
    os.mkdir(spwave)

    index1 = 0
    index2 = 0
    freq = np.zeros((30,1))
    
    if os.path.exists(spwave):

        for i in range (nships):

            name = "Wave_spectrum_"
            name_i = name + str(i) + '.csv'

            if i in cm.sea_var.list_fn_0:
                freq[:,0] = cm.fq1[:,index1]
                arraymatrix = np.stack((cm.Wave_sp[:,0,i],cm.Wave_sp[:,1,i],cm.Wave_sp[:,2,i],cm.Wave_sp[:,3,i],cm.Wave_sp[:,4,i],cm.Wave_sp[:,5,i],cm.Wave_sp[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(spwave),str(name_i)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180")

                index1 +=1
            
            if i in cm.sea_var.list_fn:
                freq[:,0] = cm.fq2[:,index2]
                arraymatrix = np.stack((cm.Wave_sp[:,0,i],cm.Wave_sp[:,1,i],cm.Wave_sp[:,2,i],cm.Wave_sp[:,3,i],cm.Wave_sp[:,4,i],cm.Wave_sp[:,5,i],cm.Wave_sp[:,6,i]), axis=-1)
                a = np.append(freq,arraymatrix,axis=1)
                np.savetxt(os.path.join(str(spwave),str(name_i)), a, delimiter=';', header="Freq[rad/s];0;30;60;90;120;150;180")
                
                index2 +=1
                
def s_spectral_moments(wkd, nships):
    # create folder to store all RAO curves
    spmoments = os.path.join(wkd,"Spectral moments")
    os.mkdir(spmoments)

    name_sp11_0= 'mov_spectrums_0_surge.csv'; name_sp11_2= 'mov_spectrums_2_surge.csv'; name_sp11_4= 'mov_spectrums_4_surge.csv'
    name_sp22_0= 'mov_spectrums_0_sway.csv' ; name_sp22_2= 'mov_spectrums_2_sway.csv'; name_sp22_4= 'mov_spectrums_4_sway.csv'
    name_sp33_0= 'mov_spectrums_0_heave.csv'; name_sp33_2= 'mov_spectrums_2_heave.csv'; name_sp33_4= 'mov_spectrums_4_heave.csv'
    name_sp44_0= 'mov_spectrums_0_roll.csv'; name_sp44_2= 'mov_spectrums_2_roll.csv'; name_sp44_4= 'mov_spectrums_4_roll.csv'
    name_sp55_0= 'mov_spectrums_0_pitch.csv'; name_sp55_2= 'mov_spectrums_2_pitch.csv'; name_sp55_4= 'mov_spectrums_4_pitch.csv'
    name_sp66_0= 'mov_spectrums_0_yaw.csv'; name_sp66_2= 'mov_spectrums_2_yaw.csv'; name_sp66_4= 'mov_spectrums_4_yaw.csv'

    if os.path.exists(spmoments):
        for i in range (nships):

            name = "Spectral_moments_ship_"
            name_i = name + str(i)
            spship = os.path.join(spmoments,name_i)
            os.mkdir(spship)
            arraymatrix = np.zeros((1,7))
            
            message = "Spectral moments Order 0 Surge:\n"
            lst = [cm.m0[0,0,i],cm.m0[0,1,i],cm.m0[0,2,i],cm.m0[0,3,i],cm.m0[0,4,i],cm.m0[0,5,i],cm.m0[0,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp11_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Surge:\n"
            lst = [cm.m2[0,0,i],cm.m2[0,1,i],cm.m2[0,2,i],cm.m2[0,3,i],cm.m2[0,4,i],cm.m2[0,5,i],cm.m2[0,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp11_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Surge:\n"
            lst = [cm.m4[0,0,i],cm.m4[0,1,i],cm.m4[0,2,i],cm.m4[0,3,i],cm.m4[0,4,i],cm.m4[0,5,i],cm.m4[0,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp11_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 0 Sway:\n"
            lst = [cm.m0[1,0,i],cm.m0[1,1,i],cm.m0[1,2,i],cm.m0[1,3,i],cm.m0[1,4,i],cm.m0[1,5,i],cm.m0[1,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp22_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Sway:\n"
            lst = [cm.m2[1,0,i],cm.m2[1,1,i],cm.m2[1,2,i],cm.m2[1,3,i],cm.m2[1,4,i],cm.m2[1,5,i],cm.m2[1,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp22_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Sway:\n"
            lst = [cm.m4[1,0,i],cm.m4[1,1,i],cm.m4[1,2,i],cm.m4[1,3,i],cm.m4[1,4,i],cm.m4[1,5,i],cm.m4[1,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp22_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 0 Heave:\n"
            lst = [cm.m0[2,0,i],cm.m0[2,1,i],cm.m0[2,2,i],cm.m0[2,3,i],cm.m0[2,4,i],cm.m0[2,5,i],cm.m0[2,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp33_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Heave:\n"
            lst = [cm.m2[2,0,i],cm.m2[2,1,i],cm.m2[2,2,i],cm.m2[2,3,i],cm.m2[2,4,i],cm.m2[2,5,i],cm.m2[2,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp33_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Heave:\n"
            lst = [cm.m4[2,0,i],cm.m4[2,1,i],cm.m4[2,2,i],cm.m4[2,3,i],cm.m4[2,4,i],cm.m4[2,5,i],cm.m4[2,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp33_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 0 Roll:\n"
            lst = [cm.m0[3,0,i],cm.m0[3,1,i],cm.m0[3,2,i],cm.m0[3,3,i],cm.m0[3,4,i],cm.m0[3,5,i],cm.m0[3,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp44_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Roll:\n"
            lst = [cm.m2[3,0,i],cm.m2[3,1,i],cm.m2[3,2,i],cm.m2[3,3,i],cm.m2[3,4,i],cm.m2[3,5,i],cm.m2[3,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp44_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Roll:\n"
            lst = [cm.m4[3,0,i],cm.m4[3,1,i],cm.m4[3,2,i],cm.m4[3,3,i],cm.m4[3,4,i],cm.m4[3,5,i],cm.m4[3,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp44_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 0 Pitch:\n"
            lst = [cm.m0[4,0,i],cm.m0[4,1,i],cm.m0[4,2,i],cm.m0[4,3,i],cm.m0[4,4,i],cm.m0[4,5,i],cm.m0[4,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp55_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Pitch:\n"
            lst = [cm.m2[4,0,i],cm.m2[4,1,i],cm.m2[4,2,i],cm.m2[4,3,i],cm.m2[4,4,i],cm.m2[4,5,i],cm.m2[4,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp55_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Pitch:\n"
            lst = [cm.m4[4,0,i],cm.m4[4,1,i],cm.m4[4,2,i],cm.m4[4,3,i],cm.m4[4,4,i],cm.m4[4,5,i],cm.m4[4,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp55_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 0 Yaw:\n"
            lst = [cm.m0[5,0,i],cm.m0[5,1,i],cm.m0[5,2,i],cm.m0[5,3,i],cm.m0[5,4,i],cm.m0[5,5,i],cm.m0[5,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp66_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 2 Yaw:\n"
            lst = [cm.m2[5,0,i],cm.m2[5,1,i],cm.m2[5,2,i],cm.m2[5,3,i],cm.m2[5,4,i],cm.m2[5,5,i],cm.m2[5,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp66_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            message = "Spectral moments Order 4 Yaw:\n"
            lst = [cm.m4[5,0,i],cm.m4[5,1,i],cm.m4[5,2,i],cm.m4[5,3,i],cm.m4[5,4,i],cm.m4[5,5,i],cm.m4[5,6,i]]
            arraymatrix[0,:] = lst
            np.savetxt(os.path.join(str(spship),str(name_sp66_4)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

            name = "Spectral_moments_ship_"
            name_i = name + str(i) + '.csv'
            join_csv(spship,name_i)

def s_seakeeping(wkd,nships):
    # create folder to store all RAO curves
    seakeeping_mag = os.path.join(wkd,"Seakeeping magnitudes")
    os.mkdir(seakeeping_mag)

    if os.path.exists(seakeeping_mag):
        arraymatrix = np.zeros((1,7))
        for i in range (nships):
            
            name = "Significant_Max_RMSACC_ship_"
            name_i = name + str(i)
            sigship = os.path.join(seakeeping_mag,name_i)
            os.mkdir(sigship)

            if os.path.exists(sigship):
                seak_11_0 = 'mov_sig_surge.csv'; seak_11_1 = 'mov_max_surge.csv'; seak_11_2 = 'accrms_surge.csv'
                seak_22_0 = 'mov_sig_sway.csv'; seak_22_1 = 'mov_max_sway.csv'; seak_22_2 = 'accrms_sway.csv'
                seak_33_0 = 'mov_sig_heave.csv'; seak_33_1 = 'mov_max_heave.csv'; seak_33_2 = 'accrms_heave.csv'
                seak_44_0 = 'mov_sig_roll.csv'; seak_44_1 = 'mov_max_roll.csv'; seak_44_2 = 'accrms_roll.csv'
                seak_55_0 = 'mov_sig_pitch.csv'; seak_55_1 = 'mov_max_pitch.csv'; seak_55_2 = 'accrms_pitch.csv'
                seak_66_0 = 'mov_sig_yaw.csv'; seak_66_1 = 'mov_max_yaw.csv'; seak_66_2 = 'accrms_yaw.csv'
                seak_77 = 'sm.csv'
                seak_88 = 'msi.csv'

                message = "Significant movement Surge:\n"
                lst = [cm.mov_11_sig[0,i], cm.mov_11_sig[1,i],cm.mov_11_sig[2,i], cm.mov_11_sig[3,i], cm.mov_11_sig[4,i], cm.mov_11_sig[5,i], cm.mov_11_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_11_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Surge:\n"
                lst = [cm.mov_11_max[0,i], cm.mov_11_max[1,i],cm.mov_11_max[2,i], cm.mov_11_max[3,i], cm.mov_11_max[4,i], cm.mov_11_max[5,i], cm.mov_11_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_11_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Surge:\n"
                lst = [cm.acc_RMS_11[0,i], cm.acc_RMS_11[1,i],cm.acc_RMS_11[2,i], cm.acc_RMS_11[3,i], cm.acc_RMS_11[4,i], cm.acc_RMS_11[5,i], cm.acc_RMS_11[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_11_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Significant movement Sway:\n"
                lst = [cm.mov_22_sig[0,i], cm.mov_22_sig[1,i],cm.mov_22_sig[2,i], cm.mov_22_sig[3,i], cm.mov_22_sig[4,i], cm.mov_22_sig[5,i], cm.mov_22_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_22_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Sway:\n"
                lst = [cm.mov_22_max[0,i], cm.mov_22_max[1,i],cm.mov_22_max[2,i], cm.mov_22_max[3,i], cm.mov_22_max[4,i], cm.mov_22_max[5,i], cm.mov_22_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_22_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Sway:\n"
                lst = [cm.mov_22_max[0,i], cm.mov_22_max[1,i],cm.mov_22_max[2,i], cm.mov_22_max[3,i], cm.mov_22_max[4,i], cm.mov_22_max[5,i], cm.mov_22_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_22_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Significant movement Heave:\n"
                lst = [cm.mov_33_sig[0,i], cm.mov_33_sig[1,i],cm.mov_33_sig[2,i], cm.mov_33_sig[3,i], cm.mov_33_sig[4,i], cm.mov_33_sig[5,i], cm.mov_33_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_33_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Heave:\n"
                lst = [cm.mov_33_max[0,i], cm.mov_33_max[1,i],cm.mov_33_max[2,i], cm.mov_33_max[3,i], cm.mov_33_max[4,i], cm.mov_33_max[5,i], cm.mov_33_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_33_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Heave:\n"
                lst = [cm.acc_RMS_33[0,i], cm.acc_RMS_33[1,i],cm.acc_RMS_33[2,i], cm.acc_RMS_33[3,i], cm.acc_RMS_33[4,i], cm.acc_RMS_33[5,i], cm.acc_RMS_33[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_33_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Significant movement Roll:\n"
                lst = [cm.mov_44_sig[0,i], cm.mov_44_sig[1,i],cm.mov_44_sig[2,i], cm.mov_44_sig[3,i], cm.mov_44_sig[4,i], cm.mov_44_sig[5,i], cm.mov_44_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_44_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Roll:\n"
                lst = [cm.mov_44_max[0,i], cm.mov_44_max[1,i],cm.mov_44_max[2,i], cm.mov_44_max[3,i], cm.mov_44_max[4,i], cm.mov_44_max[5,i], cm.mov_44_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_44_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Roll:\n"
                lst = [cm.acc_RMS_44[0,i], cm.acc_RMS_44[1,i],cm.acc_RMS_44[2,i], cm.acc_RMS_44[3,i], cm.acc_RMS_44[4,i], cm.acc_RMS_44[5,i], cm.acc_RMS_44[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_44_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Significant movement Pitch:\n"
                lst = [cm.mov_55_sig[0,i], cm.mov_55_sig[1,i],cm.mov_55_sig[2,i], cm.mov_55_sig[3,i], cm.mov_55_sig[4,i], cm.mov_55_sig[5,i], cm.mov_55_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_55_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Pitch:\n"
                lst = [cm.mov_55_max[0,i], cm.mov_55_max[1,i],cm.mov_55_max[2,i], cm.mov_55_max[3,i], cm.mov_55_max[4,i], cm.mov_55_max[5,i], cm.mov_55_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_55_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Pitch:\n"
                lst = [cm.acc_RMS_55[0,i], cm.acc_RMS_55[1,i],cm.acc_RMS_55[2,i], cm.acc_RMS_55[3,i], cm.acc_RMS_55[4,i], cm.acc_RMS_55[5,i], cm.acc_RMS_55[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_55_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Significant movement Yaw:\n"
                lst = [cm.mov_66_sig[0,i], cm.mov_66_sig[1,i],cm.mov_66_sig[2,i], cm.mov_66_sig[3,i], cm.mov_66_sig[4,i], cm.mov_66_sig[5,i], cm.mov_66_sig[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_66_0)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)
                
                message = "Max movement Yaw:\n"
                lst = [cm.mov_66_max[0,i], cm.mov_66_max[1,i],cm.mov_66_max[2,i], cm.mov_66_max[3,i], cm.mov_66_max[4,i], cm.mov_66_max[5,i], cm.mov_66_max[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_66_1)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "RMS acceleration Yaw:\n"
                lst = [cm.acc_RMS_66[0,i], cm.acc_RMS_66[1,i],cm.acc_RMS_66[2,i], cm.acc_RMS_66[3,i], cm.acc_RMS_66[4,i], cm.acc_RMS_66[5,i], cm.acc_RMS_66[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_66_2)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Significative Magnitude (SM):\n"
                lst = [cm.Significative_mag[0,i], cm.Significative_mag[1,i],cm.Significative_mag[2,i], cm.Significative_mag[3,i], cm.Significative_mag[4,i], cm.Significative_mag[5,i], cm.Significative_mag[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_77)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                message = "Motion Sickness Incidence (MSI):\n"
                lst = [cm.Motion_sickness[0,i], cm.Motion_sickness[1,i],cm.Motion_sickness[2,i], cm.Motion_sickness[3,i], cm.Motion_sickness[4,i], cm.Motion_sickness[5,i], cm.Motion_sickness[6,i]]
                arraymatrix[0,:] =lst
                np.savetxt(os.path.join(str(sigship),str(seak_88)), arraymatrix, delimiter=';', header="0;30;60;90;120;150;180", comments=message)

                name = "Movements_ship_"
                name_i = name + str(i) + '.csv'
                join_csv(sigship,name_i)
            
# create zip files with results
def zipfiles(resfolder, workdirectory):
    zipfolder = shutil.make_archive("results","zip",resfolder)
    shutil.move(zipfolder, workdirectory)
    shutil.rmtree(resfolder)
