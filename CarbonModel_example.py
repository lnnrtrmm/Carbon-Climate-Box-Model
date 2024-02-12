import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from Climate_Box_Model import EnergyAndCarbonBoxModel


################################################################
############### DATA LOADING ###################################
################################################################


scenarios = ['ssp119', 'ssp245', 'ssp534']

##############################################################
##
## CO2 emissions from RCMIP 
## -> these are total CO2 emissions as used in 
##    MPI-ESM simulation (1850-2300) (ssp585 only until 2100)
##
##############################################################
all_C_emissions = []
for sce in scenarios:
    infile = 'CO2_emissions/global_C_emission_'+sce+'.nc'
    ds=Dataset(infile)
    C_emission = ds.variables['carbon_emission_global'][1:,0,0]
    ds.close()
    
    all_C_emissions.append(C_emission)

    
    
##############################################################
##
## ERF data from Chris Smith 
## (https://github.com/Priestley-Centre/ssp_erf/tree/v3.0)
## Data is available starting from 1750
##
##############################################################
all_ERF_total = []
all_ERF_co2 = []
all_ERF_nonco2 = []
for sce in scenarios:
    if sce=='ssp534': tmp='-over'
    else: tmp=''
    infile = 'ERF_SSPs/ERF_'+sce+tmp+'_1750-2500.csv'
    df = pd.read_csv(infile)

    ERF_co2 = df.values[100:-200,1]
    ERF_total = df.values[100:-200,-1]
    ERF_nonco2 = ERF_total - ERF_co2

    all_ERF_total.append(ERF_total)
    all_ERF_co2.append(ERF_co2)
    all_ERF_nonco2.append(ERF_nonco2)


########## MPI-ESM data ######################################
##
## Loading in data from the ten ensemble members of MPI-ESM
## 
##############################################################
n_sce = len(scenarios)
    
path='/home/lennart/Arbeit/2_WorldTrans/work/MPIESM/mon_dat/'

time_MPIESM =np.linspace(1850.5,2299.5,450)

T_MPIESM_all = np.zeros((450,10,n_sce))
CO2_MPIESM_all = np.zeros((450,10,n_sce))
CO2_FLUX_MPIESM_all = np.zeros((450,10,n_sce))
C_LAND_MPIESM_all = np.zeros((450,10,n_sce))

for i in range(1,11):
    member='r'+str(i)+'i1p1f1'
    
    #print('MPIESM data: Loading ensemble member '+member)
    
    for j, sce in enumerate(scenarios):
        if sce == 'ssp119' or sce == 'ssp534': offset = -11.0
        else: offset = 0.0
        
        #print('  scenario: ', sce)  
        ds_T0 = Dataset(path+'historical/'+member+'/hist_'+member+'_tas_1850-2014_ym.nc')
        ds_T1 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_tas_2015-2099_ym.nc')
        ds_T2 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_tas_2100-2299_ym.nc')
                          
        ds_C0 = Dataset(path+'historical/'+member+'/hist_'+member+'_co2_1850-2014_ym.nc')
        ds_C1 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_co2_2015-2099_ym.nc')
        ds_C2 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_co2_2100-2299_ym.nc')
        
        ds_flx0 = Dataset(path+'historical/'+member+'/hist_'+member+'_hamocc_mon_1850-2014_ym.nc')
        ds_flx1 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_hamocc_mon_2015-2099_ym.nc')
        ds_flx2 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_hamocc_mon_2100-2299_ym.nc')
        
        ds_L0 = Dataset(path+'historical/'+member+'/hist_'+member+'_jsbach_mon_1850-2014_ym.nc')
        ds_L1 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_jsbach_mon_2015-2099_ym.nc')
        ds_L2 = Dataset(path+sce+'/'+member+'/'+sce+'_'+member+'_jsbach_mon_2100-2299_ym.nc')
    
        tmp_T = ds_T0.variables['temp2'][:,0,0]
        tmp_T = np.append(tmp_T, ds_T1.variables['temp2'][:,0,0])
        tmp_T = np.append(tmp_T, ds_T2.variables['temp2'][:,0,0])
    
        tmp_C = ds_C0.variables['CO2'][:,0,0,0]
        tmp_C = np.append(tmp_C, ds_C1.variables['CO2'][:,0,0,0]+offset)
        tmp_C = np.append(tmp_C, ds_C2.variables['CO2'][:,0,0,0]+offset)
        
        tmp_F = ds_flx0.variables['global_net_co2_flux'][:,0,0,0]
        tmp_F = np.append(tmp_F, ds_flx1.variables['global_net_co2_flux'][:,0,0,0])
        tmp_F = np.append(tmp_F, ds_flx2.variables['global_net_co2_flux'][:,0,0,0])
        
        tmp_L = ds_L0.variables['cLand'][:,0,0]
        tmp_L = np.append(tmp_L, ds_L1.variables['cLand'][:,0,0])
        tmp_L = np.append(tmp_L, ds_L2.variables['cLand'][:,0,0])
    
        ds_T0.close()
        ds_T1.close()
        ds_T2.close()
        
        ds_C0.close()
        ds_C1.close()
        ds_C2.close()
        
        ds_flx0.close()
        ds_flx1.close()
        ds_flx2.close()
        
        ds_L0.close()
        ds_L1.close()
        ds_L2.close()
    
        T_MPIESM_all[:,i-1,j] = tmp_T - 273.15
        CO2_MPIESM_all[:,i-1,j] = tmp_C
        CO2_FLUX_MPIESM_all[:,i-1,j] = tmp_F
        C_LAND_MPIESM_all[:,i-1,j] = tmp_L
                          
T_MPIESM_mean = np.mean(T_MPIESM_all, axis=1)
T_MPIESM_mean = T_MPIESM_mean - np.mean(T_MPIESM_mean[np.newaxis,:50], axis=1)
CO2_MPIESM_mean = np.mean(CO2_MPIESM_all, axis=1)
CO2_FLUX_MPIESM_mean = np.mean(CO2_FLUX_MPIESM_all, axis=1)
cum_CO2_FLUX_MPIESM_mean = np.cumsum(CO2_FLUX_MPIESM_mean, axis=0)
C_LAND_MPIESM_mean = np.mean(C_LAND_MPIESM_all, axis=1)
FLUX_LAND_MPIESM_mean = C_LAND_MPIESM_mean[:,:] - np.append(C_LAND_MPIESM_mean[np.newaxis,0,:], C_LAND_MPIESM_mean, axis=0)[:-1,:]



#  Settings of the setup
sce = 'ssp245'

if sce=='ssp119':
    sidx = 0
elif sce=='ssp245':
    sidx = 1
elif sce=='ssp534':
    sidx=2


sy=1850
ey=2300
dt = 0.1
spyr = int(1.0/dt)
nsteps=int((ey-sy)/dt)+1

time = np.linspace(1850,2300,451)
time_long = np.linspace(1850,2300,nsteps)

land_flux_mpiesm = np.append(FLUX_LAND_MPIESM_mean[:,sidx],0)

#### Cmodel needs interpolated input because of smaller time step
C_emissions_long = np.zeros(nsteps)
ERF_nonco2_long = np.zeros(nsteps)
land_flux_mpiesm_long = np.zeros(nsteps)
for i in range(len(all_C_emissions[sidx])):
    C_emissions_long[i*10:i*10+spyr] = all_C_emissions[sidx][i]
    ERF_nonco2_long[i*10:i*10+spyr]  = all_ERF_nonco2[sidx][i]
    land_flux_mpiesm_long[i*10:i*10+spyr] = land_flux_mpiesm[i]


#################################################################
########## Create the Carbon Model ##############################
#################################################################


MyModel = EnergyAndCarbonBoxModel(C_emissions_long, ERF_nonco2_long, dbg=0, dt=0.1, sy=1850, ey=2300)

MyModel.CarbonModel.prescribe_C_flux_land = True
MyModel.CarbonModel.prescribed_C_flux_land = land_flux_mpiesm_long

# Values as in LOSCAR
MyModel.CarbonModel.update_depths([250, 250, 900])
MyModel.CarbonModel.include_nutrient_Tdep = True

MyModel.spinup()
MyModel.integrate()


#################################################################
####################### Plotting ################################
#################################################################

Cmodel_T = np.copy(MyModel.getTsurf())
Cmodel_T = Cmodel_T - np.mean(Cmodel_T[:50])


plt.figure(figsize=(12,8))
font=13

plt.rcParams.update({'font.size': font})
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)

plt.suptitle(sce)
  
axes = []
for i in range(4):
    plt.subplot(221+i)
    ax=plt.gca()
    axes.append(ax)

ylabels = ['GSAT anomaly ($^{\circ}$C)', 'atm. CO$_2$ concentration (ppm)',
           'ocean CO$_2$-flux (GtC yr$^{-1}$)', 'cumulative ocean CO$_2$-flux (GtC)']


MPIESM_dat = [T_MPIESM_mean[:,sidx],
              CO2_MPIESM_mean[:,sidx],
              -CO2_FLUX_MPIESM_mean[:,sidx],
              -cum_CO2_FLUX_MPIESM_mean[:,sidx]]


Cmodel_dat =   [Cmodel_T,
                MyModel.getCO2(),
                MyModel.getAirSeaExchange(),
                MyModel.getOceanCarbon() - MyModel.getOceanCarbon()[0]]

for i, ax in enumerate(axes):
    ax.set_xlabel('')
    ax.set_xlim([1850,2300])
    ax.set_ylabel(ylabels[i])
    
    ax.plot(time_MPIESM[:-1], MPIESM_dat[i][:-1], label='MPI-ESM emission-driven', linewidth=3, color='tab:green')
    ax.plot(MyModel.getTime()[:-1], Cmodel_dat[i][:-1],  label='Carbon box model', linewidth=3, color='tab:red')
    
    if i==2 or i==4: ax.axhline(y=0, color='k')
    
    if i==3: ax.legend()
    ax.grid()


plt.tight_layout()
plt.savefig('CModel_example.png')
