# © Lennart Ramme, Max-Planck-Institute for Meteorology, 2025

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log10")

import oceanProcesses as OP
from constants import *



class ForcedOceanCarbonModel:
    """

    """

    # Tuning parameters have default values from Lenton (2000)
    def __init__(self, co2_emissions, nbp, sta, ModelType='4BoxOcean', sy=1850, ey=2300, dt=0.1, dbg=0,
                 pi_co2=283.0, initial_C_reservoirs = [730.0, 140.0, 10000.0, 26830.0],
                 depths=[100, 250, 900], advection='LOSCAR', airSeaExchange='LOSCAR'):


        if ModelType not in ['3BoxOcean', '4BoxOcean']: sys.exit('ModelType not allowed!')
        else: self.ModelType = ModelType

        if advection not in ['3BoxOcean', 'LOSCAR', 'Lenton']: sys.exit('Advection scheme not allowed!')
        else: self.advection = advection

        if airSeaExchange not in ['LOSCAR', 'HAMOCC']: sys.exit('AirSeaExchange scheme not allowed!')
        else: self.airSeaExchange = airSeaExchange

        if self.ModelType == '4BoxOcean' and self.advection == '3BoxOcean': 
            print('Warning: 4BoxOcean ModelType not compatible with 3BoxOcean advection scheme, setting to LOSCAR advection')
            self.advection = 'LOSCAR'
        elif self.ModelType == '3BoxOcean' and self.advection in ['Lenton', 'LOSCAR']:
            print('Warning: 3BoxOcean ModelType not compatible with Lenton or LOSCAR advection scheme, setting to 3BoxOcean advection')
            self.advection = '3BoxOcean'

        if dbg == 3:
            print(self.advection, self.airSeaExchange, self.ModelType)


        self.dbg = dbg
        # Timestep:
        self.startyear = sy
        self.endyear = ey
        self.dt = dt
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1

        if self.dt == 1.0:
            self.emission        = co2_emissions # emissions should enter in GT C per year
            self.land_flux_total = nbp
            self.STA             = sta
            self.nStepsPerYear   = 1
        elif self.dt < 1.0:
            self.nStepsPerYear = int(1 / dt )
            self.emission = np.zeros(self.nyears)
            self.land_flux_total = np.zeros(self.nyears)
            self.STA = np.zeros(self.nyears)

            for i in range(len(co2_emissions)):
                n = self.nStepsPerYear
                self.emission[i*n:i*n+n] = co2_emissions[i]
                self.land_flux_total[i*n:i*n+n] = nbp[i]
                self.STA[i*n:i*n+n] = sta[i]
        else:
            sys.exit('Time step dt should not be larger than 1 year')


        self.pi_co2 = pi_co2
        self.spinupLength = 5000
         
        
        ####################################################################################
        ################          Tuning parameters          ###############################
        ####################################################################################
            
        
        ## Mixing and overturning parameters
        # default values are guesses

        if self.advection == 'Lenton':
            # default values are from Lenton (2000)
            self.k_mix_WI   = 125.0e6
            self.k_mix_ID   = 200.0e6
            self.k_THC_over =  65.0e6
            self.k_HL_over  =  48.7e6
        elif self.advection == 'LOSCAR':
            self.k_mix_WI  = 63.0e6
            self.k_mix_CD  = 17.0e6
            self.k_HL_over = 20.0e6
        elif self.advection == '3Box':
            self.k_mix_WD = 100.0e6
            self.k_mix_CD = 30.0e6
            self.k_over = 20.0e6


        # Overturning circulation dependent on Warming?
        self.advection_Tdep_frac = 0   # Set to 0 in default setup.
        
        ## gas exchange rate
        self.k_gasex_warm = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1
        self.k_gasex_cold = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1


        self.withTdepSurfTemp = True
        self.withTdepSurfSal = True
        self.withTdepSurfAlk = True
        self.withTdepBiopump = True


        ## FIXME: continue here with
        # - add T+S dependent gas exchange??
        # - update the whole repository to have just one or two files that contain classes
        #   and these classes should make use of the lib as much as possible to substantially
        #   reduce code and avoid duplications



        ####################################################################################
        ####################################################################################
        ####################################################################################

        ### Initialisation of variables
        ## These neeed to be adjusted to ensure convergence of pH solver in first iteration
        self.pH_oce_cold_init = 8.36
        self.pH_oce_warm_init = 8.14


        # reservoir containers
        self.C_atm = np.zeros((self.nyears)) + self.pi_co2 / epsilon

        if self.ModelType == '3BoxOcean': self.nOce = 3
        elif self.ModelType == '4BoxOcean': self.nOce = 4
        self.C_oce = np.zeros((self.nOce, self.nyears))

        # model settings
        self.warm_area_fraction = 0.85
        self.cold_area_fraction = 1.0 - self.warm_area_fraction

        # Setting the depths and volumes of the ocean
        self.update_depths(depths)

        self.C_total = np.copy(self.C_atm)
        self.C_totalOce = np.zeros_like(self.C_atm)
         
        # initial reservoir values
        self.reservoir_inits = initial_C_reservoirs
        
        # initialize the carbon reservoir containers
        for res_i, init_reservoir in enumerate(self.reservoir_inits):
            self.C_oce[res_i, 0] = init_reservoir 
            self.C_total[0] += init_reservoir
            self.C_totalOce[0] += init_reservoir
        
        # atmospheric and oceanic pCO2
        self.pCO2_atm      = np.zeros((self.nyears)) + self.pi_co2
        self.pCO2_oce_cold = np.zeros((self.nyears))
        self.pCO2_oce_warm = np.zeros((self.nyears))
        self.pH_oce_cold = np.zeros((self.nyears)) + self.pH_oce_cold_init
        self.pH_oce_warm = np.zeros((self.nyears)) + self.pH_oce_warm_init

        self.__calc_pCO2_atm(0)
        
        # rate containers
        self.ocean_flux_warm = np.zeros((self.nyears))
        self.ocean_flux_cold = np.zeros((self.nyears))
        
        self.ocean_flux_total = np.zeros((self.nyears))
        

    def update_depths(self,depths):
        #### Call before the integration if this shall be changed!
        if self.ModelType == '3BoxOcean' and len(depths)==2:
            self.depths = np.asarray(depths) 
            self.V_oce = np.zeros(self.nOce)
    
            self.V_oce[0] = self.depths[0] * ocean_surface_area * self.warm_area_fraction
            self.V_oce[1] = self.depths[1] * ocean_surface_area * self.cold_area_fraction
            self.V_oce[2] = ocean_volume - np.sum(self.V_oce[:2])

        elif self.ModelType == '4BoxOcean' and len(depths)==3:
            self.depths = np.asarray(depths) 
            self.V_oce = np.zeros(self.nOce)
    
            self.V_oce[0] = self.depths[0] * ocean_surface_area * self.warm_area_fraction
            self.V_oce[1] = self.depths[1] * ocean_surface_area * self.cold_area_fraction
            self.V_oce[2] = self.depths[2] * ocean_surface_area
            self.V_oce[3] = ocean_volume - np.sum(self.V_oce[:3])
        else:
            sys.exit('Number of given layer depths does not fit ModelType!')

        self.__update_surface_ocean_inputs()


    def modify_initial_reservoirs(self,modifications):
        # modify the carbon reservoir containers
        for res_i, modification in enumerate(modifications):
            self.C_oce[res_i,0] += modification
            self.C_total[0] += modification
            self.C_totalOce[0] += modification
        
    def spinup(self):

        if self.dbg==1: print('Spinning up the model')

        for i in range(int(self.spinupLength/self.dt)):  
        
            self.__calc_surface_ocean_flux(0, 0.0)

            self.__calc_total_carbon(0)
            if self.dbg == 1:
                print('   ... ocean carbon uptake:', self.ocean_flux_total[0])
                print('   ... total ocean carbon before biopump:', self.C_totalOce[0])
                
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            self.__biological_carbon_pump(0, 0.0)

            self.__calc_total_carbon(0)
            
            if self.dbg == 1:
                print('   ... total ocean carbon after biopump:', self.C_totalOce[0])
            
            ##### Advect the oceanic carbon
            self.__advect_ocean_tracer(0, 0.0, lspinup=True)

            self.__calc_total_carbon(0)


            if self.dbg == 1:
                print('   ... total ocean carbon after advection', self.C_totalOce[0])


        if self.dbg == 1: print('Spinup done')
        return

        
    def integrate(self):

        if not self.withTdepSurfTemp:
            self.T_cold_surface_Tslope = 0.0
            self.T_warm_surface_Tslope = 0.0

        if not self.withTdepSurfSal:
            self.S_cold_surface_Tslope = 0.0
            self.S_warm_surface_Tslope = 0.0

        if not self.withTdepSurfAlk:
            self.Alk_cold_surface_Tslope = 0.0
            self.Alk_warm_surface_Tslope = 0.0

        if not self.withTdepBiopump:
            self.bio_pump_cold_Tslope = 0.0

        self.spinup()
        
        for i in range(self.nyears-1):

            ############ Updating atmospheric CO2 based on land flux and emissions
            self.C_atm[i+1]  = self.C_atm[i]  + self.dt * (self.emission[i] - self.land_flux_total[i])

            self.__calc_pCO2_atm(i+1)
        
            ######################################################
            ############ Ocean component #########################
            ######################################################

            # Air-Sea CO2 exchange        
            self.__calc_surface_ocean_flux(i, self.STA[i])
      
            ########### ocean internal restructuring ###########
            #### Biological carbon pump
            self.__biological_carbon_pump(i, self.STA[i])

            ##### Advect the oceanic carbon
            self.__advect_ocean_tracer(i, self.STA[i])
        
            self.__calc_total_carbon(i+1)
        return
    

    def __calc_total_carbon(self,i):
        self.C_totalOce[i] = np.sum(self.C_oce[:,i])
        self.C_total[i] = self.C_atm[i] + np.sum(self.C_oce[:,i])
        return

    def __calc_pCO2_atm(self,i):
        self.pCO2_atm[i] = self.C_atm[i] * epsilon
        return
    

    def __update_surface_ocean_inputs(self):
        
        #### Surface ocean conditions (from MPI-ESM)

        ## Relating temperature, salinity, alkalinity and the carbon export in the surface layer boxes
        ## to the global mean T anomaly and the thickness of the surface layer

        ## Default values are from fitting MPI-ESM1.2-LR emission-driven runs
        ## X0 value is the mean value at STA=0.0 and Tslope is the slope in a linear fit. As the values
        ## of these fit parameters are themselves dependent on the thickness of the surface layer, 
        ## the fits have been conducted for all possible layer thicknesses in MPI-ESM between 50 and 500 m.
        ## Subsequently, the fit parameters were fitted themselves with a quadratic fit to be able to calculate
        ## the fit values here directly for a given surface layer thickness.

        # Surface ocean pot. temperature
        T_slope_cold_params = np.asarray([-2.34e-7, 1.39e-4, 0.41])
        T0_cold_params = np.asarray([-6.71e-7, 2.50e-3, 0.87])

        T_slope_warm_params = np.asarray([1.09e-6, -9.54e-4, 0.69])
        T0_warm_params = np.asarray([2.13e-5, -2.5e-2, 20.7])

        # Surface ocean pot. salinity
        S_slope_cold_params = np.asarray([-3.26e-7, 4.19e-4, -0.178])
        S0_cold_params = np.asarray([-5.49e-6, 5.05e-3, 33.071])

        S_slope_warm_params = np.asarray([-1.63e-7, 1.54e-4, -0.041])
        S0_warm_params = np.asarray([-2.2e-6, 1.58e-3, 34.658])

        # Surface ocean alkalinity
        Alk_slope_cold_params = np.asarray([-1.86e-8, 2.76e-5, -1.447e-2])
        Alk0_cold_params = np.asarray([-3.35e-7, 3.09e-4, 2.2098])

        Alk_slope_warm_params = np.asarray([-1.45e-8, 1.23e-5, -3.33e-3])
        Alk0_warm_params = np.asarray([-7.80e-8, 8.10e-5, 2.2786])

        # Biologial carbon export from surface layer
        # -> Only use T-dependency for high latitude bio pump, as low latitude is almost constant in MPIESM
        Biopump_slope_cold_params = np.asarray([-1.76e-6, 1.45e-3, -0.407])
        Biopump_cold_params = np.asarray([3.22e-5, -2.96e-2, 9.186])

        Biopump_warm_params = np.asarray([4.1e-6, -3.55e-3, 0.889])

        # Transfer efficiency is completely constant (and 0 in low latitudes)
        self.bio_pump_cold_eff = 0.2 

        
        #### Now make the initial value and slopes out of the parameters:
        self.T0_cold_surface = OP.quadraticFit(self.depths[1], T0_cold_params)
        self.T0_warm_surface = OP.quadraticFit(self.depths[0], T0_warm_params)
        self.T_cold_surface_Tslope = OP.quadraticFit(self.depths[1], T_slope_cold_params)
        self.T_warm_surface_Tslope = OP.quadraticFit(self.depths[0], T_slope_warm_params)
        
        self.S0_cold_surface = OP.quadraticFit(self.depths[1], S0_cold_params)
        self.S0_warm_surface = OP.quadraticFit(self.depths[0], S0_warm_params)
        self.S_cold_surface_Tslope = OP.quadraticFit(self.depths[1], S_slope_cold_params)
        self.S_warm_surface_Tslope = OP.quadraticFit(self.depths[0], S_slope_warm_params)

        self.Alk0_cold_surface = OP.quadraticFit(self.depths[1], Alk0_cold_params)
        self.Alk0_warm_surface = OP.quadraticFit(self.depths[0], Alk0_warm_params)
        self.Alk_cold_surface_Tslope = OP.quadraticFit(self.depths[1], Alk_slope_cold_params)
        self.Alk_warm_surface_Tslope = OP.quadraticFit(self.depths[0], Alk_slope_warm_params)

        self.bio_pump_cold = OP.quadraticFit(self.depths[1], Biopump_cold_params)
        self.bio_pump_warm = OP.quadraticFit(self.depths[0], Biopump_warm_params)
        self.bio_pump_cold_Tslope = OP.quadraticFit(self.depths[1], Biopump_slope_cold_params)
        return


    def __calc_surface_ocean_flux(self, i, Ts, lspinup=False):

        OP.calc_surface_ocean_flux(self, i, Ts, self.airSeaExchange)

        self.ocean_flux_total[i] = self.ocean_flux_warm[i] + self.ocean_flux_cold[i]

        ##### Updating reservoirs due to ocean-flux
        self.C_oce[0,i] = self.C_oce[0,i] + self.dt * self.ocean_flux_warm[i]
        self.C_oce[1,i] = self.C_oce[1,i] + self.dt * self.ocean_flux_cold[i]

        if not lspinup:
            self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
            self.__calc_pCO2_atm(i+1)
        return


    def __advect_ocean_tracer(self, i, Ts, lspinup=False):

        if self.ModelType == '3BoxOcean':
            OP.advect_ocean_tracer_3Box(self, i, Ts)
        elif self.ModelType == '4BoxOcean':
            if self.advection == 'Lenton':
                OP.advect_ocean_tracer_Lenton(self, i, Ts)
            elif self.advection == 'LOSCAR':
                OP.advect_ocean_tracer_LOSCAR(self, i, Ts)
            else:
                sys.exit('No correct choice of advection model! Can be "Lenton" or "LOSCAR", current value: '+self.advection)

        if lspinup: self.C_oce[:,0] = self.C_oce[:,i+1]
        return


    def __biological_carbon_pump(self, i, Ts):

        if self.ModelType == '3BoxOcean':
            OP.biological_carbon_pump_3Box(self, i, Ts)
        elif self.ModelType == '4BoxOcean':
            OP.biological_carbon_pump_4Box(self, i, Ts)
        return


    def getOceanCarbon(self):
        return np.sum(self.C_oce, axis=0)

    def getTime(self):
        return np.linspace(self.startyear, self.endyear, self.nyears)

    def getTimeYearly(self):
        return np.linspace(self.startyear, self.endyear, int((self.nyears-1)*self.dt)) + 0.5

    def getAtmCGrowth(self):
        return np.append(0, self.C_atm[1:]-self.C_atm[:-1]) / self.dt

    def getOceanFluxTotalYearly(self):
        return np.mean(self.ocean_flux_total[:-1].reshape(-1,self.nStepsPerYear), axis=1)
