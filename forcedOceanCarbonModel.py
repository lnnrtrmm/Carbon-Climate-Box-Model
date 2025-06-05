# © Lennart Ramme, Max-Planck-Institute for Meteorology, 2025

import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in log10")

import oceanProcesses as OP
from constants import *



class OceanCarbonModel_4plus1Box:
    """A class of a carbon cycle model originally based on that from Lenton (2000).
    The model consists of 4 ocean boxes.

    This is not the normal Lenton Model. I used the air-sea gas exchange and advection scheme
    from LOSCAR v2.0.4 (Zeebe, 2012, GMD). The model uses input from MPI-ESM-LR ensemble
    simulation for surface ocean values of temperature, salinity, alkalinity and the
    biological carbon pump.

    The class here is a reduced version with prescribed fluxes of land carbon and prescribed 
    global mean temperature. It only includes 5 C reservoirs (4 oce, 1 atm).

    """

    #### Model parameters and initial values
    #### --> Values from Lenton (2000)

    ## constants
    earth_surface_area = 5.101e14 # m^2
    ocean_surface_area = 3.6e14   # m^2
    ocean_volume = 1.36e18        # m^3
    mole_volume_atm = 1.773e20    # moles
    mol_to_GT_C = 12.0107e-15     # GT C (mol C)^-1
    epsilon = 0.47                # ppm/Gt C
    secondsPerYear = 365.25 * 24 * 3600 # s/yr

    # Tuning parameters have default values from Lenton (2000)
    def __init__(self, co2_emissions, nbp, sta, sy=1850, ey=2300, dt=1.0, pi_co2=283.0, dbg=0,
                 initial_C_reservoirs = [730.0, 140.0, 10040.0, 26830.0], advection='LOSCAR',
                 depths=[100, 250, 900], spinupLength=5000, dt_spinup=1.0):
    
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
        self.spinupLength = spinupLength
        self.dt_spinup = dt_spinup
        
        self.advection = advection
    
        
        ####################################################################################
        ################          Tuning parameters          ###############################
        ####################################################################################
            
        
        ## advection rates for LOSCAR-like advection
        # default values are from LOSCAR (Zeebe, 2012)
        self.k_mix_warm_int = 6.3e7
        self.k_mix_cold_deep = 1.7e7
        self.k_conveyor = 2.0e7

        ## advection rates for Lenton-like advection
        # default values are from Lenton (2000)
        self.k_warm_int_exchange =  12.5e7
        self.k_int_deep_exchange = 20.0e7
        self.k_THC_over = 6.5e7
        self.k_hl_over = 4.87e7

        # Overturning circulation dependent on Warming?
        self.advection_Tdep_frac = 0   # Set to 0 in default setup.
        
        ## gas exchange rate
        self.k_gasex = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1


        self.withTdepSurfTemp = True
        self.withTdepSurfSal = True
        self.withTdepSurfAlk = True
        self.withTdepBiopump = True



        ####################################################################################
        ####################################################################################
        ####################################################################################
    
        # model settings
        self.warm_area_fraction = 0.85
        self.cold_area_fraction = 1.0 - self.warm_area_fraction

        # Setting the depths and volumes of the ocean
        self.update_depths(depths)

        

        ### Initialisation of variables

        ## These neeed to be adjusted to ensure convergence of pH solver in first iteration
        self.pH_oce_cold_init = 8.36
        self.pH_oce_warm_init = 8.14

        # 1e-10 is typically sufficient
        self.pCO2_iterative_limit = 1e-10

        # reservoir containers
        self.C_atm = np.zeros((self.nyears)) + self.pi_co2 / self.epsilon

        self.C_warm_surf = np.zeros((self.nyears))
        self.C_cold_surf = np.zeros((self.nyears))
        self.C_int_wat = np.zeros((self.nyears))
        self.C_deep_oce = np.zeros((self.nyears))
        
        self.C_total = np.copy(self.C_atm)
        self.C_total = np.copy(self.C_atm)
        
        self.reservoirs = [self.C_warm_surf, self.C_cold_surf, self.C_int_wat, self.C_deep_oce]
        
        # initial reservoir values
        self.reservoir_inits = initial_C_reservoirs
        
        # initialize the carbon reservoir containers
        for res_i, init_reservoir in enumerate(self.reservoir_inits):
            self.reservoirs[res_i][0] = init_reservoir 
            self.C_total[0] += init_reservoir
        
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
        self.d_surface_warm = depths[0]  # m
        self.d_surface_cold = depths[1]
        self.d_intermediate = depths[2]
        
        self.V_warm = self.d_surface_warm * self.ocean_surface_area * self.warm_area_fraction
        self.V_cold = self.d_surface_cold * self.ocean_surface_area * self.cold_area_fraction
        self.V_int = self.d_intermediate * self.ocean_surface_area
        self.V_deep = self.ocean_volume - self.V_warm - self.V_cold - self.V_int

        self.__update_surface_ocean_inputs()        

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

        # Surface ocean alkalinit
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
        self.T0_cold_surface = self.__quadraticFit(self.d_surface_cold, T0_cold_params)
        self.T0_warm_surface = self.__quadraticFit(self.d_surface_warm, T0_warm_params)
        self.T_cold_surface_Tslope = self.__quadraticFit(self.d_surface_cold, T_slope_cold_params)
        self.T_warm_surface_Tslope = self.__quadraticFit(self.d_surface_warm, T_slope_warm_params)
        
        self.S0_cold_surface = self.__quadraticFit(self.d_surface_cold, S0_cold_params)
        self.S0_warm_surface = self.__quadraticFit(self.d_surface_warm, S0_warm_params)
        self.S_cold_surface_Tslope = self.__quadraticFit(self.d_surface_cold, S_slope_cold_params)
        self.S_warm_surface_Tslope = self.__quadraticFit(self.d_surface_warm, S_slope_warm_params)

        self.Alk0_cold_surface = self.__quadraticFit(self.d_surface_cold, Alk0_cold_params)
        self.Alk0_warm_surface = self.__quadraticFit(self.d_surface_warm, Alk0_warm_params)
        self.Alk_cold_surface_Tslope = self.__quadraticFit(self.d_surface_cold, Alk_slope_cold_params)
        self.Alk_warm_surface_Tslope = self.__quadraticFit(self.d_surface_warm, Alk_slope_warm_params)

        self.bio_pump_cold = self.__quadraticFit(self.d_surface_cold, Biopump_cold_params)
        self.bio_pump_warm = self.__quadraticFit(self.d_surface_warm, Biopump_warm_params)
        self.bio_pump_cold_Tslope = self.__quadraticFit(self.d_surface_cold, Biopump_slope_cold_params)
        return


    def __quadraticFit(self, depth, params):
        return params[0]*depth**2 + params[1]*depth + params[2]

        

    def modify_initial_reservoirs(self,modifications):
        # modify the carbon reservoir containers
        for res_i, modification in enumerate(modifications):
            self.reservoirs[res_i][0] += modification
            self.C_total[0] += modification
        
    def spinup(self):

        if self.dbg==1: print('Spinning up the model')

        tmp_dt = self.dt
        self.dt = self.dt_spinup

        for i in range(int(self.spinupLength/self.dt)):  
        
            self.__calc_surface_ocean_flux(0, 0.0)
            self.ocean_flux_total[0] = self.ocean_flux_warm[0] + self.ocean_flux_cold[0]

            ##### Updating reservoirs due to ocean-flux
            self.C_warm_surf[0] = self.C_warm_surf[0] + self.dt * self.ocean_flux_warm[0]
            self.C_cold_surf[0] = self.C_cold_surf[0] + self.dt * self.ocean_flux_cold[0]
                
            ########### ocean internal restructuring ###########

            
            self.__calc_total_carbon(0)
            if self.dbg == 1:
                print('  '+str(i))
                print('   ... ocean carbon uptake:', self.ocean_flux_total[0])
                print('   ... total ocean carbon before biopump:', self.C_total[0])
   
            #### Biological carbon pump
            self.__biological_carbon_pump(0, 0.0)

            self.__calc_total_carbon(0)
            
            if self.dbg == 1:
                print('   ... total ocean carbon after biopump:', self.C_total[0])

            ##### Advect the oceanic carbon
            tmp_C_warm, tmp_C_cold, tmp_C_int, tmp_C_deep = self.__advect_ocean_tracer(0, 0.0, 
                                                                        self.C_warm_surf[0] / self.V_warm,
                                                                        self.C_cold_surf[0] / self.V_cold,
                                                                        self.C_int_wat[0]   / self.V_int,
                                                                        self.C_deep_oce[0]  / self.V_deep)


            ## Final update of the tracer concentrations
            self.C_warm_surf[0] = tmp_C_warm * self.V_warm
            self.C_cold_surf[0] = tmp_C_cold * self.V_cold
            self.C_int_wat[0]   = tmp_C_int  * self.V_int
            self.C_deep_oce[0]  = tmp_C_deep * self.V_deep

            self.__calc_total_carbon(0)
            if self.dbg == 1:
                print('  '+str(i))
                print('   ... total ocean carbon after advection:', self.C_total[0])


        self.dt = tmp_dt
        if self.dbg == 1: print('Spinup done')

        
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
        
            self.__calc_surface_ocean_flux(i, self.STA[i])
            self.ocean_flux_total[i] = self.ocean_flux_warm[i] + self.ocean_flux_cold[i]

            ##### Updating reservoirs due to ocean-flux
            self.C_warm_surf[i] = self.C_warm_surf[i] + self.dt * self.ocean_flux_warm[i]
            self.C_cold_surf[i] = self.C_cold_surf[i] + self.dt * self.ocean_flux_cold[i]

            self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
        
            self.__calc_pCO2_atm(i+1)          
            
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            self.__biological_carbon_pump(i, self.STA[i])

            ##### Advect the oceanic carbon
            tmp_C_warm, tmp_C_cold, tmp_C_int, tmp_C_deep = self.__advect_ocean_tracer(i, self.STA[i], 
                                                                                self.C_warm_surf[i] / self.V_warm,
                                                                                self.C_cold_surf[i] / self.V_cold,
                                                                                self.C_int_wat[i]   / self.V_int,
                                                                                self.C_deep_oce[i]  / self.V_deep)
        
            ## Final update of the tracer concentrations
            self.C_warm_surf[i+1] = tmp_C_warm * self.V_warm
            self.C_cold_surf[i+1] = tmp_C_cold * self.V_cold
            self.C_int_wat[i+1]   = tmp_C_int  * self.V_int
            self.C_deep_oce[i+1]  = tmp_C_deep * self.V_deep

            self.__calc_total_carbon(i+1)
        return
    
    def __calc_total_carbon(self,i):
        self.C_total[i] = self.C_atm[i] + self.C_warm_surf[i] + self.C_cold_surf[i] + self.C_int_wat[i] + self.C_deep_oce[i]
        return

    def __calc_pCO2_atm(self,i):
        self.pCO2_atm[i] = self.C_atm[i] * self.epsilon
        return
    
    def __calc_surface_ocean_flux(self, i, Ts):
        # tropical and high latitude oceans should warm differently under global warming !
        T_warm = self.T_warm_surface_Tslope * Ts + self.T0_warm_surface
        T_cold = self.T_cold_surface_Tslope * Ts + self.T0_cold_surface
    
        dic_w = self.C_warm_surf[i] / self.mol_to_GT_C / self.V_warm
        dic_c = self.C_cold_surf[i] / self.mol_to_GT_C / self.V_cold

        salt_w = self.S0_warm_surface + self.S_warm_surface_Tslope * Ts
        salt_c = self.S0_cold_surface + self.S_cold_surface_Tslope * Ts
        talk_w = self.Alk0_warm_surface + self.Alk_warm_surface_Tslope * Ts
        talk_c = self.Alk0_cold_surface + self.Alk_cold_surface_Tslope * Ts

        if self.dbg == 1:
            print(i, 'calculating surface ocean fluxes')
            print('   T_warm: ', T_warm)
            print('   T_cold: ', T_cold)
            print('   S_warm: ', salt_w,)
            print('   S_cold: ', salt_c)
            print('   talk_warm: ', talk_w)
            print('   talk_cold: ', talk_c)
            print('   dic_warm: ', dic_w)
            print('   dic_warm: ', dic_c)


        self.pH_oce_warm[i], self.pCO2_oce_warm[i] = self.__calc_ocean_pCO2(T_warm, salt_w, talk_w, dic_w, self.pH_oce_warm[i-1])
        self.pH_oce_cold[i], self.pCO2_oce_cold[i] = self.__calc_ocean_pCO2(T_cold, salt_c, talk_c, dic_c, self.pH_oce_cold[i-1])

        if self.dbg == 1:
            print('....calulcated the ocean pH:', self.pH_oce_warm[i], self.pH_oce_cold[i])

        
        self.ocean_flux_warm[i] = self.k_gasex * self.mol_to_GT_C \
                                 * self.ocean_surface_area * self.warm_area_fraction \
                                 * (self.pCO2_atm[i] - self.pCO2_oce_warm[i])
        
        self.ocean_flux_cold[i] = self.k_gasex * self.mol_to_GT_C \
                                 * self.ocean_surface_area * self.cold_area_fraction \
                                 * (self.pCO2_atm[i] - self.pCO2_oce_cold[i])


        if self.dbg == 1:
            print( '....calulcated the surface fluxes:', self.ocean_flux_warm[i], self.ocean_flux_cold[i])
        return

    def __calc_ocean_pCO2(self, TC, S, alk, dic, ph_old=8.1):
        ''' This is taken from the iLOSCAR model
        (https://github.com/Shihan150/iloscar/tree/main)
        Zeebe (2012), https://doi.org/10.5194/gm/d-5-149-2012
        Li et al. (2024), https://doi.org/10.1016/j.gloplacha.2024.104413
        
        This solution to the carbonate chemistry is based on Follows et al. (2006, https://doi.org/10.1016/j.ocemod.2005.05.004)
        
        '''
    
        alk*=1.0e-3
        dic*=1.0e-3
        
        TK = TC + 273.15
        bor = (432.5 * (S/35.0))*1.0e-6
        #bor = 1.179e-5 * S

        # Weiss, 1974
        kh = np.exp( 9345.17/TK - 60.2409 + 23.3585*np.log(TK/100) \
                    + S * (0.023517 - 2.3656e-4 * TK + 0.0047036e-4 * TK * TK) )
        
        # Mehrbach et al. 1973, efit by Lueker et al. (2000)
        k1 = 10**(-(3633.86/TK - 61.2172 + 9.6777 * np.log(TK)-0.011555* S + 1.152e-4*S*S))
        k2 = 10**(-(471.78/TK +25.9290 - 3.16967 * np.log(TK) - 0.01781 * S + 1.122e-4*S*S))

        # Dickson, 1990 in Dickson and Goyet, 1994, Chapter 5
        kb = np.exp((-8966.90 - 2890.51 * np.sqrt(S) - 77.942 * S \
                + 1.728 * S**1.5 - 0.0996*S**2) / TK + (148.0248 + 137.194 * np.sqrt(S) + 1.62142 * S) \
                + (-24.4344 - 25.085 * np.sqrt(S) - 0.2474 * S) * np.log(TK) + 0.053105 * np.sqrt(S) * TK)
        
        # Millero (1995) in Dickson and Goyet, 1994, Chapter 5
        kw = np.exp((-13847.26/TK + 148.96502 - 23.6521 * np.log(TK)) + \
                    (118.67/TK -5.977 +1.0495 * np.log(TK)) * np.sqrt(S) - 0.01615 * S )


        # Initial guess of H         
        hx = 10.**(-ph_old)

        for it in range(20):
            hgss = hx
            
            bo4hg = bor*kb/(hgss+kb)
            fg = -bo4hg - (kw/hgss) + hgss
            calkg = alk + fg
            gam = dic/calkg

            tmp = (1-gam)*(1-gam)*k1*k1 - 4 *k1 *k2 *(1-2*gam)    
            hx = 0.5 *((gam-1)*k1+np.sqrt(tmp)) 

            change = np.abs(hx-hgss) - self.pCO2_iterative_limit
            if change < 0: break
                
        if self.dbg == 2: print(it, 'steps needed in iterative pH solver')

        co2 = dic/(1+k1/hx + k1*k2/hx/hx)
        pCO2 = co2 / kh * 1.e6
        pH = -np.log10(hx)

        if pH <= 0:
            print('pH got negative! Model settings:')
            print('  d_warm:', self.d_surface_warm, 'm')
            print('  d_cold:', self.d_surface_cold, 'm')
            print('  d_int:', self.d_intermediate, 'm')
            print('  k_mix_WI:', self.k_mix_warm_int*1e-6, 'Sv')
            print('  k_mix_CD:', self.k_mix_cold_deep*1e-6, 'Sv')
            print('  k_conveyor:', self.k_conveyor*1e-6, 'Sv')
            print('  k_gasex:', self.k_gasex, 'm/s')
            #sys.exit('Error: negative pH!')
            
        return pH, pCO2

    def __biological_carbon_pump(self, i, T_surf):
        C_exp_warm = self.bio_pump_warm
        C_exp_cold = self.bio_pump_cold + self.bio_pump_cold_Tslope * T_surf
        C_inp_int  = C_exp_warm + C_exp_cold * (1.0 - self.bio_pump_cold_eff)
        C_inp_deep = C_exp_cold * self.bio_pump_cold_eff

        self.C_warm_surf[i] = self.C_warm_surf[i] - self.dt * C_exp_warm
        self.C_cold_surf[i] = self.C_cold_surf[i] - self.dt * C_exp_cold
        self.C_int_wat[i]   = self.C_int_wat[i]   + self.dt * C_inp_int
        self.C_deep_oce[i]  = self.C_deep_oce[i]  + self.dt * C_inp_deep
        return


    def __advect_ocean_tracer(self, i, Ts, C_w, C_c, C_i, C_d):

        ##### Advect the oceanic carbon
        if self.advection=='Lenton':
            C_warm, C_cold, C_int, C_deep = self.__advect_ocean_tracer_Lenton(i, Ts, C_w, C_c, C_i, C_d)
        elif self.advection=='LOSCAR':
            C_warm, C_cold, C_int, C_deep = self.__advect_ocean_tracer_LOSCAR(i, Ts, C_w, C_c, C_i, C_d)
        else:
            sys.exit('No correct choice of advection model! Can be "Lenton" or "LOSCAR", current value: '+self.advection)
            
        return C_warm, C_cold, C_int, C_deep


    def __advect_ocean_tracer_LOSCAR(self, i, Ts, C_w, C_c, C_i, C_d):
        ### assumes a list of tracer concentrations is given in order:
        ## C_warm, C_cold, C_int, C_deep
        
        k_T = self.k_conveyor
        # Temperature dependence of advection
        k_T  += self.advection_Tdep_frac * self.k_conveyor * Ts

        k_wi = self.k_mix_warm_int
        k_cd = self.k_mix_cold_deep

        dts = self.dt * self.secondsPerYear
        
        Cw = C_w + (dts / self.V_warm) * ( k_wi * (C_i - C_w) )
        Cc = C_c + (dts / self.V_cold) * ( k_T * (C_i - C_c) + k_cd * (C_d - C_c) )
        Ci = C_i + (dts / self.V_int)  * ( k_T * (C_d - C_i) - k_wi * (C_i - C_w) )
        Cd = C_d + (dts / self.V_deep) * ( k_T * (C_c - C_d) - k_cd * (C_d - C_c) )
    
        return Cw, Cc, Ci, Cd

    def __advect_ocean_tracer_Lenton(self, i, Ts, C_w, C_c, C_i, C_d):
        ### assumes a list of tracer concentrations is given in order:
        ## C_warm, C_cold, C_int, C_deep 
        
        k_T  = self.k_THC_over + self.advection_Tdep_frac * self.k_THC_over * Ts
        k_U  = self.k_hl_over  + self.advection_Tdep_frac * self.k_hl_over  * Ts

        k_O = k_T + k_U
        k_wi = self.k_warm_int_exchange
        k_id = self.k_int_deep_exchange

        dts = self.dt * self.secondsPerYear
        
        Cw = C_w + (dts / self.V_warm) * ( (k_T+k_wi)*C_i - (k_T+k_wi)*C_w )
        Cc = C_c + (dts / self.V_cold) * ( k_T*C_w + k_U*C_i - k_O*C_c )
        Ci = C_i + (dts / self.V_int) * ( (k_O+k_id)*C_d + k_wi*C_w - (k_O + k_wi + k_id)*C_i )
        Cd = C_d + (dts / self.V_deep) * ( k_O*C_c + k_id*C_i - (k_O + k_id)*C_d )
            
        return Cw, Cc, Ci, Cd


    def getInitialCReservoirs(self):  return [self.C_warm_surf[0], self.C_cold_surf[0], self.C_int_wat[0], self.C_deep_oce[0]]
    def getFinalCReservoirs(self):    return [self.C_warm_surf[-1], self.C_cold_surf[-1], self.C_int_wat[-1], self.C_deep_oce[-1]]

    def getOceanCarbon(self): return self.C_warm_surf + self.C_cold_surf + self.C_int_wat + self.C_deep_oce

    def getTime(self): return np.linspace(self.startyear, self.endyear, self.nyears)
    def getTimeYearly(self): return np.linspace(self.startyear, self.endyear, int((self.nyears-1)*self.dt)) + 0.5

    def getAtmCGrowth(self): return np.append(0, self.C_atm[1:]-self.C_atm[:-1]) / self.dt

    def getOceanFluxTotalYearly(self): return np.mean(self.ocean_flux_total[:-1].reshape(-1,self.nStepsPerYear), axis=1)























class ForcedOceanCarbonModel:
    """

    """

    # Tuning parameters have default values from Lenton (2000)
    def __init__(self, co2_emissions, nbp, sta, ModelType='4BoxOcean', sy=1850, ey=2300, dt=1.0, dbg=0,
                 pi_co2=283.0, initial_C_reservoirs = [730.0, 140.0, 10000.0, 26830.0],
                 depths=[100, 250, 1000], spinupLength=5000, advection='LOSCAR', airSeaExchange='LOSCAR'):


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
        self.spinupLength = spinupLength
         
        
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
            self.k_mix_CD  = 17.0e7
            self.k_HL_over = 20.0e7
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
            self.ocean_flux_total[0] = self.ocean_flux_warm[0] + self.ocean_flux_cold[0]

            ##### Updating reservoirs due to ocean-flux
            self.C_oce[0,0] = self.C_oce[0,0] + self.dt * self.ocean_flux_warm[0]
            self.C_oce[1,0] = self.C_oce[1,0] + self.dt * self.ocean_flux_cold[0]

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

        # Surface ocean alkalinit
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
