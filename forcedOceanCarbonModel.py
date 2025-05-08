import numpy as np
import sys




class CarbonBoxModel:
    """A class of a carbon cycle model orginially based on that from Lenton (2000).
    The model consists of 7 distinct C reservoirs (4 oce, 2 land, 1 atm).

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

    # Tuning parameters have default values from Lenton (2000)
    def __init__(self, co2_emissions, nbp, tas, sy=1850, ey=2300, dt=1.0, pi_co2=283.0,
                 initial_C_reservoirs = [730.0, 140.0, 10040.0, 26830.0],
                 depths=[100, 100, 1000]):
    
        self.emission   = co2_emissions # emissions should enter in GT C per year
        self.land_flux_total = nbp
        self.STA = sta

        self.pi_co2 = pi_co2
    
        # Timestep:
        self.startyear = sy
        self.endyear = ey
        self.dt = dt
        self.dts = self.dt * 365.25 * 24 * 3600.
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        
        ####################################################################################
        ################          Tuning parameters          ###############################
        ####################################################################################
            
        
        ## advection rates for LOSCAR-like advection
        # values are defaults from LOSCAR (Zeebe, 2012)
        self.k_mix_warm_int = 6.3e7
        self.k_mix_cold_deep = 1.7e7
        self.k_conveyor = 2.0e7
        
        # Overturning circulation dependent on Warming?
        self.advection_Tdep_frac = 0   # Set to 0 in default setup.
        
        ## gas exchange rate
        self.k_gasex = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1

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

        # reservoir containers
        self.C_atm = np.zeros((self.nyears)) + self.pi_co2 / self.epsilon

        self.C_warm_surf = np.zeros((self.nyears))
        self.C_cold_surf = np.zeros((self.nyears))
        self.C_int_wat = np.zeros((self.nyears))
        self.C_deep_oce = np.zeros((self.nyears))
        
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
        
        self.land_flux_total = np.zeros((self.nyears))
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

        Tslope_warm_params = np.asarray([1.09e-6, -9.54e-4, 0.69])
        T0_warm_params = np.asarray([2-13e-5, -2.5e-2, 20.7])

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
        
    def spinup(self, nyears=3000):

        for i in range(nyears):  
        
            self.__calc_surface_ocean_flux(0, 0.0)
            self.ocean_flux_total[0] = self.ocean_flux_warm[0] + self.ocean_flux_cold[0]

            ##### Updating reservoirs due to ocean-flux
            self.C_warm_surf[0] = self.C_warm_surf[0] + self.dt * self.ocean_flux_warm[0]
            self.C_cold_surf[0] = self.C_cold_surf[0] + self.dt * self.ocean_flux_cold[0]
                
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            self.__biological_carbon_pump(0, 0.0)
            
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

        
    def integrate(self):

        self.spinup()
        
        for i in range(self.nyears-1):

            ############ Updating atmospheric CO2 based on land flux and emissions
            if not self.co2_is_fixed:
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
            if not self.co2_is_fixed:
                self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
        
            self.__calc_pCO2_atm(i+1)          
            
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            self.__biological_carbon_pump(i, self.STA[i])
            
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
        talk_w = self.Alk0_warm_surface + self.Alk_warm_Tslope * Ts
        talk_c = self.Alk0_cold_surface + self.Alk_cold_Tslope * Ts


        self.pH_oce_warm[i], self.pCO2_oce_warm[i], it = self.__calc_ocean_pCO2(T_warm, salt_w, talk_w, dic_w, self.pH_oce_warm[i-1])
        self.pH_oce_cold[i], self.pCO2_oce_cold[i], it = self.__calc_ocean_pCO2(T_cold, salt_c, talk_c, dic_c, self.pH_oce_cold[i-1])

        
        self.ocean_flux_warm[i] = self.k_gasex * self.mol_to_GT_C \
                                 * self.ocean_surface_area * self.warm_area_fraction \
                                 * (self.pCO2_atm[i] - self.pCO2_oce_warm[i])
        
        self.ocean_flux_cold[i] = self.k_gasex * self.mol_to_GT_C \
                                 * self.ocean_surface_area * self.cold_area_fraction \
                                 * (self.pCO2_atm[i] - self.pCO2_oce_cold[i])
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

        ### Just calculates H+ once in this setup, because it basically always converges in the first try, if dt <= 1 year

        hgss = hx
            
        bo4hg = bor*kb/(hgss+kb)
        fg = -bo4hg - (kw/hgss) + hgss
        calkg = alk + fg
        gam = dic/calkg

        tmp = (1-gam)*(1-gam)*k1*k1 - 4 *k1 *k2 *(1-2*gam)
            
        hx = 0.5 *((gam-1)*k1+np.sqrt(tmp)) 
                
        

        co2 = dic/(1+k1/hx + k1*k2/hx/hx)
        pCO2 = co2 / kh * 1.e6
        pH = -np.log10(hx)

        if pH <= 0: sys.exit('Error: negative pH!')
            
        return pH, pCO2, it


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
        ### assumes a list of tracer concentrations is given in order:
        ## C_warm, C_cold, C_int, C_deep
        
        k_T = self.k_conveyor
        # Temperature dependence of advection
        k_T  += self.advection_Tdep_frac * self.k_conveyor * Ts

        k_wi = self.k_mix_warm_int
        k_cd = self.k_mix_cold_deep
        
        Cw = C_w + (self.dts / self.V_warm) * ( k_wi * (C_i - C_w) )
        Cc = C_c + (self.dts / self.V_cold) * ( k_T * (C_i - C_c) + k_cd * (C_d - C_c) )
        Ci = C_i + (self.dts / self.V_int)  * ( k_T * (C_d - C_i) - k_wi * (C_i - C_w) )
        Cd = C_d + (self.dts / self.V_deep) * ( k_T * (C_c - C_d) - k_cd * (C_d - C_c) )
    
        return Cw, Cc, Ci, Cd

    def getInitialCReservoirs(self):  return [self.C_warm_surf[0], self.C_cold_surf[0], self.C_int_wat[0], self.C_deep_oce[0]]
    def getFinalCReservoirs(self):    return [self.C_warm_surf[-1], self.C_cold_surf[-1], self.C_int_wat[-1], self.C_deep_oce[-1]]

