
import numpy as np
import sys
import carbon_chemistry as carchm


class IMPRSModel:
    """This model is the one from the IMPRS introductory course"""
    k_a = 2.12
    C0_atm = 600.0
    #P0_atm = 283.0

    beta_land = 0.4
    xi_land = 1.8
    tau_land = 41.0
    npp_0 = 60.0
    C0_land = 2460.0
    
    
    #ocean_depth = 3620.0
    #zeta = 10.5
    #C0_ocean_total = 37800.0
    delta = 0.02
    gamma = 0.005
    eq_air_frac = 1.0 / 7.0  
    c_star = 10.8e9
        
    def __init__(self, co2_emissions, sy=1850, ey=2300, dt=1.0, dbg=False, 
                 eta_deep = 0.0, eta_c = 60.0e-12, delta=0.015, lambda_rad = 1.77,
                 lambda_oce = 0.75, beta_co2 = 5.77):
        self.dbg = dbg

        # Timestep:
        self.startyear = sy
        self.endyear = ey
        # time step is one year, but in seconds
        self.dt = dt
        self.dts = self.dt * 365.25 * 24 * 3600.
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        
        
        # tuning parameters
        self.eta_deep = eta_deep
        self.eta_c = eta_c * 365.25 * 24 * 3600
        self.delta = delta
        self.lambda_rad = lambda_rad
        self.lambda_oce = lambda_oce
        self.beta_co2 = beta_co2
        self.k_o = self.k_a * (1.0 - self.eq_air_frac) / self.eq_air_frac * self.delta
        
        
        self.emission   = co2_emissions # emissions should enter in GTC (per year)

        
        self.T_surf = np.zeros((self.nyears))  # Temperature anomaly in mixed layer
        self.T_deep = np.zeros((self.nyears))  # Temperature anomaly in deep ocean
        
        self.C_atm = np.zeros((self.nyears))  # Carbon anomaly in atmosphere
        self.C_land = np.zeros((self.nyears)) # Carbon anomaly in vegetation
        self.C_surf = np.zeros((self.nyears)) # Carbon anomaly in surf ocean
        self.C_deep = np.zeros((self.nyears)) # Carbon anomaly in deep ocean
        
        # Variables for the fluxes
        self.air_sea_exchange = np.zeros((self.nyears))
        self.npp = np.zeros((self.nyears))
        self.respiration = np.zeros((self.nyears))
        self.surf_deep_exchange = np.zeros((self.nyears))
        
    def integrate(self, silent=True):
        
        if not silent: print('Start integrating...')
        for i in range(0,self.nyears-1):
            
            self.__updateCarbon(i)
            self.__updateTemp(i)
            
        if not silent: print('  ... finished')
    
    
    def __updateCarbon(self,i):
        
        # calculate the tendencies
        self.__calc_land_carbon_flux(i)
        self.__calc_air_sea_exchange(i)
        self.__calc_surf_deep_exchange(i)
        
        # update the reservoirs
        self.C_land[i+1] = self.C_land[i] + self.dt * (self.npp[i] - self.respiration[i])
        self.C_surf[i+1] = self.C_surf[i] + self.dt * (self.air_sea_exchange[i] - self.surf_deep_exchange[i])
        self.C_deep[i+1] = self.C_deep[i] + self.dt * self.surf_deep_exchange[i]
        self.C_atm[i+1] = self.C_atm[i] + self.dt * (self.emission[i] - (self.npp[i] - self.respiration[i]) - self.air_sea_exchange[i])
        
        # Debug output
        if self.dbg: print()
        if self.dbg: print(i, 'lnd_carbon_flux: ', self.__land_carbon_flux(i))
        if self.dbg: print(i, 'air_sea_exchange: ', self.__air_sea_exchange(i))
        if self.dbg: print(i, 'surf_deep_exhange: ', self.__surf_deep_exchange(i))
        
        if self.dbg: print(i, 'new C land: ', self.C_land[i+1])
        if self.dbg: print(i, 'new C atm: ', self.C_atm[i+1])
        if self.dbg: print(i, 'new C surf: ', self.C_surf[i+1])
        if self.dbg: print(i, 'new C deep: ', self.C_deep[i+1])
            
    def __updateTemp(self,i):
        self.T_surf[i+1] = self.T_surf[i] + self.dts * (1.0 / self.delta / self.c_star) \
                            * ( 
                                - self.lambda_rad * self.T_surf[i]
                                + self.beta_co2 * np.log(self.C_atm[i]/self.C0_atm + 1.0)
                                - self.eta_deep * (self.T_surf[i] - self.T_deep[i])
                                - self.lambda_oce * (self.T_surf[i] - self.T_deep[i])
                              )
        self.T_deep[i+1] = self.T_deep[i] + self.dts * self.eta_deep * ( self.T_surf[i] - self.T_deep[i] ) \
                                            / (1.-self.delta) / self.c_star
        
        if self.dbg: print(i, 'new T surf: ', self.T_surf[i+1])
        if self.dbg: print(i, 'new T deep: ', self.T_deep[i+1])

        
    def __calc_land_carbon_flux(self,i):
        self.npp[i] = self.npp_0 * (1.0 + self.beta_land * np.log(self.C_atm[i]/self.C0_atm + 1.0)) 
        self.respiration[i] = (self.C_land[i] + self.C0_land) * self.xi_land**(self.T_surf[i]/10.0) / self.tau_land
        return 
        
        
    def __calc_air_sea_exchange(self,i):
        self.air_sea_exchange[i] = self.gamma * (self.C_atm[i]/self.k_a - self.C_surf[i]/self.k_o)
        return 
    
    def __calc_surf_deep_exchange(self,i):
        self.surf_deep_exchange[i] = self.eta_c * (self.C_surf[i]/self.delta - self.C_deep[i]/(1.0-self.delta))
        return
    
    
    
    def getTsurf(self): return self.T_surf 
    def getCO2(self): return (self.C_atm + self.C0_atm) / self.k_a
    def getNPP(sef): return self.npp
    def getRespiration(self): return self. respiration
    def getAirSeaExchange(self): return self.air_sea_exchange
    def getSurfDeepExchange(self): return self.surf_deep_exchange








class ThreeLayerOceanModel:
    """ A class of a 3-Layer ocean model.
    This is the classic model from Li et al. (2020).
    Radiative forcing is read in from CMIP6 data.
    Temperature and carbon cycle response can be
    chosen from different options."""
    def __init__(self, co2_emissions, ERF_nonco2, ccycle_type='default', sy=1850, ey=2300, dt=1.0, dbg=False):
        
        self.dbg = dbg
        
        # Timestep:
        self.startyear = sy
        self.endyear = ey
        # time step is one year, but in seconds
        self.dt = dt
        self.dts = self.dt * 365.25 * 24 * 3600.
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        self.time = np.linspace(self.startyear, self.endyear, self.nyears)
        
        
        #### Model parameters
        self.alpha = 1.16 # W m^-2 K^-1
        self.w_E = 1.0e-6 # m s^-1
        self.w_D = 0.4e-6 # m s^-1
        self.C_ML = 2.0e8 # W s m^-2 K^-1 
        self.R4co2 = 7.4  # W m^-2
        self.beta = 1.33  # dmnl
    
        self.epsilon = 0.47 # ppmv GtC^-1
        self.delta   = 0.0215 # yr^-1
        self.B       = 0.15 * self.delta * self.epsilon
        
        self.pi_co2  = 283.0

        ### initialize values        
        self.T_ML = np.zeros((self.nyears))  # Temperature anomaly in mixed layer
        self.T_TC = np.zeros((self.nyears))  # Temperature anomaly in thermocline
        self.T_D  = np.zeros((self.nyears))  # Temperature anomaly in deep ocean
        self.T_surf  = np.zeros((self.nyears))  # Surface temperature anomaly
        
        self.co2        = np.zeros((self.nyears))  # atmospheric CO2 concentration in ppm
        self.co2[0]     = self.pi_co2
        self.emission   = co2_emissions            # emissions should enter in GTC (per year)
        self.F          = np.zeros((self.nyears))  # cumulative emissions
        self.ERF        = np.zeros((self.nyears))  # Total Radiative Forcing
        self.ERF_nonco2 = ERF_nonco2              # effective radiative forcing should enter in Wm^-2
        self.ERF_co2    = np.zeros((self.nyears))
        
        self.ccycle_type = ccycle_type
        
        if self.ccycle_type == 'FAIR':
            self.FAIR_co2_containers = np.zeros((4,self.nyears))
            self.lifetimes = np.array([1.0e9, 394.4, 36.54, 4.304])
            self.partitioning = np.array([0.2173, 0.224, 0.2824, 0.2763])
            
    def integrate(self):
        
        print('Start integrating', self.ccycle_type, '...')
        for i in range(0,self.nyears-1):
            
            if self.ccycle_type == 'default':
                
                self.__calcERF(i)
                self.__updateCO2(i)
                
            elif self.ccycle_type == 'FAIR':
                
                self.__calcERF_FAIR(i)
                self.__updateCO2_FAIR(i)
                
            else:
                sys.exit('Wrong ccycle type given!')
            
            self.__updateOceanModel(i)
            
        print('  ... finished')
    
        
    def __calcERF(self, i):
        self.ERF_co2[i] = 0.5 * self.R4co2 * np.log(self.co2[i] / self.pi_co2) / np.log(2.0)
        self.ERF[i] =  self.ERF_co2[i] + self.ERF_nonco2[i]
        
        
    def __updateCO2(self, i):
        self.co2[i+1] = self.co2[i] + self.dt * (self.epsilon * self.emission[i] \
                                     + self.B * self.F[i] \
                                     - self.delta * (self.co2[i]-self.pi_co2))
        self.F[i+1] = self.F[i] + self.dt * self.emission[i]
        
        
    def __calcERF_FAIR(self, i):
        f1_co2 = 4.57
        f3_co2 = 0.086
        scale_co2 = 1.0202
        co2_topo_adj = 0.05
        
        self.ERF_co2[i] = scale_co2 * (f1_co2 * np.log(self.co2[i] / self.pi_co2) + \
                                       (np.sqrt(f3_co2**2 * self.co2[i]) - np.sqrt(f3_co2**2 * self.pi_co2)) * \
                                       (1.0 + co2_topo_adj))
        self.ERF[i] = self.ERF_co2[i] + self.ERF_nonco2[i]
        
        
    def __updateCO2_FAIR(self, i):
        iirf0 = 33.4377
        iirf_airborne = 0.001916
        iirf_uptake = 0.00328
        iirf_temperature = 2.2042
        
        g0 = 0.01018
        g1 = 8.0#11.413
        
        #print('Going into calculation of iirf:')
        #print('   cum. emis (GT C): ', self.F[i])
        #print('   CO2 conc. (ppm)', self.co2[i])
        #print('   T surf (deg):', self.T_surf[i])
        #print('   ')
        
        
        iirf = iirf0 + iirf_uptake * ( self.F[i] - (self.co2[i]-self.pi_co2)/self.epsilon) \
                     + iirf_temperature * self.T_surf[i] \
                     + iirf_airborne * (self.co2[i]-self.pi_co2)/self.epsilon
                
        lsf = g0 * np.exp(iirf/g1) # lifetime scale factor
        
        for ri in range(4):
            self.FAIR_co2_containers[ri,i+1] = self.FAIR_co2_containers[ri,i] \
                                               + self.emission[i] * self.partitioning[ri] \
                                               - self.FAIR_co2_containers[ri,i] / (self.lifetimes[ri] * lsf)

            
        #print('GT in containers (before): ', np.sum(self.FAIR_co2_containers[:,i]))
        #print('iirf: ', iirf)
        #print('lsf: ',lsf)
        #print('emissions: ', self.emission[i])
        #print('GT in containers (after): ', np.sum(self.FAIR_co2_containers[:,i+1]))
        #print()
        #print()
        
        self.co2[i+1] = np.sum(self.FAIR_co2_containers[:,i+1]) * self.epsilon + self.pi_co2
        
        self.F[i+1] = self.F[i] + self.dt * self.emission[i]
        
    def __updateOceanModel(self, i):
        
        self.T_ML[i+1] = self.T_ML[i] + self.dts * ((self.ERF[i] - self.alpha*self.T_surf[i]) / self.C_ML - \
                                                   (self.w_E/self.Depths[0]) * (self.T_ML[i] - self.T_TC[i]))
        
        self.T_TC[i+1] = self.T_TC[i] + self.dts * ((self.w_E/self.Depths[1]) * (self.T_ML[i] - self.T_TC[i]) - \
                                                    (self.w_D/self.Depths[1]) * (self.T_TC[i] - self.T_D[i]))
        
        self.T_D[i+1] = self.T_D[i] + self.dts * ((self.w_D/self.Depths[2]) * (self.T_TC[i] - self.T_D[i]))
                
        self.T_surf[i+1] = self.beta * self.T_ML[i+1]
    
    
    def getCO2(self):       return self.co2
    def getTsurf(self):     return self.T_surf
    def getTml(self):       return self.T_ML
    def getTtc(self):       return self.T_TC
    def getTd(self):        return self.T_D
    def getERF(self):       return self.ERF
    def getERFCO2(self):    return self.ERF_co2
    def getERFnonCO2(self): return self.ERF_nonco2
    def getCumEmis(self):   return self.F
    def getEmis(self):      return self.emission
    def getTime(self):      return self.time










class EnergyAndCarbonBoxModel:
    """This is a box model for energy and Carbon in the Earth system.
    The energy component is that of Li et al.: 'Optimal temperature overshoot ...' (2020).
    The radiative forcing is calculated as in FaiR (Leach, 2020; Smith, 2018).
    The carbon box model is a combination of different existing carbon cycle models
    (Lenton (2000), LOSCAR (Zeebe, 2012)).
    Radiative forcing is read in from CMIP6 data.
    
    To-Dos:
        - make ocean energy and carbon transport consistent
        - replace pCO2 calculation with simplified versions, if it is good enough

    """

    def __init__(self, co2_emissions, ERF_nonco2, sy=1850, ey=2300, dt=1.0, dbg=0,
                  lfix_co2=False, co2fix=283.0, depths=[50,500,3150], advection='LOSCAR',
                  initial_C_reservoirs=[596.0, 550.0, 1500.0, 730.0, 140.0, 10040.0, 26830.0]):
        
        self.dbg = dbg
        
        # Timestep:
        self.startyear = sy
        self.endyear = ey
        self.dt = dt                                # in years
        self.dts = self.dt * 365.25 * 24 * 3600.    # in seconds
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        self.time = np.linspace(self.startyear, self.endyear, self.nyears)
        
        self.Depths = depths
        self.advection = advection 
        
        #### Model parameters
        # Energy model 
        self.alpha = 1.16 # W m^-2 K^-1
        self.w_E = 1.0e-6 # m s^-1
        self.w_D = 0.4e-6 # m s^-1
        self.C_ML = 2.0e8 # W s m^-2 K^-1 
        #self.R4co2 = 7.4  # W m^-2
        self.beta = 1.33  # dmnl
    
        # self.epsilon = 0.47 # ppmv GtC^-1
        # self.delta   = 0.0215 # yr^-1
        # self.B       = 0.15 * self.delta * self.epsilon
        
        self.pi_co2  = 283.0

        ### initialize values        
        self.T_ML    = np.zeros((self.nyears))  # Temperature anomaly in mixed layer
        self.T_TC    = np.zeros((self.nyears))  # Temperature anomaly in thermocline
        self.T_D     = np.zeros((self.nyears))  # Temperature anomaly in deep ocean
        self.T_surf  = np.zeros((self.nyears))  # Surface temperature anomaly (GSAT)
        
                
        self.co2        = np.zeros((self.nyears))  # atmospheric CO2 concentration in ppm
        self.co2[0]     = self.pi_co2
        self.emission   = co2_emissions            # emissions should enter in GTC (per year)
        self.F          = np.zeros((self.nyears))  # cumulative emissions
        self.ERF        = np.zeros((self.nyears))  # Total Radiative Forcing
        self.ERF_nonco2 = ERF_nonco2              # effective radiative forcing should enter in Wm^-2
        self.ERF_co2    = np.zeros((self.nyears))
        
        
        self.CarbonModel = self.CarbonBoxModel(co2_emissions, sy=self.startyear, ey=self.endyear,
                                          dt=self.dt, dbg=self.dbg, lfix_co2=lfix_co2, co2fix=co2fix,
                                          initial_C_reservoirs=initial_C_reservoirs, advection=advection,
                                          depths=[250, 250, 900])    # Tuned values for CarbonBoxModel


    def spinup(self, co2fix=283.0, years=3000, silent=False):

        ny = int(years/self.dt)+1
        zero_emissions = np.zeros(ny)
        
        # initiating a spinup model
        self.SpinupModel = self.CarbonBoxModel(zero_emissions, sy=0, ey=years, dt=self.dt, dbg=self.dbg,
                                               lfix_co2=True, co2fix=co2fix, advection=self.advection, isSpinup=True,
                                               depths=[self.CarbonModel.d_surface_warm, self.CarbonModel.d_surface_cold,
                                               self.CarbonModel.d_intermediate])

        
        # change tuning parameters to be as in the Carbon Model        
        self.SpinupModel.C_veg_steady        = self.CarbonModel.C_veg_steady          # 2-box land component from Lenton (2000)
        self.SpinupModel.k_turn              = self.CarbonModel.k_turn                # 2-box land component from Lenton (2000)
        self.SpinupModel.K_halfsat           = self.CarbonModel.K_halfsat             # 2-box land component from Lenton (2000)

        self.SpinupModel.k_conveyor          = self.CarbonModel.k_conveyor            # Advection as in LOSCAR (Zeebe, 2012)
        self.SpinupModel.k_mix_cold_deep     = self.CarbonModel.k_mix_cold_deep       # Advection as in LOSCAR (Zeebe, 2012)
        self.SpinupModel.k_mix_warm_int      = self.CarbonModel.k_mix_warm_int        # Advection as in LOSCAR (Zeebe, 2012)

        self.SpinupModel.k_int_deep_exchange = self.CarbonModel.k_int_deep_exchange   # Advection as in Lenton (2000)
        self.SpinupModel.k_warm_int_exchange = self.CarbonModel.k_warm_int_exchange   # Advection as in Lenton (2000)
        self.SpinupModel.k_hl_over           = self.CarbonModel.k_hl_over             # Advection as in Lenton (2000)
        self.SpinupModel.k_THC_over          = self.CarbonModel.k_THC_over            # Advection as in Lenton (2000)

        self.SpinupModel.k_gasex             = self.CarbonModel.k_gasex
        self.SpinupModel.include_bio_pump    = self.CarbonModel.include_bio_pump
        self.SpinupModel.bio_pump_warm       = self.CarbonModel.bio_pump_warm
        self.SpinupModel.bio_pump_cold       = self.CarbonModel.bio_pump_cold
        self.SpinupModel.bio_pump_cold_eff   = self.CarbonModel.bio_pump_cold_eff
        
    
        # Spinning up the model without any temperature changes
        if not silent: print('Spinning up the model...')
        for i in range(0,ny-1): self.SpinupModel.step_forward(i, 0.0, 0.0, 0.0)
        if not silent: print('   ...finished')
            
        new_inits = self.SpinupModel.getFinalCReservoirs()
            
        # newly initialize the carbon reservoir containers
        for res_i, init_reservoir in enumerate(new_inits):
            self.CarbonModel.reservoirs[res_i][0] = init_reservoir 
            self.CarbonModel.C_total[0] += init_reservoir


    def integrate(self, silent=False):
        
        if not silent: print('Start integrating...')

        for i in range(0,self.nyears-1):
            # Calculate the radiative forcing
            self.__calcERF_FAIR(i)
            
            # Calculate the Carbon cycle component for one time step
            self.CarbonModel.step_forward(i, self.T_surf[i], self.T_TC[i], self.T_D[i])

            # Update atmospheric CO2 in energy model with that from carbon model
            self.co2[i+1] = self.CarbonModel.pCO2_atm[i+1]

            # Calculate the energy exchange within ocean
            self.__updateOceanModel(i)
            
        if not silent: print('  ... finished')



    def __calcERF_FAIR(self, i):
        """ This calculates the effective radiative forcing as in FAIR."""
        f1_co2 = 4.57
        f3_co2 = 0.086
        scale_co2 = 1.0202
        co2_topo_adj = 0.05
        
        self.ERF_co2[i] = scale_co2 * (f1_co2 * np.log(self.co2[i] / self.pi_co2) + \
                                       (np.sqrt(f3_co2**2 * self.co2[i]) - np.sqrt(f3_co2**2 * self.pi_co2)) * \
                                       (1.0 + co2_topo_adj))
        self.ERF[i] = self.ERF_co2[i] + self.ERF_nonco2[i]


    def __updateOceanModel(self, i):

        """Energy exchange within the ocean.
        This is not consistent with the carbon transport within the ocean!!!
        Specifically, there is only one box for the surface ocean instead of one for the high latitudes
        and one for the low latitudes."""
        
        self.T_ML[i+1] = self.T_ML[i] + self.dts * ((self.ERF[i] - self.alpha*self.T_surf[i]) / self.C_ML - \
                                                   (self.w_E/self.Depths[0]) * (self.T_ML[i] - self.T_TC[i]))
        
        self.T_TC[i+1] = self.T_TC[i] + self.dts * ((self.w_E/self.Depths[1]) * (self.T_ML[i] - self.T_TC[i]) - \
                                                    (self.w_D/self.Depths[1]) * (self.T_TC[i] - self.T_D[i]))
        
        self.T_D[i+1] = self.T_D[i] + self.dts * ((self.w_D/self.Depths[2]) * (self.T_TC[i] - self.T_D[i]))
                
        self.T_surf[i+1] = self.beta * self.T_ML[i+1]




    def getCO2(self):       return self.co2
    def getTsurf(self):     return self.T_surf
    def getTml(self):       return self.T_ML
    def getTtc(self):       return self.T_TC
    def getTd(self):        return self.T_D
    def getERF(self):       return self.ERF
    def getERFCO2(self):    return self.ERF_co2
    def getERFnonCO2(self): return self.ERF_nonco2
    def getCumEmis(self):   return self.F
    def getEmis(self):      return self.emission
    def getTime(self):      return self.time
    
    ##### Loading informations from the carbon model
    def getPhotosynthesis(self):      return self.CarbonModel.photosynthesis
    def getRespiration(self):         return self.CarbonModel.respiration
    def getSoilRespiration(self):     return self.CarbonModel.soil_respiration
    def getTurnover(self):            return self.CarbonModel.turnover
    def getOceanFluxWarm(self):       return self.CarbonModel.ocean_flux_warm
    def getOceanFluxCold(self):       return self.CarbonModel.ocean_flux_cold
    def getLowLatitudeMixing(self):   return self.CarbonModel.k_wi
    def getHighLatitudeMixing(self):  return self.CarbonModel.k_cd
    def getOverturning(self):         return self.CarbonModel.k_T
    def getAirSeaExchange(self):      return self.CarbonModel.ocean_flux_total
    def getAirLandExchange(self):     return self.CarbonModel.land_flux_total
    def getCReservoirAtm(self):       return self.CarbonModel.C_atm
    def getCReservoirVeg(self):       return self.CarbonModel.C_veg
    def getCReservoirSoil(self):      return self.CarbonModel.C_soil
    def getCReservoirOceanWarm(self): return self.CarbonModel.C_warm_surf
    def getCReservoirOceanCold(self): return self.CarbonModel.C_cold_surf
    def getCReservoirOceanInt(self):  return self.CarbonModel.C_int_wat
    def getCReservoirOceanDeep(self): return self.CarbonModel.C_deep_oce
    def getCConcOceanWarm(self):      return self.CarbonModel.C_warm_surf / self.CarbonModel.V_warm
    def getCConcOceanCold(self):      return self.CarbonModel.C_cold_surf / self.CarbonModel.V_cold
    def getCConcOceanInt(self):       return self.CarbonModel.C_int_wat / self.CarbonModel.V_int
    def getCConcOceanDeep(self):      return self.CarbonModel.C_deep_oce / self.CarbonModel.V_deep
    def getOceanPco2Warm(self):       return self.CarbonModel.pCO2_oce_warm
    def getOceanPco2Cold(self):       return self.CarbonModel.pCO2_oce_cold
    def getLandCarbon(self):          return self.CarbonModel.C_veg + self.CarbonModel.C_soil
    def getOceanCarbon(self):         return self.CarbonModel.C_warm_surf + self.CarbonModel.C_cold_surf \
                                             + self.CarbonModel.C_int_wat + self.CarbonModel.C_deep_oce








    class CarbonBoxModel:
        """A class of a carbon cycle model orginially based on that from Lenton (2000).
        The model consists of 7 distinct C reservoirs (4 oce, 2 land, 1 atm).
    
        This is not the normal Lenton Model. I used the air-sea gas exchange
        from LOSCAR v2.0.4 (Zeebe, 2012, GMD). The carbon chemistry calcualtions
        are done as in HAMOCC. The model uses input from MPI-ESM-LR ensemble
        simulation for surface values of PO4, Si and TAlk.
    
        The advection scheme was adapted to that of LOSCAR as default.
    
        A component for the biological carbon pump was added, based on values from MPI-ESM."""
    
        #### Model parameters and initial values
        #### --> Values from Lenton (2020)
    
        ## constants
        earth_surface_area = 5.101e14 # m^2
        ocean_surface_area = 3.6e14   # m^2
        ocean_volume = 1.36e18        # m^3
        mole_volume_atm = 1.773e20    # moles
        area_vegetated = 1.33e14      # m^2
        gas_constant = 8.314          # J K^-1 mol^-1
        epsilon = 0.47                # ppmv GtC^-1
        temp_PI = 286.75              # K
        rho_water = 1027.0            # Kg m^-3
        mol_to_GT_C = 12.0107e-15     # GT C (mol C)^-1
        pi_co2  = 283.0

        #### ocean conditions (from MPI-ESM)
    
        ## Relating temperature in the surface layers to global mean T anomaly from heat model
        ## Default values from MPIESM
    
        # pot. Temperature
        T0_warm_surface      = 18.64
        T_warm_surface_Tsdep = 0.61
    
        T0_cold_surface      = 1.01
        T_cold_surface_Tsdep = 0.41
    
        # Salinity
        S0_warm_surface      = 34.774
        S_warm_surface_Tsdep = -0.028
    
        S0_cold_surface      = 33.498
        S_cold_surface_Tsdep = -0.146
    
        # ocean biogeochemistry conditions (from MPI-ESM)
        po4_warm = 0.00052  # mol m^-3
        po4_cold = 0.00122  # mol m^-3
        si_warm = 0.0112    # mol m^-3
        si_cold = 0.0535    # mol m^-3
        talk_warm = 2.285   # mol m^-3
        talk_cold = 2.236   # mol m^-3
        
        po4_warm_Tdep   = -0.00003 # mol m^-3 K^-1
        po4_cold_Tdep   = -0.00003 # mol m^-3 K^-1
        si_warm_Tdep    = -0.00032 # mol m^-3 K^-1
        si_cold_Tdep    = -0.00129 # mol m^-3 K^-1
        talk_warm_Tdep  = -0.00230 # mol m^-3 K^-1
        talk_cold_Tdep  = -0.01233 # mol m^-3 K^-1
    
        # Tuning parameters have default values from Lenton (2000)
        def __init__(self, co2_emissions, sy=1850, ey=2300, dt=1.0, dbg=0, lfix_co2=False, co2fix=283.0,
                     initial_C_reservoirs = [596.0, 550.0, 1500.0, 730.0, 140.0, 10040.0, 26830.0],
                     depths=[100, 100, 1000], advection='LOSCAR', isSpinup=False):
        
            self.emission   = co2_emissions # emissions should enter in GT C per year
        
            self.dbg=dbg # 0: no debug output
                         # 1: C mass check output
                         # 2: print out fluxes
                         # 3: print out mixing coefficients
        
            # Timestep:
            self.startyear = sy
            self.endyear = ey
            self.dt = dt
            self.dts = self.dt * 365.25 * 24 * 3600.
            self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        
            # run with constant atm. CO2?
            self.co2_is_fixed = lfix_co2
            self.fixed_co2_value = co2fix
        
            self.advection = advection
            self.isSpinup = isSpinup
        
            ####################################################################################
            ############          Tune these parameters          ###############################
            ####################################################################################
                
            ### Land
            self.C_veg_steady        = 550.0 # steady state reservoir size in GtC (Lenton default: 550)
            self.K_halfsat           = 120.0 # ppmv (Lenton default: 120 (145) ppmv)
            self.k_turn              = 0.092 # yr^-1 (Lenton default: 0.092)
            # rarely tuned:
            self.k_photo             = 0.184 # yr^-1
            self.k_resp              = 0.092 # yr^-1
            self. k_sresp            = 0.0337 # yr^-1
            self.k_c                 = 29 # ppmv
            self.E_a                 = 54830.0 # J mol^-1
            self.k_mm                = 1.478
            self.k_A                 = 8.7039e9
            self.k_B                 = 157.072
                    
            ###  Ocean
            ## advection rates for Lenton advection
            self.k_THC_over          = 6.5e7   # Lenton default:  6.5e7 m^3 s^-1
            self.k_hl_over           = 4.87e7  # Lenton detault:  4.87e7 m^3 s^-1
            self.k_warm_int_exchange = 1.25e7  # Lenton default:  1.25e7 m^3 s^-1
            self.k_int_deep_exchange = 20.0e7  # Lenton default: 20.0e7 m^3 s^-1
            
            ## advection rates for LOSCAR-like advection
            # values are defaults from LOSCAR (Zeebe, 2012)
            self.k_mix_warm_int = 6.3e7
            self.k_mix_cold_deep = 1.7e7
            self.k_conveyor = 2.0e7
            
            # Overturning circulation dependent on Warming?
            self.include_Tdep_advection = False
            self.advection_Tdep_frac = -0.089 
            
            ## gas exchange rate
            self.k_gasex = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1
        
            ## biological carbon pump
            self.include_bio_pump = True
            self.bio_pump_warm = 0.6  # GtC yr^-1; Carbon export in low latitudes;
            self.bio_pump_cold = 6.9  # GtC yr^-1; Carbon export in high latitudes;
                                      # default value (6.9) is fitted to MPIESM data
            self.bio_pump_cold_T_dep = -0.3 # GtC yr^-1 K^-1; Temperature dependence of carbon export;
                                            # default value (-0.3) is fitted to MPIESM data
            self.bio_pump_cold_eff = 0.2 # Transfer efficiency in high latitudes
    
            # Include T-dependence of ocean surface salinity
            self.include_salinity_Tdep = True
   
            # Include T-dependence of ocean surface nutrients
            self.include_nutrient_Tdep = False
        
            # Prescribe land or ocean carbon fluxes during the simulation period?
            self.prescribe_C_flux_land = False        
            self.prescribe_C_flux_ocean = False
            self.prescribed_C_flux_land = np.zeros((self.nyears))
            self.prescribed_C_flux_ocean = np.zeros((self.nyears))

            ####################################################################################
            ####################################################################################
            ####################################################################################
        
            # model settings
            self.warm_area_fraction = 0.85
            self.cold_area_fraction = 1.0 - self.warm_area_fraction
            self.d_surface_warm = depths[0]  # m
            self.d_surface_cold = depths[1]
            self.d_intermediate = depths[2]
            
            self.V_warm = self.d_surface_warm * self.ocean_surface_area * self.warm_area_fraction
            self.V_cold = self.d_surface_cold * self.ocean_surface_area * self.cold_area_fraction
            self.V_int = self.d_intermediate * self.ocean_surface_area
            self.V_deep = self.ocean_volume - self.V_warm - self.V_cold - self.V_int
            

            ### Initialisation of variables

            # reservoir containers
            self.C_atm = np.zeros((self.nyears))
            self.C_veg = np.zeros((self.nyears))
            self.C_soil = np.zeros((self.nyears))
            self.C_warm_surf = np.zeros((self.nyears))
            self.C_cold_surf = np.zeros((self.nyears))
            self.C_int_wat = np.zeros((self.nyears))
            self.C_deep_oce = np.zeros((self.nyears))
            
            self.C_total = np.zeros((self.nyears))
            
            self.reservoirs = [self.C_atm, self.C_veg, self.C_soil, self.C_warm_surf,
                               self.C_cold_surf, self.C_int_wat, self.C_deep_oce]
            
            # initial reservoir values
            self.reservoir_inits = initial_C_reservoirs
            
            # initialize the carbon reservoir containers
            for res_i, init_reservoir in enumerate(self.reservoir_inits):
                self.reservoirs[res_i][0] = init_reservoir 
                self.C_total[0] += init_reservoir
                
            if self.co2_is_fixed:
                self.C_atm[:] = self.fixed_co2_value / self.epsilon
            
            # atmospheric and oceanic pCO2
            self.pCO2_atm      = np.zeros((self.nyears))
            self.pCO2_oce_cold = np.zeros((self.nyears))
            self.pCO2_oce_warm = np.zeros((self.nyears))
            self.__calc_pCO2_atm(0)
            
            # rate containers
            self.photosynthesis = np.zeros((self.nyears))
            self.respiration = np.zeros((self.nyears))
            self.turnover = np.zeros((self.nyears))
            self.soil_respiration = np.zeros((self.nyears))
            self.ocean_flux_warm = np.zeros((self.nyears))
            self.ocean_flux_cold = np.zeros((self.nyears))
            
            self.land_flux_total = np.zeros((self.nyears))
            self.ocean_flux_total = np.zeros((self.nyears))
            

        def update_depths(self,depths):
            #### Call this before the integration!!!
            self.d_surface_warm = depths[0]  # m
            self.d_surface_cold = depths[1]
            self.d_intermediate = depths[2]
            
            self.V_warm = self.d_surface_warm * self.ocean_surface_area * self.warm_area_fraction
            self.V_cold = self.d_surface_cold * self.ocean_surface_area * self.cold_area_fraction
            self.V_int = self.d_intermediate * self.ocean_surface_area
            self.V_deep = self.ocean_volume - self.V_warm - self.V_cold - self.V_int
            

        def modify_initial_reservoirs(self,modifications):
            # modify the carbon reservoir containers
            for res_i, modification in enumerate(modifications):
                self.reservoirs[res_i][0] += modification
                self.C_total[0] += modification
            
            
        def step_forward(self, i, T_surf, T_thermo, T_deep):

            if self.dbg == 1:
                print()
                print()
                self.__calc_total_carbon(i)
                print('   dbg ',i,', atm pCO2 before fluxes:', self.pCO2_atm[i])   
                print('   dbg ',i,', total carbon before fluxes:', self.C_total[i])            
            
            ######################################################
            ############ Land component ##########################
            ######################################################
        
            ##### Calculate the fluxes between the reservoirs
            if self.prescribe_C_flux_land:
                self.land_flux_total[i] = self.prescribed_C_flux_land[i]
            else:
                self.__calc_photosynthesis(i, T_surf)
                self.__calc_respiration(i, T_surf)
                self.__calc_turnover(i)
                self.__calc_soil_respiration(i, T_surf)
            
                self.land_flux_total[i] = self.photosynthesis[i] - self.respiration[i] - self.soil_respiration[i]
    
                ##### Updating reservoirs due to vegetation
                self.C_veg[i+1]  = self.C_veg[i]  + self.dt * (self.photosynthesis[i] - self.respiration[i] - self.turnover[i])
                self.C_soil[i+1] = self.C_soil[i] + self.dt * (self.turnover[i] - self.soil_respiration[i])
        
            # Updating atmospheric CO2
            if not self.co2_is_fixed:
                self.C_atm[i+1]  = self.C_atm[i]  - self.dt * (self.land_flux_total[i])
        
            # Debug output
            if self.dbg == 1:
                tmp_total_C = self.C_atm[i+1] + self.C_veg[i+1] + self.C_soil[i+1] \
                                + self.C_warm_surf[i] + self.C_cold_surf[i] \
                                + self.C_int_wat[i] + self.C_deep_oce[i]
                print('   dbg ',i,', total carbon after vegetation fluxes:', tmp_total_C)  
        
        
            ######################################################
            ############ Adding emissions ########################
            ######################################################
            
            if not self.co2_is_fixed:
                self.C_atm[i+1] = self.C_atm[i+1] + self.dt * self.emission[i]

            

            if self.dbg == 2 and not self.prescribe_C_flux_land:
                print()
                print()
                print('   dbg ',i,', old C in vegetation: ',  self.C_veg[i])
                print('   dbg ',i,', old C in soil      : ',  self.C_soil[i])
                print('   dbg ',i,', old C in atmosphere: ',  self.C_atm[i])
                print()
                print('   dbg ',i,', photosynthesis: ', - self.photosynthesis[i])
                print('   dbg ',i,', respiration: ', self.respiration[i])
                print('   dbg ',i,', soil_respiration: ', self.soil_respiration[i])
                print('   dbg ',i,', total land flux: ',  self.land_flux_total[i])
                print()
                print('   dbg ',i,', new C in vegetation: ',  self.C_veg[i+1])
                print('   dbg ',i,', new C in soil      : ',  self.C_soil[i+1])
                print('   dbg ',i,', new C in atmosphere: ',  self.C_atm[i+1])


            self.__calc_pCO2_atm(i+1)
        
            ######################################################
            ############ Ocean component #########################
            ######################################################        
        
            # Nothing to do, if ocean carbon flux is prescribed
            if self.prescribe_C_flux_ocean:
                self.ocean_flux_total[i] = self.prescribed_C_flux_ocean[i]
                self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]

                if self.dbg == 2:
                    print()
                    print()
                    print('   dbg ',i,', total ocean flux: ',  self.ocean_flux_total[i])
                    print('   dbg ',i,', new C in atmosphere: ',  self.C_atm[i+1])
                return
            
        
            self.__calc_surface_ocean_flux(i, T_surf)
            self.ocean_flux_total[i] = self.ocean_flux_warm[i] + self.ocean_flux_cold[i]
    
            ##### Updating reservoirs due to ocean-flux
            self.C_warm_surf[i] = self.C_warm_surf[i] + self.dt * self.ocean_flux_warm[i]
            self.C_cold_surf[i] = self.C_cold_surf[i] + self.dt * self.ocean_flux_cold[i]
            if not self.co2_is_fixed:
                self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
        
            self.__calc_pCO2_atm(i+1)
            
            if self.dbg == 1:
                tmp_total_C = self.C_atm[i+1] + self.C_veg[i+1] + self.C_soil[i+1] \
                                + self.C_warm_surf[i] + self.C_cold_surf[i] \
                                + self.C_int_wat[i] + self.C_deep_oce[i]
                print('   dbg ',i,', total carbon after ocean fluxes:', tmp_total_C)  
            
            
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            if self.include_bio_pump: self.__biological_carbon_pump(i, T_surf)
            
            ##### Advect the oceanic carbon
            if self.advection=='Lenton':
                tmp_C_warm, tmp_C_cold, tmp_C_int, tmp_C_deep = self.__advect_ocean_tracer(i, T_surf, 
                                                                                self.C_warm_surf[i] / self.V_warm,
                                                                                self.C_cold_surf[i] / self.V_cold,
                                                                                self.C_int_wat[i]   / self.V_int,
                                                                                self.C_deep_oce[i]  / self.V_deep)

            elif self.advection=='LOSCAR':
                tmp_C_warm, tmp_C_cold, tmp_C_int, tmp_C_deep = self.__advect_ocean_tracer_LOSCAR(i, T_surf,
                                                                                self.C_warm_surf[i] / self.V_warm,
                                                                                self.C_cold_surf[i] / self.V_cold,
                                                                                self.C_int_wat[i]   / self.V_int,
                                                                                self.C_deep_oce[i]  / self.V_deep)
            else:
                sys.exit('No correct choice of advection model! Can be "Lenton" or "LOSCAR", current value: '+self.advection)
        

            ## Final update of the tracer concentrations
            self.C_warm_surf[i+1] = tmp_C_warm * self.V_warm
            self.C_cold_surf[i+1] = tmp_C_cold * self.V_cold
            self.C_int_wat[i+1]   = tmp_C_int  * self.V_int
            self.C_deep_oce[i+1]  = tmp_C_deep * self.V_deep

            self.__calc_total_carbon(i+1)
        
            # Debugging output
            if self.dbg == 1:
                tmp_total_atm = self.C_atm[i+1] + self.C_veg[i+1] + self.C_soil[i+1]
                tmp_total_oce_before = self.C_warm_surf[i] + self.C_cold_surf[i] + self.C_int_wat[i] + self.C_deep_oce[i]
                tmp_total_oce_after = self.C_warm_surf[i+1] + self.C_cold_surf[i+1] + self.C_int_wat[i+1] + self.C_deep_oce[i+1]
            
                print('   dbg ',i,', total carbon after advection', self.C_total[i+1])            
                print('   dbg ',i,', ocean carbon before advection', tmp_total_oce_before)            
                print('   dbg ',i,', ocean carbon after advection', tmp_total_oce_after)            
            
        
            if self.dbg == 2:
                print()
                print()
                print('   dbg ',i,', atm pCO2         : ', self.pCO2_atm[i])
                print('   dbg ',i,', ocean pCO2 (cold): ', self.pCO2_oce_cold[i])
                print('   dbg ',i,', ocean pCO2 (warm): ', self.pCO2_oce_warm[i])
                print('   dbg ',i,', ocean flux (cold): ', self.ocean_flux_cold[i])
                print('   dbg ',i,', ocean flux (warm): ', self.ocean_flux_warm[i])
                print()
                print('   dbg ',i,', total ocean flux: ', self.ocean_flux_total[i])
                print('   dbg ',i,', atm C inv. after fluxes:', self.C_atm[i+1])

        
        def __calc_total_carbon(self,i):
            self.C_total[i] = self.C_atm[i] + self.C_veg[i] + self.C_soil[i] + self.C_warm_surf[i] + self.C_cold_surf[i] \
                                + self.C_int_wat[i] + self.C_deep_oce[i]

        def __calc_pCO2_atm(self,i):
            self.pCO2_atm[i] = self.C_atm[i] * self.epsilon
        
        def __calc_respiration(self,i,Ts):
            self.respiration[i] = self.k_resp  * self.C_veg[i] * self.k_A \
                                  * np.exp(-self.E_a / (self.gas_constant * (Ts+self.temp_PI)))
    
        def __calc_photosynthesis(self,i, Ts):
            self.photosynthesis[i] = self.k_photo * self.C_veg_steady * self.k_mm * (self.pCO2_atm[i] - self.k_c) \
                                   / (self.K_halfsat + self.pCO2_atm[i] - self.k_c) \
                                   * ((15.0 + Ts)**2 * (25.0 - Ts)) / 5625.0
                
        def __calc_turnover(self,i):
            self.turnover[i] = self.k_turn * self.C_veg[i]
        
        def __calc_soil_respiration(self, i, Ts):
            self.soil_respiration[i] = self.k_sresp * self.C_soil[i] * self.k_B \
                                       * np.exp(-308.56/(Ts + self.temp_PI - 227.13))
        
        def __calc_surface_ocean_flux(self, i, Ts):
            # tropical and high latitude oceans should warm differently under global warming !
            T_warm = self.T_warm_surface_Tsdep * Ts + self.T0_warm_surface
            T_cold = self.T_cold_surface_Tsdep * Ts + self.T0_cold_surface
        
            dic_w = self.C_warm_surf[i] / self.mol_to_GT_C / self.V_warm
            dic_c = self.C_cold_surf[i] / self.mol_to_GT_C / self.V_cold

            if self.include_salinity_Tdep:
                salt_w = self.S0_warm_surface + self.S_warm_surface_Tsdep * Ts
                salt_c = self.S0_cold_surface + self.S_cold_surface_Tsdep * Ts
            else:
                salt_w = self.S0_warm_surface
                salt_c = self.S0_cold_surface
        
            if self.include_nutrient_Tdep:
                po4_w  = self.po4_warm  + self.po4_warm_Tdep  * Ts
                po4_c  = self.po4_cold  + self.po4_cold_Tdep  * Ts
                si_w   = self.si_warm   + self.si_warm_Tdep   * Ts
                si_c   = self.si_cold   + self.si_cold_Tdep   * Ts
                talk_w = self.talk_warm + self.talk_warm_Tdep * Ts
                talk_c = self.talk_cold + self.talk_cold_Tdep * Ts
            else:
                po4_w  = self.po4_warm 
                po4_c  = self.po4_cold
                si_w   = self.si_warm
                si_c   = self.si_cold
                talk_w = self.talk_warm
                talk_c = self.talk_cold

            self.pCO2_oce_warm[i] = self.__calc_ocean_pCO2(T_warm, salt_w, talk_w, dic_w, po4_w, si_w)
            self.pCO2_oce_cold[i] = self.__calc_ocean_pCO2(T_cold, salt_c, talk_c, dic_c, po4_c, si_c)
            
            self.ocean_flux_warm[i] = self.k_gasex * self.mol_to_GT_C \
                                     * self.ocean_surface_area * self.warm_area_fraction \
                                     * (self.pCO2_atm[i] - self.pCO2_oce_warm[i])
            
            self.ocean_flux_cold[i] = self.k_gasex * self.mol_to_GT_C \
                                     * self.ocean_surface_area * self.cold_area_fraction \
                                     * (self.pCO2_atm[i] - self.pCO2_oce_cold[i])
        
    
        def __calc_ocean_pCO2(self, TC, S, alk, dic, po4, sil):
            # carchm routine does calculations using kmol m^-3
            alk*=1.0e-3
            dic*=1.0e-3
            po4*=1.0e-3
            sil*=1.0e-3
        
            all_aks = carchm.calc_diss_const(np.array([0.0]), TC, S)
        
            aks = all_aks[0]
            akf = all_aks[1]
            ak1p = all_aks[2]
            ak2p = all_aks[3]
            ak3p = all_aks[4]
            aksi = all_aks[5]
            ak1 = all_aks[6]
            ak2 = all_aks[7]
            akb = all_aks[8]
            akw = all_aks[9]
            aksp = all_aks[10]
        
            hi = carchm.update_hi(dic,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,po4,alk,dim=0, silent=True)
            solco2 = carchm.get_solco2(TC,S)
            
            return carchm.get_pco2(dic, hi, ak1, ak2, solco2)


        def __biological_carbon_pump(self, i, T_surf):
            C_exp_warm = self.bio_pump_warm
            C_exp_cold = self.bio_pump_cold + self.bio_pump_cold_T_dep * T_surf
            C_inp_int  = C_exp_warm + C_exp_cold * (1.0 - self.bio_pump_cold_eff)
            C_inp_deep = C_exp_cold * self.bio_pump_cold_eff
                
            if self.dbg == 1:            
                tmp_total_oce_before = self.C_warm_surf[i] + self.C_cold_surf[i] + self.C_int_wat[i] + self.C_deep_oce[i]

            self.C_warm_surf[i] = self.C_warm_surf[i] - self.dt * C_exp_warm
            self.C_cold_surf[i] = self.C_cold_surf[i] - self.dt * C_exp_cold
            self.C_int_wat[i]   = self.C_int_wat[i]   + self.dt * C_inp_int
            self.C_deep_oce[i]  = self.C_deep_oce[i]  + self.dt * C_inp_deep
            
            if self.dbg == 1:            
                tmp_total_oce_after = self.C_warm_surf[i] + self.C_cold_surf[i] + self.C_int_wat[i] + self.C_deep_oce[i]
                
                print('   dbg ',i,', ocean carbon before bio pump', tmp_total_oce_before)            
                print('   dbg ',i,', ocean carbon after bio pump ', tmp_total_oce_after)

            return
    
        def __advect_ocean_tracer(self, i, Ts, C_w, C_c, C_i, C_d):
            ### assumes a list of tracer concentrations is given in order:
            ## C_warm, C_cold, C_int, C_deep 
        
            if self.include_Tdep_advection:
                k_T  = self.k_THC_over + self.advection_Tdep_frac * self.k_THC_over * Ts
                k_U  = self.k_hl_over  + self.advection_Tdep_frac * self.k_hl_over  * Ts
            else:
                k_T  = self.k_THC_over
                k_U  = self.k_hl_over
            k_O = k_T + k_U
            k_wi = self.k_warm_int_exchange
            k_id = self.k_int_deep_exchange
        
            Cw = C_w + (self.dts / self.V_warm) * ( (k_T+k_wi)*C_i - (k_T+k_wi)*C_w )
            Cc = C_c + (self.dts / self.V_cold) * ( k_T*C_w + k_U*C_i - k_O*C_c )
            Ci = C_i + (self.dts / self.V_int) * ( (k_O+k_id)*C_d + k_wi*C_w - (k_O + k_wi + k_id)*C_i )
            Cd = C_d + (self.dts / self.V_deep) * ( k_O*C_c + k_id*C_i - (k_O + k_id)*C_d )
            
            return Cw, Cc, Ci, Cd
    
    
        def __advect_ocean_tracer_LOSCAR(self, i, Ts, C_w, C_c, C_i, C_d):
            ### assumes a list of tracer concentrations is given in order:
            ## C_warm, C_cold, C_int, C_deep
            
            k_T = self.k_conveyor
            if self.include_Tdep_advection: k_T  += self.advection_Tdep_frac * self.k_conveyor * Ts

            k_wi = self.k_mix_warm_int
            k_cd = self.k_mix_cold_deep
            
            Cw = C_w + (self.dts / self.V_warm) * ( k_wi * (C_i - C_w) )
            Cc = C_c + (self.dts / self.V_cold) * ( k_T * (C_i - C_c) + k_cd * (C_d - C_c) )
            Ci = C_i + (self.dts / self.V_int)  * ( k_T * (C_d - C_i) - k_wi * (C_i - C_w) )
            Cd = C_d + (self.dts / self.V_deep) * ( k_T * (C_c - C_d) - k_cd * (C_d - C_c) )
        
            return Cw, Cc, Ci, Cd

        def getInitialCReservoirs(self):  return [self.C_atm[0], self.C_veg[0], self.C_soil[0], self.C_warm_surf[0], self.C_cold_surf[0], self.C_int_wat[0], self.C_deep_oce[0]]
        def getFinalCReservoirs(self):    return [self.C_atm[-1], self.C_veg[-1], self.C_soil[-1], self.C_warm_surf[-1], self.C_cold_surf[-1], self.C_int_wat[-1], self.C_deep_oce[-1]]
    
