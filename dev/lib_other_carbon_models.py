# This includes some other, simpler climate models which are stored here for tests.
# Need to add references and creators of these models!

import numpy as np
import sys

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
