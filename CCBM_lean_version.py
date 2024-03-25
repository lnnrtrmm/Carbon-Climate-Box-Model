import numpy as np
import sys

class EnergyAndCarbonBoxModel:
    """This is a box model for energy and Carbon in the Earth system.
    The energy component is that of Li et al.: 'Optimal temperature overshoot ...' (2020).
    The radiative forcing is calculated as in FaiR (Leach, 2020; Smith, 2018).
    The carbon box model is a combination of different existing carbon cycle models
    (Lenton (2000), LOSCAR (Zeebe, 2012)).
    Radiative forcing is read in from CMIP6 data.
    
    This is a "lean" version, where some old options and debugging output etc. are removed.

    """

    def __init__(self, co2_emissions, ERF_nonco2, sy=1850, ey=2300, dt=1.0,
                  lfix_co2=False, co2fix=283.0, depths=[50,500,3150],
                  initial_C_reservoirs=[596.0, 550.0, 1500.0, 730.0, 140.0, 10040.0, 26830.0]):
        
        # Timestep:
        self.startyear = sy
        self.endyear = ey
        self.dt = dt                                # in years
        self.dts = self.dt * 365.25 * 24 * 3600.    # in seconds
        self.nyears = int((self.endyear - self.startyear) / self.dt) + 1
        self.time = np.linspace(self.startyear, self.endyear, self.nyears)
        
        self.Depths = depths
        
        #### Model parameters
        # Energy model 
        self.alpha = 1.16 # W m^-2 K^-1
        self.w_E = 1.0e-6 # m s^-1
        self.w_D = 0.4e-6 # m s^-1
        self.C_ML = 2.0e8 # W s m^-2 K^-1 
        self.beta = 1.33  # dmnl

        
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
                                          dt=self.dt, lfix_co2=lfix_co2, co2fix=co2fix,
                                          initial_C_reservoirs=initial_C_reservoirs,
                                          depths=[300, 250, 600])    # These depths give approximately the best fit with MPIESM


    def spinup(self, co2fix=283.0, years=3000, silent=False):

        ny = int(years/self.dt)+1
        zero_emissions = np.zeros(ny)
        
        # initiating a spinup model
        self.SpinupModel = self.CarbonBoxModel(zero_emissions, sy=0, ey=years, dt=self.dt,
                                               lfix_co2=True, co2fix=co2fix, advection=self.advection, isSpinup=True,
                                               depths=[self.CarbonModel.d_surface_warm, self.CarbonModel.d_surface_cold,
                                               self.CarbonModel.d_intermediate], initial_C_reservoirs = self.CarbonModel.reservoir_inits)

        
        # change tuning parameters to be as in the Carbon Model        
        self.SpinupModel.C_veg_steady        = self.CarbonModel.C_veg_steady          # 2-box land component from Lenton (2000)
        self.SpinupModel.k_turn              = self.CarbonModel.k_turn                # 2-box land component from Lenton (2000)
        self.SpinupModel.K_halfsat           = self.CarbonModel.K_halfsat             # 2-box land component from Lenton (2000)

        self.SpinupModel.k_conveyor          = self.CarbonModel.k_conveyor            # Advection as in LOSCAR (Zeebe, 2012)
        self.SpinupModel.k_mix_cold_deep     = self.CarbonModel.k_mix_cold_deep       # Advection as in LOSCAR (Zeebe, 2012)
        self.SpinupModel.k_mix_warm_int      = self.CarbonModel.k_mix_warm_int        # Advection as in LOSCAR (Zeebe, 2012)

        self.SpinupModel.k_gasex             = self.CarbonModel.k_gasex
        self.SpinupModel.bio_pump_warm       = self.CarbonModel.bio_pump_warm
        self.SpinupModel.bio_pump_cold       = self.CarbonModel.bio_pump_cold
        self.SpinupModel.bio_pump_cold_eff   = self.CarbonModel.bio_pump_cold_eff

        self.SpinupModel.pCO2_iterative_limit = self.CarbonModel.pCO2_iterative_limit
        
    
        # Spinning up the model without any temperature changes
        for i in range(0,ny-1): self.SpinupModel.step_forward(i, 0.0, 0.0, 0.0)
            
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
    def getOceanPHWarm(self):         return self.CarbonModel.pH_oce_warm
    def getOceanPHCold(self):         return self.CarbonModel.pH_oce_cold
    def getOceanPCO2Warm(self):       return self.CarbonModel.pCO2_oce_warm
    def getOceanPCO2Cold(self):       return self.CarbonModel.pCO2_oce_cold
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
        talk_warm = 2.285   # mol m^-3
        talk_cold = 2.236   # mol m^-3
        
        talk_warm_Tdep  = -0.00230 # mol m^-3 K^-1
        talk_cold_Tdep  = -0.01233 # mol m^-3 K^-1

	# standard pH values from testing
        pH_oce_cold_init = 8.36
        pH_oce_warm_init = 8.14
    
        # Tuning parameters have default values from Lenton (2000)
        def __init__(self, co2_emissions, sy=1850, ey=2300, dt=1.0, lfix_co2=False, co2fix=283.0,
                     initial_C_reservoirs = [596.0, 550.0, 1500.0, 730.0, 140.0, 10040.0, 26830.0],
                     depths=[100, 100, 1000], isSpinup=False):
        
            self.emission   = co2_emissions # emissions should enter in GT C per year
        
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

            
            ## advection rates for LOSCAR-like advection
            # values are defaults from LOSCAR (Zeebe, 2012)
            self.k_mix_warm_int = 6.3e7
            self.k_mix_cold_deep = 1.7e7
            self.k_conveyor = 2.0e7
            
            # Overturning circulation dependent on Warming?
            self.advection_Tdep_frac = 0   # Set to 0 in default setup. MPIESM value: -0.089 
            
            ## gas exchange rate
            self.k_gasex = 0.06   # LOSCAR default: 0.06 mol (uatm m^2 yr)^-1
        
            ## biological carbon pump
            self.bio_pump_warm = 0.6  # GtC yr^-1; Carbon export in low latitudes;
            self.bio_pump_cold = 6.9  # GtC yr^-1; Carbon export in high latitudes;
                                      # default value (6.9) is fitted to MPIESM data
            self.bio_pump_cold_T_dep = -0.3 # GtC yr^-1 K^-1; Temperature dependence of carbon export;
                                            # default value (-0.3) is fitted to MPIESM data
            self.bio_pump_cold_eff = 0.2 # Transfer efficiency in high latitudes
        
            # Prescribe land or ocean carbon fluxes during the simulation period?
            self.prescribe_C_flux_land = False        
            self.prescribe_C_flux_ocean = False
            self.prescribed_C_flux_land = np.zeros((self.nyears))
            self.prescribed_C_flux_ocean = np.zeros((self.nyears))

            # Threshold for difference in H+ concentration during iterative calculation of pCO2:
            self.pCO2_iterative_limit = 1e-12

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
            self.pH_oce_cold = np.zeros((self.nyears)) + self.pH_oce_cold_init
            self.pH_oce_warm = np.zeros((self.nyears)) + self.pH_oce_warm_init

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
            #### Call before the integration if this shall be changed!
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
        
            ######################################################
            ############ Adding emissions ########################
            ######################################################
            
            if not self.co2_is_fixed:
                self.C_atm[i+1] = self.C_atm[i+1] + self.dt * self.emission[i]


            self.__calc_pCO2_atm(i+1)
        
            ######################################################
            ############ Ocean component #########################
            ######################################################        
        
            # Nothing to do, if ocean carbon flux is prescribed
            if self.prescribe_C_flux_ocean:
                self.ocean_flux_total[i] = self.prescribed_C_flux_ocean[i]
                self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
        
            self.__calc_surface_ocean_flux(i, T_surf)
            self.ocean_flux_total[i] = self.ocean_flux_warm[i] + self.ocean_flux_cold[i]
    
            ##### Updating reservoirs due to ocean-flux
            self.C_warm_surf[i] = self.C_warm_surf[i] + self.dt * self.ocean_flux_warm[i]
            self.C_cold_surf[i] = self.C_cold_surf[i] + self.dt * self.ocean_flux_cold[i]
            if not self.co2_is_fixed:
                self.C_atm[i+1] = self.C_atm[i+1] - self.dt * self.ocean_flux_total[i]
        
            self.__calc_pCO2_atm(i+1)          
            
            ########### ocean internal restructuring ###########
                
            #### Biological carbon pump
            self.__biological_carbon_pump(i, T_surf)
            
            tmp_C_warm, tmp_C_cold, tmp_C_int, tmp_C_deep = self.__advect_ocean_tracer(i, T_surf,
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

            salt_w = self.S0_warm_surface + self.S_warm_surface_Tsdep * Ts
            salt_c = self.S0_cold_surface + self.S_cold_surface_Tsdep * Ts
            talk_w = self.talk_warm + self.talk_warm_Tdep * Ts
            talk_c = self.talk_cold + self.talk_cold_Tdep * Ts


            self.pH_oce_warm[i], self.pCO2_oce_warm[i], it = self.__calc_ocean_pCO2(T_warm, salt_w, talk_w, dic_w, self.pH_oce_warm[i-1])
            self.pH_oce_cold[i], self.pCO2_oce_cold[i], it = self.__calc_ocean_pCO2(T_cold, salt_c, talk_c, dic_c, self.pH_oce_cold[i-1])

            
            self.ocean_flux_warm[i] = self.k_gasex * self.mol_to_GT_C \
                                     * self.ocean_surface_area * self.warm_area_fraction \
                                     * (self.pCO2_atm[i] - self.pCO2_oce_warm[i])
            
            self.ocean_flux_cold[i] = self.k_gasex * self.mol_to_GT_C \
                                     * self.ocean_surface_area * self.cold_area_fraction \
                                     * (self.pCO2_atm[i] - self.pCO2_oce_cold[i])
        

        def __calc_ocean_pCO2(self, TC, S, alk, dic, ph_old=8.1):
            ''' This is taken from the iLOSCAR model
            (https://github.com/Shihan150/iloscar/tree/main)
            Zeebe (2012), https://doi.org/10.5194/gm/d-5-149-2012
            Li et al. (2024), https://doi.org/10.1016/j.gloplacha.2024.104413
            
            This solution to the carbonate chemistry is based on Follows et al. (2006, https://doi.org/10.1016/j.ocemod.2005.05.004)
            
            The limit can probably be reduced to 1e-10 in order to speed up the calculation a bit more.
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
            C_exp_cold = self.bio_pump_cold + self.bio_pump_cold_T_dep * T_surf
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

        def getInitialCReservoirs(self):  return [self.C_atm[0], self.C_veg[0], self.C_soil[0], self.C_warm_surf[0], self.C_cold_surf[0], self.C_int_wat[0], self.C_deep_oce[0]]
        def getFinalCReservoirs(self):    return [self.C_atm[-1], self.C_veg[-1], self.C_soil[-1], self.C_warm_surf[-1], self.C_cold_surf[-1], self.C_int_wat[-1], self.C_deep_oce[-1]]
    
