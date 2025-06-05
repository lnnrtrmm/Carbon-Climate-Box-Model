# © Lennart Ramme, Max-Planck-Institute for Meteorology, 2025

import numpy as np
import sys


from constants import seconds_per_year, mol2GtC, ocean_surface_area


'''
This is a collection of functions that describe ocean processes.

For use in a modular simple ocean carbon cycle framework.
'''




###################################################################################
####################### Advection schemes #########################################
###################################################################################


def advect_ocean_tracer_LOSCAR(CModel, i, Tglobal=0.0):
    ### assumes a list of tracer concentrations is given in order:
    ## C_warm, C_cold, C_int, C_deep
    
    k_T = CModel.k_HL_over
    # Temperature dependence of advection
    k_T  += CModel.advection_Tdep_frac * CModel.k_HL_over * Tglobal

    k_wi = CModel.k_mix_WI
    k_cd = CModel.k_mix_CD

    dts = CModel.dt * seconds_per_year

    C_w = CModel.C_oce[0,i] / CModel.V_oce[0]
    C_c = CModel.C_oce[1,i] / CModel.V_oce[1]
    C_i = CModel.C_oce[2,i] / CModel.V_oce[2]
    C_d = CModel.C_oce[3,i] / CModel.V_oce[3]

    CModel.C_oce[0,i+1] = CModel.C_oce[0,i] + dts * ( k_wi * (C_i - C_w) )
    CModel.C_oce[1,i+1] = CModel.C_oce[1,i] + dts * ( k_T * (C_i - C_c) + k_cd * (C_d - C_c) )
    CModel.C_oce[2,i+1] = CModel.C_oce[2,i] + dts * ( k_T * (C_d - C_i) - k_wi * (C_i - C_w) )
    CModel.C_oce[3,i+1] = CModel.C_oce[3,i] + dts * ( k_T * (C_c - C_d) - k_cd * (C_d - C_c) )

    return

def advect_ocean_tracer_Lenton(CModel, i, Tglobal=0.0):
    ### assumes a list of tracer concentrations is given in order:
    ## C_warm, C_cold, C_int, C_deep 
    
    # Adapt overturning circulation depending on temperature?
    k_T  = CModel.k_THC_over + CModel.advection_Tdep_frac * CModel.k_THC_over * Tglobal
    k_U  = CModel.k_HL_over  + CModel.advection_Tdep_frac * CModel.k_HL_over  * Tglobal

    k_O = k_T + k_U
    k_wi = CModel.k_mix_WI
    k_id = CModel.k_mix_ID

    dts = CModel.dt * seconds_per_year

    C_w = CModel.C_oce[0,i] / CModel.V_oce[0]
    C_c = CModel.C_oce[1,i] / CModel.V_oce[1]
    C_i = CModel.C_oce[2,i] / CModel.V_oce[2]
    C_d = CModel.C_oce[3,i] / CModel.V_oce[3]
    
    CModel.C_oce[0,i+1] = CModel.C_oce[0,i] + dts * ( (k_T+k_wi)*C_i            - (k_T+k_wi)*C_w )
    CModel.C_oce[1,i+1] = CModel.C_oce[1,i] + dts * ( k_T*C_w + k_U*C_i         - k_O*C_c )
    CModel.C_oce[2,i+1] = CModel.C_oce[2,i] + dts * ( (k_O+k_id)*C_d + k_wi*C_w - (k_O + k_wi + k_id)*C_i )
    CModel.C_oce[3,i+1] = CModel.C_oce[3,i] + dts * ( k_O*C_c + k_id*C_i        - (k_O + k_id)*C_d )
        
    return


def advect_ocean_tracer_3Box(CModel, i, Tglobal=0.0):
    # 3 boxes are in order: warm surface, cold surface, deep oce
    
    # Adapt overturning circulation depending on temperature?
    k_T  = CModel.k_over + CModel.advection_Tdep_frac * CModel.k_over * Tglobal

    k_wd = CModel.k_mix_WD
    k_cd = CModel.k_mix_CD

    dts = CModel.dt * seconds_per_year

    C_w = CModel.C_oce[0,i] / CModel.V_oce[0]
    C_c = CModel.C_oce[1,i] / CModel.V_oce[1]
    C_d = CModel.C_oce[2,i] / CModel.V_oce[2]
    
    CModel.C_oce[0,i+1] = CModel.C_oce[0,i] + dts * ( (k_T + k_wd)*C_d - (k_T + k_wd)*C_w )
    CModel.C_oce[1,i+1] = CModel.C_oce[1,i] + dts * ( k_T*C_w + k_cd*C_d - (k_T + k_cd)*C_c )
    CModel.C_oce[2,i+1] = CModel.C_oce[2,i] + dts * ( (k_T + k_cd)*C_c + k_wd*C_w - (k_T + k_cd + k_wd)*C_d )
        
    return

###################################################################################
####################### Biological C pump #########################################
###################################################################################

def biological_carbon_pump_4Box(CModel, i, Tglobal=0.0):
    # 4 boxes are in order: warm surface, cold surface, int. oce, deep oce

    dt = CModel.dt
    BCP_w = CModel.bio_pump_warm
    BCP_c = CModel.bio_pump_cold + CModel.bio_pump_cold_Tslope * Tglobal
    transferFrac_w = 0.0
    transferFrac_c = CModel.bio_pump_cold_eff

    C_inp_int  = BCP_w * (1.0 - transferFrac_w) + BCP_c * (1.0 - transferFrac_c)
    C_inp_deep = BCP_w + BCP_c - C_inp_int

    CModel.C_oce[0,i] += - dt * BCP_w
    CModel.C_oce[1,i] += - dt * BCP_c
    CModel.C_oce[2,i] += dt * C_inp_int
    CModel.C_oce[3,i] += dt * C_inp_deep

    return


def biological_carbon_pump_3Box(CModel, i, Tglobal=0.0):
    # 3 boxes are in order: warm surface, cold surface, deep oce

    dt = CModel.dt
    BCP_w = CModel.bio_pump_warm
    BCP_c = CModel.bio_pump_cold + CModel.bio_pump_cold_Tslope * Tglobal

    CModel.C_oce[0,i] += - dt * BCP_w
    CModel.C_oce[1,i] += - dt * BCP_c
    CModel.C_oce[2,i] += dt * (BCP_w + BCP_c)

    return

###################################################################################
####################### Surface CO2-Flux ##########################################
###################################################################################


def calc_ocean_pCO2(TC, S, alk, dic, ph_old=8.1, dbg=0, iterative_limit=1e-10):
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

        change = np.abs(hx-hgss) - iterative_limit
        if change < 0: break
            
    if dbg == 2: print(it, 'steps needed in iterative pH solver')

    co2 = dic/(1+k1/hx + k1*k2/hx/hx)
    pCO2 = co2 / kh * 1.e6
    pH = -np.log10(hx)

    if pH <= 0:
        print('pH got negative! Model settings:')
        sys.exit('Error: negative pH!')
        
    return pH, pCO2



def calc_surface_ocean_flux(CModel, i, Ts, fluxType):
    T_warm = CModel.T_warm_surface_Tslope * Ts + CModel.T0_warm_surface
    T_cold = CModel.T_cold_surface_Tslope * Ts + CModel.T0_cold_surface

    dic_w = CModel.C_oce[0,i] / mol2GtC / CModel.V_oce[0]
    dic_c = CModel.C_oce[1,i] / mol2GtC / CModel.V_oce[1]

    salt_w = CModel.S0_warm_surface + CModel.S_warm_surface_Tslope * Ts
    salt_c = CModel.S0_cold_surface + CModel.S_cold_surface_Tslope * Ts
    talk_w = CModel.Alk0_warm_surface + CModel.Alk_warm_surface_Tslope * Ts
    talk_c = CModel.Alk0_cold_surface + CModel.Alk_cold_surface_Tslope * Ts

    if CModel.dbg == 1:
        print(i, 'calculating surface ocean fluxes')
        print('   T_warm: ', T_warm)
        print('   T_cold: ', T_cold)
        print('   S_warm: ', salt_w,)
        print('   S_cold: ', salt_c)
        print('   talk_warm: ', talk_w)
        print('   talk_cold: ', talk_c)
        print('   dic_warm: ', dic_w)
        print('   dic_warm: ', dic_c)


    
    CModel.pH_oce_warm[i], CModel.pCO2_oce_warm[i] = calc_ocean_pCO2(T_warm, salt_w, talk_w, dic_w, CModel.pH_oce_warm[i-1])
    CModel.pH_oce_cold[i], CModel.pCO2_oce_cold[i] = calc_ocean_pCO2(T_cold, salt_c, talk_c, dic_c, CModel.pH_oce_cold[i-1])

    if CModel.dbg == 1:
        print(i, '....calulcated the ocean pH:', CModel.pH_oce_warm[i], CModel.pH_oce_cold[i])

    
    if fluxType == 'LOSCAR':
        flux_warm = calcAirSeaCO2Flux_LOSCAR(CModel.k_gasex_warm, ocean_surface_area * CModel.warm_area_fraction,
                                               CModel.pCO2_atm[i], CModel.pCO2_oce_warm[i])
        flux_cold = calcAirSeaCO2Flux_LOSCAR(CModel.k_gasex_cold, ocean_surface_area * CModel.cold_area_fraction,
                                               CModel.pCO2_atm[i], CModel.pCO2_oce_cold[i])
    else:
        sys.exit('No other flux type than LOSCAR implemented yet!')

    CModel.ocean_flux_warm[i] = flux_warm    
    CModel.ocean_flux_cold[i] = flux_cold


    if CModel.dbg == 1:
        print(i, '....calulcated the surface fluxes:', CModel.ocean_flux_warm[i], CModel.ocean_flux_cold[i])
        print()
    return



def calcAirSeaCO2Flux_LOSCAR(k, area, pCO2_atm, pCO2_oce):
    '''
    Calculates air-sea CO2-fluxes, as implemented in the LOSCAR model (Zeebe, 2012).
    '''
    return k * mol2GtC * area * (pCO2_atm - pCO2_oce)
def calcAirSeaCO2Flux_HAMOCC(T, area, pCO2_atm, pCO2_oce, u10=7.0):
    '''
    Calculates air-sea CO2-fluxes, as implemented in the HAMOCC model (Ilyina et al., 2013)
    '''
    return















###################################################################################
####################### Mathematical formulations #################################
###################################################################################
def quadraticFit(depth, params):
    return params[0]*depth**2 + params[1]*depth + params[2]


