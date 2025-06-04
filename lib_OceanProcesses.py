'''
This is a collection of functions that can be used externally
by an ocean C model.
'''


def advect_ocean_tracer_LOSCAR(CModel, C_w, C_c, C_i, C_d, Tglobal=0.0):
    ### assumes a list of tracer concentrations is given in order:
    ## C_warm, C_cold, C_int, C_deep
    
    k_T = CModel.k_conveyor
    # Temperature dependence of advection
    k_T  += CModel.advection_Tdep_frac * CModel.k_conveyor * Tglobal

    k_wi = CModel.k_mix_warm_int
    k_cd = CModel.k_mix_cold_deep

    dts = CModel.dt * 31557600 # converting to seconds
    
    Cw = C_w + (dts / CModel.V_warm) * ( k_wi * (C_i - C_w) )
    Cc = C_c + (dts / CModel.V_cold) * ( k_T * (C_i - C_c) + k_cd * (C_d - C_c) )
    Ci = C_i + (dts / CModel.V_int)  * ( k_T * (C_d - C_i) - k_wi * (C_i - C_w) )
    Cd = C_d + (dts / CModel.V_deep) * ( k_T * (C_c - C_d) - k_cd * (C_d - C_c) )

    return Cw, Cc, Ci, Cd

def advect_ocean_tracer_Lenton(CModel, C_w, C_c, C_i, C_d, Tglobal=0.0):
    ### assumes a list of tracer concentrations is given in order:
    ## C_warm, C_cold, C_int, C_deep 
    
    # Adapt overturning circulation depending on temperature?
    k_T  = CModel.k_THC_over + CModel.advection_Tdep_frac * CModel.k_THC_over * Tglobal
    k_U  = CModel.k_hl_over  + CModel.advection_Tdep_frac * CModel.k_hl_over  * Tglobal

    k_O = k_T + k_U
    k_wi = CModel.k_warm_int_exchange
    k_id = CModel.k_int_deep_exchange

    dts = CModel.dt * 31557600
    
    Cw = C_w + (dts / CModel.V_warm) * ( (k_T+k_wi)*C_i - (k_T+k_wi)*C_w )
    Cc = C_c + (dts / CModel.V_cold) * ( k_T*C_w + k_U*C_i - k_O*C_c )
    Ci = C_i + (dts / CModel.V_int) * ( (k_O+k_id)*C_d + k_wi*C_w - (k_O + k_wi + k_id)*C_i )
    Cd = C_d + (dts / CModel.V_deep) * ( k_O*C_c + k_id*C_i - (k_O + k_id)*C_d )
        
    return Cw, Cc, Ci, Cd


def advect_ocean_tracer_3Box(CModel, C_w, C_c, C_d, Tglobal=0.0):
    ### assumes a list of tracer concentrations is given in order:
    ## C_warm, C_cold, C_deep 
    
    # Adapt overturning circulation depending on temperature?
    k_T  = CModel.k_conveyor + CModel.advection_Tdep_frac * CModel.k_conveyor * Tglobal

    k_wd = CModel.k_mix_warm_deep
    k_cd = CModel.k_mix_cold_deep

    dts = CModel.dt * 31557600
    
    Cw = C_w + (dts / CModel.V_warm) * ( (k_T + k_wd)*C_d - (k_T + k_wd)*C_w )
    Cc = C_c + (dts / CModel.V_cold) * ( k_T*C_w + k_cd*C_d - (k_T + k_cd)*C_c )
    Cd = C_d + (dts / CModel.V_deep) * ( (k_T + k_cd)*C_c + k_wd*C_w - (k_T + k_cd + k_wd)*C_d )
        
    return Cw, Cc, Ci, Cd







def calc_ocean_pCO2(self, TC, S, alk, dic, ph_old=8.1, dbg=0, iterative_limit=1e-10):
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
