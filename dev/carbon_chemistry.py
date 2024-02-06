import numpy as np
from mocsy_solver import solve_at_general
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

# As in ICON
def calc_ak0(TC,S):

    smicr = 1.e-6
    perc  = 1.e-2

    c00 = 9345.17
    c01 = -60.2409
    c02 = 23.3585
    c03 = 0.023517
    c04 = -0.00023656
    c05 = 0.0047036

    T = TC + 273.15
    Ti = 1./T
    q = T * perc
    q2 = q*q
    log_q = np.log(q)

    cek0 = c00*Ti + c01 + c02 * log_q + S * (c03 + c04*T + c05*q2)
    ak0 = np.exp(cek0)*smicr
    
    return ak0

# After Benezeth et al. (2018)
def calc_aksp_dol(TC): 
    a = 17.502
    b = -4220.119
    c = -0.0689

    T = TC + 273.15

    log10_ksp = a + b/T + c*T

    return 10**log10_ksp

# After Zeebe and Tyrell (2019)
def adj_aksp(aksp, ca=0.0103, mg=0.0528):
    s_ca = 0.185
    s_mg = 0.518
    return aksp + aksp * ( s_ca*(ca/0.0103 - 1.) + s_mg*(mg/0.0528 - 1.) )

# After Zeebe and Tyrell (2019)
def adj_ak1(ak1, ca=0.0103, mg=0.0528):
    s_ca = 0.005
    s_mg = 0.017
    return ak1 + ak1 * ( s_ca*(ca/0.0103 - 1.) + s_mg*(mg/0.0528 - 1.) )

# After Zeebe and Tyrell (2019)
def adj_ak2(ak2, ca=0.0103, mg=0.0528):
    s_ca = 0.157
    s_mg = 0.420
    return ak2 + ak2 * ( s_ca*(ca/0.0103 - 1.) + s_mg*(mg/0.0528 - 1.) )

def calc_aksp(depth_levels,TC,S,stretch_c=1.0,draftave=0.0, ca=0.0103, mg=0.0528):

    if np.shape(depth_levels) == np.shape(np.array([0.0])) and depth_levels[0]==0.0 and draftave == 0.0: 
        print('Calc. AKSP for surface only')
        lsurf=True
    else:
        print('Calc. AKSP for subsurface level(s)')
        lsurf=False

    rgas = 83.131

    pa0_11 = -45.96
    pa1_11 = 0.5304
    #pa2_11 = 0.0
    pb0_11 = -11.76e-3
    pb1_11 =  -0.3692e-3
    #pb2_11 = 0.0

    akcc1 = -171.9065
    akcc2 = -0.077993
    akcc3 = 2839.319
    akcc4 = 71.595
    akcc5 = -0.77712
    akcc6 = 0.0028426
    akcc7 = 178.34
    akcc8 = -0.07711
    akcc9 = 0.0041249

    arafra = 0.0
    calfra = 1.0 - arafra
    aracal = arafra * 1.45 + calfra

    T = TC + 273.15
    Ti = 1./T
    log_T = np.log(T)
    invlog10 = 1./np.log(10.)
    sqrt_S = np.sqrt(S)
    S2 = S*S

    # Pressure related calculations
    if lsurf:
        P = 0.0
        cp = 0.0
    else:
        ptiestu = depth_levels * stretch_c + draftave
        P = 0.1025 * ptiestu
        cp = P / (rgas * T)


    log10ksp = akcc1 + akcc2*T +  akcc3*Ti + akcc4*log_T*invlog10       \
            + (akcc5 +akcc6*T + akcc7*Ti) * sqrt_S                      \
            + akcc8*S + akcc9*(S*sqrt_S)
    aksp0 = 10.**log10ksp


    if lsurf:
        lnkpk0_11 = 0.0
    else:
        deltav = pa0_11 + pa1_11 * TC# + pa2_11 * TC**2
        deltak = pb0_11 + pb1_11 * TC# + pb2_11 * TC**2
        lnkpk0_11 = -1.0 * (deltav * cp + 0.5 * deltak * cp * P)

    aksp0 = aracal* aksp0 * np.exp(lnkpk0_11)

    aksp0 = adj_aksp(aksp0, ca=ca, mg=mg)

    return aksp0
    

def calc_diss_const(depth_levels,TC,S,stretch_c=1.0,draftave=0.0, silent=True, ca=0.0103, mg=0.0528):
    ## definition of constants as in
    # MO_BGC_CONSTANTS.f90
    #zlevs=len(depth_levels)

    if not silent: print('Calculate the dissociation constants')
    if np.shape(depth_levels) == np.shape(np.array([0.0])) and depth_levels[0]==0.0 and draftave == 0.0: 
        if not silent: print('... for surface only')
        lsurf=True
    else:
        #print('Calc. AKSP for subsurface level(s)')
        lsurf=False

    rgas = 83.131

    pa0 = np.array((-18.03, -9.78,  -14.51, -23.12, -26.57, -29.48, \
                   -25.5, -15.82, -29.48, -20.02, -45.96))
    pa1 = np.array((0.0466, -0.0090, 0.1211, 0.1758, 0.2020, 0.1622, \
                   0.1271, -0.0219, 0.1622, 0.1119, 0.5304))
    pa2 =np.array((0.316e-3, -0.942e-3, -0.321e-3, -2.647e-3, -3.042e-3,\
                  -2.6080e-3,  0.0,  0.0, -2.608e-3,-1.409e-3, 0.0))
    pb0 = np.array((-4.53e-3, -3.91e-3, -2.67e-3, -5.15e-3,-4.08e-3, \
                -2.84e-3,  -3.08e-3,   1.13e-3, -2.84e-3, -5.13e-3,-11.76e-3))
    pb1 = np.array((0.09e-3, 0.054e-3, 0.0427e-3, 0.09e-3, 0.0714e-3, \
                0.0, 0.0877e-3, -0.1475e-3, 0.0, 0.0794e-3, -0.3692e-3))
    pb2 = np.zeros(11)
    #lnkpk0 = np.zeros((11, zlevs))


    c10 = -3633.86
    c11 = 61.2172
    c12 = -9.67770
    c13 = 0.011555
    c14 = -0.0001152

    c20 = -471.78
    c21 = -25.9290
    c22 = 3.16967
    c23 = 0.0178
    c24 =-0.0001122

    cb0 = -8966.90
    cb1 = -2890.53
    cb2 = -77.942
    cb3 = 1.728
    cb4 = -0.0996
    cb5 = 148.0248
    cb6 = 137.1942
    cb7 = 1.62142
    cb8 = 24.4344
    cb9 = 25.085
    cb10 =0.2474
    cb11 =0.053105

    cw0 = 148.9802
    cw1 = -13847.26
    cw2 = -23.6521
    cw3 = 118.67
    cw4 = -5.977
    cw5 = 1.0495
    cw6 = -0.01615


    akcc1 = -171.9065
    akcc2 = -0.077993
    akcc3 = 2839.319
    akcc4 = 71.595
    akcc5 = -0.77712
    akcc6 = 0.0028426
    akcc7 = 178.34
    akcc8 = -0.07711
    akcc9 = 0.0041249

    cksi1  =   -8904.2
    cksi2  =   117.385
    cksi3  =   -19.334
    cksi4  =   -458.79
    cksi5  =    3.5913
    cksi6  =    188.74
    cksi7  =  - 1.5998
    cksi8  =  -12.1652
    cksi9  =   0.07871
    cksi10 =  0.001005
    
    
    cks1  =   -4276.1
    cks2  =   141.328
    cks3  =  - 23.093
    cks4  =   -13856.
    cks5  =    324.57
    cks6  =  - 47.986
    cks7  =    35474.
    cks8  =  - 771.54
    cks9  =   114.723
    cks10 =   - 2698.
    cks11 =     1776.
    cks12 = -0.001005
    
    ckf1 = 874.
    ckf2 = -9.68
    ckf3 = 0.111
        
    ck1p1 = -4576.752
    ck1p2 =   115.525
    ck1p3 =  - 18.453
    ck1p4 =  -106.736
    ck1p5 =   0.69171 
    ck1p6 =  -0.65643
    ck1p7 =  -0.01844
        
    ck2p1 = -8814.715
    ck2p2 =  172.0883
    ck2p3 =  - 27.927
    ck2p4 =  -160.340
    ck2p5 =    1.3566
    ck2p6 =   0.37335
    ck2p7 = - 0.05778

    ck3p1 =  -3070.75
    ck3p2 =  - 18.141
    ck3p3 =  17.27039
    ck3p4 =   2.81197
    ck3p5 = -44.99486
    ck3p6 = - 0.09984
    

    arafra = 0.0
    calfra = 1.0 - arafra
    aracal = arafra * 1.45 + calfra

    if not silent: print('...Preparation of help variables')
    #### calculation of dissociation constant as in
    # MO_CHEMCON.f90
    if not lsurf: ptiestu = depth_levels * stretch_c + draftave


    T = TC + 273.15
    Ti = 1./T
    log_T = np.log(T)
    invlog10 = 1./np.log(10.)
    sqrt_S = np.sqrt(S)
    S2 = S*S


    pis = 19.924 * S / ( 1000. - 1.005 * S )
    pis2 = pis * pis

    sti = 0.14     * S * 1.025/1.80655/96.062
    fti = 0.000067 * S * 1.025/1.80655/18.9984

    if not silent: print('...Calculation of cks')
    # calculation of ck's
    ck1 =  c10*Ti + c11 + c12*log_T + c13*S + c14* S2
    ck2 =  c20*Ti + c21 + c22*log_T + c23*S + c24* S2

    ckb = (cb0 + cb1*sqrt_S + cb2*S + cb3*(S*sqrt_S) + cb4*(S2)) * Ti + cb5 + cb6*sqrt_S \
            + cb7*S - (cb8+cb9*sqrt_S + cb10*S) * log_T + cb11 * sqrt_S * T

    ckw = cw0  + cw1*Ti + cw2*log_T + sqrt_S*(cw3*Ti + cw4 +cw5*log_T) + cw6*S

    cksi =  cksi1*Ti + cksi2 + cksi3 * log_T + ( cksi4/T + cksi5 ) * np.sqrt(pis)  \
            + ( cksi6*Ti + cksi7) * pis + ( cksi8*Ti + cksi9) * pis2 + np.log(1.0 + cksi10*S)

    cks =  cks1/T + cks2 + cks3 * log_T + ( cks4/T + cks5 + cks6 * log_T ) * np.sqrt(pis) \
          + (cks7/T + cks8 + cks9 * log_T) * pis + cks10/T * pis**1.5 \
          +  cks11/T * pis2 + np.log(1.0 + cks12 * S ) 

    ckf = ckf1*Ti + ckf2  + ckf3*sqrt_S

    ck1p =  ck1p1/T + ck1p2 + ck1p3 * log_T + ( ck1p4/T + ck1p5 ) * sqrt_S + ( ck1p6/T + ck1p7 ) * S
    ck2p =  ck1p1/T + ck1p2 + ck1p3 * log_T + ( ck1p4/T + ck1p5 ) * sqrt_S + ( ck1p6/T + ck1p7 ) * S
    ck3p =  ck3p1/T + ck3p2 + ( ck3p3/T + ck3p4 ) * sqrt_S + ( ck3p5/T + ck3p6 ) * S 

    if not silent: print('...Convert into aks')
    # convert into ak's
    ak1 = 10. ** ck1
    ak2 = 10. ** ck2
    akb = np.exp(ckb)
    akw = np.exp(ckw)
    aksi = np.exp(cksi)
    aks = np.exp(ckf)
    akf = np.exp(ckf)
    ak1p = np.exp(ck1p)
    ak2p = np.exp(ck2p)
    ak3p = np.exp(ck3p)

    log10ksp = akcc1 + akcc2*T +  akcc3*Ti + akcc4*log_T*invlog10       \
               + (akcc5 +akcc6*T + akcc7*Ti) * sqrt_S                      \
               + akcc8*S + akcc9*(S*sqrt_S)
    aksp0 = 10.**log10ksp


    if not lsurf:
        if not silent: print('... Pressure related calculations')
        # Pressure related calculations
        P = 0.1025 * ptiestu
        cp = P / (rgas * T)



        for i in range(0,11):
            deltav = pa0[i] + pa1[i] * TC + pa2[i] * TC**2
            deltak = pb0[i] + pb1[i] * TC + pb2[i] * TC**2
            tmp = -1.0 * (deltav * cp + 0.5 * deltak * cp * P)
            if i == 0: lnkpk0 = tmp[np.newaxis]
            else: lnkpk0 = np.append(lnkpk0, tmp[np.newaxis], axis=0)
    else: 
        lnkpk0 = np.zeros((11))


    if not silent: print('... Final conversion')
    total2free_0p = 1./(1. + sti/aks)
    free2sws_0p  = 1. + sti/aks + fti/(akf*total2free_0p)
    total2sws_0p = total2free_0p * free2sws_0p 

    aks = aks * np.exp(lnkpk0[1-1])   # -1 comes from FORTRAN -> Python conversion

    total2free = 1. / (1. + sti/aks)

    akf = akf * np.exp(lnkpk0[2-1])
    akf = akf / total2free

    free2sws = 1. + sti/aks + fti/(akf*total2free)
    total2sws = total2free * free2sws
    sws2total = 1. / total2sws

    ak1 = ak1 * total2sws_0p
    ak2 = ak2 * total2sws_0p
    akb = akb * np.exp(lnkpk0[9-1]) 
    akw = akw * np.exp(lnkpk0[10-1]) 
    aksp0 = aracal* aksp0 * np.exp(lnkpk0[11-1]) 
    ak1p = ak1p * np.exp(lnkpk0[3-1])
    ak2p = ak2p * np.exp(lnkpk0[4-1])
    ak3p = ak3p * np.exp(lnkpk0[5-1])
    aksi = aksi * np.exp(lnkpk0[6-1]) 

    ak1 = ak1 * np.exp(lnkpk0[7-1])
    ak2 = ak2 * np.exp(lnkpk0[8-1])

    aks = aks
    akf = akf
    ak1p = ak1p * sws2total
    ak2p = ak2p * sws2total
    ak3p = ak3p * sws2total
    aksi = aksi * sws2total
    ak1 = ak1 * sws2total
    ak2 = ak2 * sws2total
    akb = akb * sws2total
    akw = akw * sws2total
    aksp = aksp0

    # Adjustment of dissociation constants, when
    # Ca and Mg concentration vary from modern values.
    # -> From Zeebe and Tyrell (2019)
    #
    # This should prob. only be valid for surface ocean?
    # Better avoid using it for deeper levels
    ak1 = adj_ak1(ak1, ca=ca, mg=mg)
    ak2 = adj_ak2(ak2, ca=ca, mg=mg)
    aksp0 = adj_aksp(aksp0, ca=ca, mg=mg)
    

    if not silent: print('  ...finished')
    
    return [aks, akf, ak1p, ak2p, ak3p, aksi, ak1, ak2, akb, akw, aksp]


def update_hi(c,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,s,akb,sit,pt,alk, hi_init=None, dim=1, hastime=False,
        silent=False, max_diff=1e-8):

    salchl = 1./1.80655
    bor1 = 0.00023
    bor2 = 1./10.82

    rrrcl = salchl * 1.025 * bor1 * bor2

    bt = rrrcl*s
    # sulfate Morris & Riley (1966)
    sti = 0.14 *  s*1.025/1.80655/96.062
    # fluoride Riley (1965)
    ft = 0.000067 * s*1.025/1.80655/18.9984

    #zeros=np.zeros_like(c)
    if hi_init is None: hi_init = np.zeros_like(c)

    if dim == 0:
        if not silent: print('Scalar H+ calculation')
        h = solve_at_general(alk,c,bt,pt,sit,sti,ft,ak1,ak2,akb,akw,aks,akf,ak1p,ak2p,ak3p,aksi,hi_init, max_diff=max_diff)

    elif dim == 1:
        h = np.zeros_like(c)        
        if not silent: print('One dimensional H+ calculation')
        for i in range(len(h)):
            if i > 0 and hastime: hi_init_1d = h[i-1]
            else: hi_init_1d = hi_init[i]
            h[i] = solve_at_general(alk[i],c[i],bt[i],pt[i],sit[i],sti[i],ft[i],ak1[i],ak2[i],
                                     akb[i],akw[i],aks[i],akf[i],ak1p[i],ak2p[i],ak3p[i],aksi[i],hi_init_1d, max_diff=max_diff)
    elif dim == 2:
        if not silent: print('Two dimensional H+ calculation')
        h = np.zeros_like(c)        
        for i in range(len(h[:,0])):
            for j in range(len(h[0,:])):
                if i > 0 and hastime: hi_init_2d = h[i-1,j]
                else: hi_init_2d = hi_init[i,j]
                h[i,j] = solve_at_general(alk[i,j],c[i,j],bt[i,j],pt[i,j],sit[i,j],sti[i,j],ft[i,j],
                                       ak1[i,j],ak2[i,j],akb[i,j],akw[i,j],aks[i,j],akf[i,j],ak1p[i,j],
                                       ak2p[i,j],ak3p[i,j],aksi[i,j],hi_init_2d, max_diff=max_diff)
    elif dim == 3:
        if not silent: print('Three dimensional H+ calculation')
        h = np.zeros_like(c)        
        for i in range(len(h[:,0,0])):
            for j in range(len(h[0,:,0])):
                for k in range(len(h[0,0,:])):
                    if i > 0 and hastime: hi_init_3d = h[i-1,j,k]
                    else: hi_init_3d = hi_init[i,j,k]
                    h[i,j,k] = solve_at_general(alk[i,j,k],c[i,j,k],bt[i,j,k],pt[i,j,k],sit[i,j,k],sti[i,j,k],
                            ft[i,j,k],ak1[i,j,k],ak2[i,j,k],akb[i,j,k],akw[i,j,k],aks[i,j,k],akf[i,j,k],
                            ak1p[i,j,k],ak2p[i,j,k],ak3p[i,j,k],aksi[i,j,k],hi_init_3d, max_diff=max_diff)
    else:
        print('Error: Function update_hi not built for array with ndim > 3')
                



    return h

def get_co3(dic, hi, ak13, ak23):
    return dic / (1. + hi * (1. + hi/ak13)/ak23)

def get_co2(dic, hi):
    print('Not yet implemented')
    return

# calcium is hard-coded in ICON as 1./97. 
def get_omega(co3, aksp, calcium=0.0103):
    return calcium * co3 / aksp

def get_omega_dol(co3, T, ca=0.0103,  mg=0.0528):
    # This still gives weird values.
    # Probably doing something wrong here
    aksp = calc_aksp_dol(T)
    return ca * mg * co3**2 / aksp

# This may only be correct for surface ?!
def get_solco2(TC,S):
    return calc_ak0(TC,S)

def get_pco2(dic, hi, ak1, ak2, solco2):
    return dic / ((1 + ak1 * (1. + ak2/hi)/hi)*solco2)

def get_dic(pco2, hi, ak1, ak2, solco2):
    return pco2 * ((1 + ak1 * (1. + ak2/hi)/hi)*solco2)

def calc_revelle(dic, alk, TC, S, sil, phos, all_aks_inp=[], depth=0.0, delta=1.0e-6, dim=0, calc_aks=False,
        return_ph=True, return_pco2=False, return_aks=False, ca=0.0103, mg=0.0528, max_diff=1.0e-6):

    dic_1 = dic - delta
    dic_2 = dic + delta


    if calc_aks:
        all_aks = calc_diss_const(depth, TC, S, ca=ca, mg=mg)
    else:
        all_aks = np.copy(all_aks_inp)

    try:
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
    except:
        print('all_aks not correctly handed to "calc_revelle"!')
        return

    hi = update_hi(dic,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True, max_diff=max_diff)
    pco2 = get_pco2(dic, hi, ak1, ak2, get_solco2(TC,S))

    hi_1 = update_hi(dic_1,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True, max_diff=max_diff)
    pco2_1 = get_pco2(dic_1, hi_1, ak1, ak2, get_solco2(TC,S))

    hi_2 = update_hi(dic_2,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True, max_diff=max_diff)
    pco2_2 = get_pco2(dic_2, hi_2, ak1, ak2, get_solco2(TC,S))

    revelle = (dic / pco2) * ((pco2_2 - pco2_1)/(2.0*delta))

    if return_ph and return_pco2 and return_aks:
        return revelle, pco2, -np.log10(hi), all_aks
    elif not return_ph and return_pco2 and return_aks:
        return revelle, pco2, all_aks
    elif not return_ph and not return_pco2 and return_aks:
        return revelle, all_aks
    elif not return_ph and return_pco2 and not return_aks:
        return revelle, pco2
    elif not return_ph and not return_pco2 and not return_aks:
        return revelle
    elif return_ph and not return_pco2 and return_aks:
        return revelle, -np.log10(hi), all_aks
    elif return_ph and return_pco2 and not return_aks:
        return revelle, pco2, -np.log10(hi)
    elif return_ph and not return_pco2 and not return_aks:
        return revelle, -np.log10(hi)
    else:
        print("Don't know what to return, returning only revelle")
        return revelle

def calc_revelle_hetero(dic, alk, TC, S, sil, phos, all_aks_inp=[], depth=0.0, delta=1.0e-6, dim=0, calc_rate=2.0,
        calc_aks=False, return_ph=True, return_pco2=False, return_aks=False, ca=0.0103, mg=0.0528):

    dic_1 = dic - delta
    dic_2 = dic + delta

    alk_1 = alk - delta
    alk_2 = alk + delta


    if calc_aks:
        all_aks = calc_diss_const(depth, TC, S, ca=ca, mg=mg)
    else:
        all_aks = np.copy(all_aks_inp)

    try:
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
    except:
        print('all_aks not correctly handed to "calc_revelle"!')
        return

    hi = update_hi(dic,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True)
    pco2 = get_pco2(dic, hi, ak1, ak2, get_solco2(TC,S))

    hi_1_dic = update_hi(dic_1,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True)
    pco2_1_dic = get_pco2(dic_1, hi_1_dic, ak1, ak2, get_solco2(TC,S))

    hi_2_dic = update_hi(dic_2,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk,dim=dim,silent=True)
    pco2_2_dic = get_pco2(dic_2, hi_2_dic, ak1, ak2, get_solco2(TC,S))


    hi_1_alk = update_hi(dic,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk_1,dim=dim,silent=True)
    pco2_1_alk = get_pco2(dic, hi_1_alk, ak1, ak2, get_solco2(TC,S))

    hi_2_alk = update_hi(dic,ak1,ak2,akw,aks,akf,aksi,ak1p,ak2p,ak3p,S,akb,sil,phos,alk_2,dim=dim,silent=True)
    pco2_2_alk = get_pco2(dic, hi_2_alk, ak1, ak2, get_solco2(TC,S))


    revelle = (dic / pco2) * ( (pco2_2_dic - pco2_1_dic)/(2.0*delta) + (pco2_2_alk - pco2_1_alk)/(2.0*delta) * calc_rate)


    if return_ph and return_pco2 and return_aks:
        return revelle, pco2, -np.log10(hi), all_aks
    elif not return_ph and return_pco2 and return_aks:
        return revelle, pco2, all_aks
    elif not return_ph and not return_pco2 and return_aks:
        return revelle, all_aks
    elif not return_ph and return_pco2 and not return_aks:
        return revelle, pco2
    elif not return_ph and not return_pco2 and not return_aks:
        return revelle
    elif return_ph and not return_pco2 and return_aks:
        return revelle, -np.log10(hi), all_aks
    elif return_ph and return_pco2 and not return_aks:
        return revelle, pco2, -np.log10(hi)
    elif return_ph and not return_pco2 and not return_aks:
        return revelle, -np.log10(hi)
    else:
        print("Don't know what to return, returning only revelle")
        return revelle
