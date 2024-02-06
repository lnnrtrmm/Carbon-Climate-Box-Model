import numpy as np

def anw_infsup(p_dictot, p_bortot, p_po4tot, p_siltot, p_so4tot, p_flutot):
    p_alknw_inf =  -p_po4tot - p_so4tot - p_flutot
    p_alknw_sup =   p_dictot + p_dictot + p_bortot + p_po4tot + p_po4tot + p_siltot
    return  p_alknw_inf, p_alknw_sup


# Purpose: Compute total alkalinity from ion concentrations and equilibrium constants
def equation_at(p_alktot, p_h, p_dictot, p_bortot, p_po4tot, p_siltot, p_so4tot, \
                 p_flutot, K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi):

    # TOTAL H+ scale: conversion factor for Htot = aphscale * Hfree
    aphscale = 1. + p_so4tot/Ks

    # H2CO3 - HCO3 - CO3 : n=2, m=0
    znumer_dic = 2. * K1 * K2 + p_h * K1
    zdenom_dic = K1*K2 + p_h * (K1 + p_h)
    zalk_dic = p_dictot * (znumer_dic/zdenom_dic)

    # B(OH)3 - B(OH)4 : n=1, m=0
    znumer_bor =       Kb
    zdenom_bor =       Kb + p_h
    zalk_bor   = p_bortot * (znumer_bor/zdenom_bor)

    # H3PO4 - H2PO4 - HPO4 - PO4 : n=3, m=1
    znumer_po4 = 3. * K1p*K2p*K3p + p_h*(2. * K1p*K2p + p_h* K1p)
    zdenom_po4 = K1p*K2p*K3p + p_h*( K1p*K2p + p_h*(K1p + p_h))
    zalk_po4   = p_po4tot * (znumer_po4/zdenom_po4 - 1.) # Zero level of H3PO4 = 1
    
    # H4SiO4 - H3SiO4 : n=1, m=0
    znumer_sil =       Ksi
    zdenom_sil =       Ksi + p_h
    zalk_sil   = p_siltot * (znumer_sil/zdenom_sil)

    # HSO4 - SO4 : n=1, m=1
    znumer_so4 =       Ks
    zdenom_so4 =       Ks + p_h
    zalk_so4   = p_so4tot * (znumer_so4/zdenom_so4 - 1.)

    # HF - F : n=1, m=1
    znumer_flu =       Kf
    zdenom_flu =       Kf + p_h
    zalk_flu   = p_flutot * (znumer_flu/zdenom_flu - 1.)

    # H2O - OH
    zalk_wat   = Kw/p_h - p_h/aphscale
    
    equ_at = zalk_dic + zalk_bor + zalk_po4 + zalk_sil + zalk_so4 + zalk_flu \
            + zalk_wat - p_alktot

    # H2CO3 - HCO3 - CO3 : n=2
    zdnumer_dic = K1*K1*K2 + p_h*(4.*K1*K2 + p_h * K1 )
    zdalk_dic   = -p_dictot*(zdnumer_dic/zdenom_dic**2)

    # B(OH)3 - B(OH)4 : n=1
    zdnumer_bor = Kb
    zdalk_bor   = -p_bortot*(zdnumer_bor/zdenom_bor**2)

    # H3PO4 - H2PO4 - HPO4 - PO4 : n=3
    zdnumer_po4 = K1p*K2p*K1p*K2p*K3p + p_h*(4.*K1p*K1p*K2p*K3p \
                  + p_h*(9.*K1p*K2p*K3p + K1p*K1p*K2p \
                  + p_h*(4.*K1p*K2p + p_h*K1p)))
    zdalk_po4   = -p_po4tot * (zdnumer_po4/zdenom_po4**2)

    # H4SiO4 - H3SiO4 : n=1
    zdnumer_sil = Ksi
    zdalk_sil   = -p_siltot * (zdnumer_sil/zdenom_sil**2)


    # HSO4 - SO4 : n=1
    zdnumer_so4 = Ks
    zdalk_so4   = -p_so4tot * (zdnumer_so4/zdenom_so4**2)

    # HF - F : n=1
    zdnumer_flu = Kf
    zdalk_flu   = -p_flutot * (zdnumer_flu/zdenom_flu**2)

    p_deriveqn =   zdalk_dic + zdalk_bor + zdalk_po4 + zdalk_sil \
                  + zdalk_so4 + zdalk_flu - Kw/p_h**2 - 1./aphscale

    return equ_at, p_deriveqn


# Subroutine returns the root for the 2nd order approximation of the
# DIC -- B_T -- A_CB equation for [H+] (reformulated as a cubic polynomial)
# around the local minimum, if it exists.

# Returns * 1E-03 if p_alkcb <= 0
#         * 1E-10 if p_alkcb >= 2*p_dictot + p_bortot
#         * 1E-07 if 0 < p_alkcb < 2*p_dictot + p_bortot
#                    and the 2nd order approximation does not have a solution
def ahini_for_at(p_alkcb, p_dictot, p_bortot, K1, K2, Kb):

    if (p_alkcb <= 0.):
        p_hini = 1.e-3
    elif (p_alkcb >= (2.*p_dictot + p_bortot)):
        p_hini = 1.e-10
    else:
        zca = p_dictot/p_alkcb
        zba = p_bortot/p_alkcb

        # Coefficients of the cubic polynomial
        za2 = Kb*(1. - zba) + K1*(1.-zca)
        za1 = K1*Kb*(1. - zba - zca) + K1*K2*(1. - (zca+zca))
        za0 = K1*K2*Kb*(1. - zba - (zca+zca))
                                        # Taylor expansion around the minimum   
        zd = za2*za2 - 3.*za1           # Discriminant of the quadratic equation
                                        # for the minimum close to the root


        if(zd > 0.):                 # If the discriminant is positive
            zsqrtd = np.sqrt(zd)
            if(za2 < 0):
                zhmin = (-za2 + zsqrtd)/3.
            else:
                zhmin = -za1/(za2 + zsqrtd)
            p_hini = zhmin + np.sqrt(-(za0 + zhmin*(za1 + zhmin*(za2 + zhmin)))/zsqrtd)
        else:
            p_hini = 1.e-7

    return p_hini



def solve_at_general(p_alktot, p_dictot, p_bortot, p_po4tot, p_siltot, p_so4tot, p_flutot, \
        K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi, p_hini=0.0, max_diff=1e-8):

    aphscale = 1. + p_so4tot/Ks


    if p_hini==0.0: zh_ini = ahini_for_at(p_alktot, p_dictot, p_bortot, K1, K2, Kb)
    else:           zh_ini = p_hini
    
    zalknw_inf, zalknw_sup = anw_infsup(p_dictot, p_bortot, p_po4tot, p_siltot, p_so4tot, p_flutot)

    zdelta = (p_alktot-zalknw_inf)**2 + 4.*Kw/aphscale

    if(p_alktot >= zalknw_inf):
        zh_min = 2.*Kw /( p_alktot-zalknw_inf + np.sqrt(zdelta) )
    else:
        zh_min = aphscale*(-(p_alktot-zalknw_inf) + np.sqrt(zdelta) ) / 2.


    zdelta = (p_alktot-zalknw_sup)**2 + 4.*Kw/aphscale

    if(p_alktot <= zalknw_sup):
        zh_max = aphscale*(-(p_alktot-zalknw_sup) + np.sqrt(zdelta) ) / 2.
    else:
        zh_max = 2.*Kw /( p_alktot-zalknw_sup + np.sqrt(zdelta))


    zh = max(min(zh_max, zh_ini), zh_min)
    niter_atgen        = 0                 # Reset counters of iterations
    zeqn_absmin        = 999999999.        # WATCHOUT, Fortran: HUGE(1.)
    zh_lnfactor        = 1.

    while abs(zh_lnfactor) > max_diff:
        if(niter_atgen >= 50):
            zh = -1.
            break

        zh_prev = zh
        zeqn, zdeqndh = equation_at(p_alktot, zh, p_dictot, p_bortot, p_po4tot, p_siltot, p_so4tot, \
                                 p_flutot, K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi)

        # Adapt bracketing interval
        if(zeqn > 0.):
            zh_min = zh_prev
        elif(zeqn < 0.):
            zh_max = zh_prev
        else:
            # zh is the root; unlikely but, one never knows
            break

        # Now determine the next iterate zh
        niter_atgen = niter_atgen + 1

        if(abs(zeqn) >= 0.5*zeqn_absmin):
            # if the function evaluation at the current point is
            # not decreasing faster than with a bisection step (at least linearly)
            # in absolute value take one bisection step on [ph_min, ph_max]
            # ph_new = (ph_min + ph_max)/2d0
      
            # In terms of [H]_new:
            # [H]_new = 10**(-ph_new)
            #         = 10**(-(ph_min + ph_max)/2d0)
            #         = SQRT(10**(-(ph_min + phmax)))
            #         = SQRT(zh_max * zh_min)
            zh = np.sqrt(zh_max * zh_min)
            zh_lnfactor = (zh - zh_prev)/zh_prev # Required to test convergence below
        else:
            # dzeqn/dpH = dzeqn/d[H] * d[H]/dpH
            #           = -zdeqndh * LOG(10) * [H]
            # \Delta pH = -zeqn/(zdeqndh*d[H]/dpH) = zeqn/(zdeqndh*[H]*LOG(10))
            #
            # pH_new = pH_old + \deltapH
            #
            # [H]_new = 10**(-pH_new)
            #         = 10**(-pH_old - \Delta pH)
            #         = [H]_old * 10**(-zeqn/(zdeqndh*[H]_old*LOG(10)))
            #         = [H]_old * EXP(-LOG(10)*zeqn/(zdeqndh*[H]_old*LOG(10)))
            #         = [H]_old * EXP(-zeqn/(zdeqndh*[H]_old))

            zh_lnfactor = -zeqn/(zdeqndh*zh_prev)

            if(abs(zh_lnfactor) > 1.0):
                zh = zh_prev*np.exp(zh_lnfactor)
            else:
                zh_delta    = zh_lnfactor*zh_prev
                zh          = zh_prev + zh_delta
            

            if( zh < zh_min ):
                # if [H]_new < [H]_min
                # i.e., if ph_new > ph_max then
                # take one bisection step on [ph_prev, ph_max]
                # ph_new = (ph_prev + ph_max)/2d0
                # In terms of [H]_new:
                # [H]_new = 10**(-ph_new)
                #         = 10**(-(ph_prev + ph_max)/2d0)
                #         = SQRT(10**(-(ph_prev + phmax)))
                #         = SQRT([H]_old*10**(-ph_max))
                #         = SQRT([H]_old * zh_min)
                zh                = np.sqrt(zh_prev * zh_min)
                zh_lnfactor       = (zh - zh_prev)/zh_prev # Required to test convergence below

            if( zh > zh_max ):
                # if [H]_new > [H]_max
                # i.e., if ph_new < ph_min, then
                # take one bisection step on [ph_min, ph_prev]
                # ph_new = (ph_prev + ph_min)/2d0
                # In terms of [H]_new:
                # [H]_new = 10**(-ph_new)
                #         = 10**(-(ph_prev + ph_min)/2d0)
                #         = SQRT(10**(-(ph_prev + ph_min)))
                #         = SQRT([H]_old*10**(-ph_min))
                #         = SQRT([H]_old * zhmax)
                zh                = np.sqrt(zh_prev * zh_max)
                zh_lnfactor       = (zh - zh_prev)/zh_prev # Required to test convergence below

        zeqn_absmin = min( abs(zeqn), zeqn_absmin)

        # Stop iterations once |\delta{[H]}/[H]| < rdel
        # <=> |(zh - zh_prev)/zh_prev| = |EXP(-zeqn/(zdeqndh*zh_prev)) -1| < rdel
        # |EXP(-zeqn/(zdeqndh*zh_prev)) -1| ~ |zeqn/(zdeqndh*zh_prev)|

        # Alternatively:
        # |\Delta pH| = |zeqn/(zdeqndh*zh_prev*LOG(10))|
        #             ~ 1/LOG(10) * |\Delta [H]|/[H]
        #             < 1/LOG(10) * rdel

        # Hence |zeqn/(zdeqndh*zh)| < rdel

        # rdel <-- pp_rdel_ah_target

    
    #solve_at_general = zh

    #IF(PRESENT(p_val)) THEN
    #IF(zh > 0.) THEN
    #  p_val = equation_at(p_alktot, zh,       p_dictot, p_bortot,              &
    #                      p_po4tot, p_siltot,                                  &
    #                      p_so4tot, p_flutot,                                  &
    #                      K1, K2, Kb, Kw, Ks, Kf, K1p, K2p, K3p, Ksi)    
    #else:
    #   p_val = HUGE(1.)

    return zh  

