## Carbon-Climate Box Model
This repository contains a simple process-based box model for Temperature and Carbon, which can reproduce the behaviour of MPIESM-LR-1.2. This box model is based on a number of pre-existing models:

- Box model from Lenton (2000):
  - Land component of carbon box model
  - General idea of ocean carbon box model component
- (L)OSCAR (Zeebe, 2012):
  - Ocean carbon box model component improvements
  - Advection scheme (simplified here)
  - Air-sea CO2 exchange
- FaIR (Leach et al., 2021), version implementation as in FRIDA v1.0
  - CO2 radiation calculation
  - reading in non-CO2 effective radiative forcing computed with FaIR (data from Chris Smith)
- Ocean temperature box model from Li et al. (2020) (similar to that of FaIR)
  - Three layer ocean box model for the temperature model

## References

Lenton, T. M.: Land and ocean carbon cycle feedback effects on global warming in a simple Earth system model, Tellus B: Chemical and Physical Meteorology, 52:5, 1159-1188, https://doi.org/10.3402/tellusb.v52i5.17097, 2000.

Zeebe, R. E.: LOSCAR: Long-term Ocean-atmosphere-Sediment CArbon cycle Reservoir Model v2.0.4, Geosci. Model Dev., 5, 149–166, https://doi.org/10.5194/gmd-5-149-2012, 2012.

Leach, N. J., Jenkins, S., Nicholls, Z., Smith, C. J., Lynch, J., Cain, M., Walsh, T., Wu, B., Tsutsui, J., and Allen, M. R.: FaIRv2.0.0: a generalized impulse response model for climate uncertainty and future scenario exploration, Geosci. Model Dev., 14, 3007--3036, https://doi.org/10.5194/gmd-14-3007-2021, 2021.

Smith, C. J., Forster, P. M., Allen, M., Leach, N., Millar, R. J., Passerello, G. A., and Regayre, L. A.: FAIR v1.3: A simple emissions-based impulse response and carbon cycle model, Geosci. Model Dev., https://doi.org/10.5194/gmd-11-2273-2018, 2018.

Li, C., et al.: Optimal temperature overshoot profile found by limiting global sea level rise as a lower-cost climate target.Sci. Adv.6,eaaw9490, https://doi.org/10.1126/sciadv.aaw9490, 2020.
