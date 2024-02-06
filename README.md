# Climate box model
This repository contains a simple box model for Temperature and Carbon. It is based on a number of pre-existing models:

- Box model from Lenton (2000, "Land and ocean carbon cycle feedback effects on global warming in a simple Earth system model"):
  - Land component of carbon box model
  - General idea of ocean carbon box model component
- (L)OSCAR (Zeebe, 2012):
  - Ocean carbon box model component improvements
  - Advection scheme (simplified here)
  - Air-sea CO2 exchange
- FaIR (Leach et al., 2021), version implementation as in FRIDA v1.0
  - CO2 radiation calculation
  - reading in non-CO2 effective radiative forcing computed with FaIR (data from Chris Smith)
- Li et al. (2020, "Optimal temperature overshoot profile found by limiting global sea level rise as a lower-cost climate target")
  - Three layer ocean box model for the temperature model
  
