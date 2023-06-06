# np2mcpl
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

np2mcpl is a low-key tool to help write mcpl-files directly from numpy-arrays.

## Build/Install/Use
1. Make sure you have installed the MCPL-library (and NumPy)
2. Build the python/c module: ```python setup.py build```
3. Add the build directory to PYTHONPATH _or_ run: ```python setup.py install```
4. In your python script: ```import np2mcpl```
5. Create a numpy array and save it. E.g. : ```np2mcpl.save("output",M)```

## Notes and limitations
- np2mcpl takes as input a 2D NumPy array with 10 or 13 columns, where the columns are assumed to be:
```
PDG-code  x y z   ux uy uz   t e_kin weight    [ px py pz ]
```
 where the PDG-code denotes which kind of particle it is, e.g. 2112 for neutrons, 22 for photon, ... as documented here: [Particle Numbering Scheme](https://pdg.lbl.gov/2023/mcdata/mc_particle_id_contents.html)
- The vector ux,uy,uz must have unit length. The optional vector px,py,pz is the polarisation vector associated with the Monte Carlo-particle.
- Units follow the MCPL-standard. I.e. x,y,z in cm, e_kin in MeV, t in ms.
- The input array is expected to be made of floating point numbers (including the PDG-code) in either double or single precision. If the numpy array is in single precision, this will be reflected in the mcpl-file.
