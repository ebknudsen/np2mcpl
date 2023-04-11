# np2mcpl

np2mcpl is a low-key tool to help write mcpl-files directly from numpy-arrays.

## build/install
1. Make sure you have installed the MCPL-library (and NumPy)
2. Build the python/c module: ```python setup.py build```
3. Add the build directory to PYTHONPATH _or_ run: ```python setup.py install```
4. In your python script: ```import np2mcpl```
5. Create a numpy array and save it. E.g. : ```np2mcpl.save("output",M)```

## Notes and limitations
- np2mcpl takes as input a 2D NumPy array with 9 or 12 columns, where the columns are assumed to be:
```
x y z   ux uy uz   t e_kin weight    [ px py pz ]
```
- The vector ux,uy,uz must have unit length. The optional vector px,py,pz is the polarisation vector assciated with the Monte Carlo-particle.
- At present all particles are assumed to be neutrons, and so will have the PDG-code 2112.
