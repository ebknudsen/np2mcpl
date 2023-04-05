import numpy2mcpl
import numpy as np
import math


system_p = np.array([[1., 2., 3.,   4./math.sqrt(), 5./math.sqrt(15.), 6./math.sqrt(15.),    7., 9., 4., 0.,1.,0. ],
                     [3., 2., 5.,   6./math.sqrt(14.), 3./math.sqrt(14.), 5./math.sqrt(14.),    6., 3., 5., 1.,0.,0.]])

numpy2mcpl.numpy2mcpl_dump("thefile",system_p)

