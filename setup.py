from distutils.core import setup, Extension
import numpy as np

module1 = Extension('np2mcpl',
                    sources = ['src/np2mcplmodule.c'],
                    include_dirs =[np.get_include(),'/usr/local/include'],
                    library_dirs = ['/usr/local/lib'],
                    libraries = ['mcpl'])

setup (name = 'np2mcpl',
       version = '1.0',
       description = 'A module to facilitate IO btw. numpy and mcpl-files.',
       ext_modules = [module1])
