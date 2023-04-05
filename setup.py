from distutils.core import setup, Extension

module1 = Extension('np2mcpl',
                    sources = ['numpy2mcplmodule.c'],
                    include_dirs =['/usr/lib/python3.10/site-packages/numpy/core/include',
                      '/usr/local/include'],
                    library_dirs = ['/usr/local/lib'],
                    libraries = ['mcpl'])


setup (name = 'np2mcpl',
       version = '1.0',
       description = 'module for saving a numpy array to an mcpl-file.',
       ext_modules = [module1])
