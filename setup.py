from distutils.core import setup, Extension

module1 = Extension('np2mcpl',
                    sources = ['np2mcplmodule.c'],
                    include_dirs =['/usr/lib/python3.10/site-packages/numpy/core/include',
                      '/usr/local/include'],
                    library_dirs = ['/usr/local/lib'],
                    libraries = ['mcpl'])

setup (name = 'np2mcpl',
       version = '1.0',
       description = 'A module to facilitate IO btw. numpy and mcpl-files.',
       ext_modules = [module1])
