from distutils.core import setup, Extension

module1 = Extension('numpy2mcpl',
                    sources = ['numpy2mcplmodule.c'])

setup (name = 'Numpy2MCPL',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
