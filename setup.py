from distutils.core import setup, Extension

module1 = Extension('numpy2mcpl',
                    sources = ['numpy2mcplmodule.c'],
                    include_dirs =['/usr/lib/python3.10/site-packages/numpy/core/include',
                      '/usr/local/include'],
                    library_dirs = ['/usr/local/lib'],
                    libraries = ['mcpl'])


setup (name = 'numpy2mcpl',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
