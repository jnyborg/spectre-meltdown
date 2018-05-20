from distutils.core import setup, Extension

setup(name='Spectre',
      version='1.0',
      ext_modules=[Extension('pyspectre', sources=['pyspectre.c'])])
