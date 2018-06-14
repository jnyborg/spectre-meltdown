from distutils.core import setup, Extension

setup(name='Spectre',
      version='1.0',
      ext_modules=[Extension('pyspectre', sources=['pyspectre.c'], extra_compile_args=["-std=c99"]),
                   Extension('pyspectre35', sources=['pyspectre35.c'], extra_compile_args=["-std=c99"]),
                   Extension('pyspectre150', sources=['pyspectre150.c'], extra_compile_args=["-std=c99"])
                   ],
      )