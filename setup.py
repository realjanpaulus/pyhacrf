from setuptools import setup, Extension
import numpy

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False


if use_cython:
    ext_modules = cythonize([Extension('pyhacrf.algorithms',
                                       ['pyhacrf/algorithms.pyx'],
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args = ["-ffast-math", "-O4"]),
                             Extension('pyhacrf.adjacent',
                                       ['pyhacrf/adjacent.pyx'],
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args = ["-ffast-math", "-O4"])])
else:
    ext_modules = [Extension('pyhacrf.algorithms',
                             ['pyhacrf/algorithms.c'],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = ["-ffast-math", "-O4"]),
                   Extension('pyhacrf.adjacent',
                             ['pyhacrf/adjacent.c'],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = ["-ffast-math", "-O4"])]


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='pyhacrf-datamade',
    version='0.2.2',
    packages=['pyhacrf'],
    install_requires=['numpy>=1.10', 'PyLBFGS>=0.1.3'],
    ext_modules=ext_modules,
    url='https://github.com/datamade/pyhacrf',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    maintainer='Forest Gregg',
    maintiner_email='fgregg@gmail.com',
    description='Hidden alignment conditional random field, a discriminative string edit distance',
    long_description=readme(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
    )
