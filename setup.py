import os, sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension
from Cython.Distutils import build_ext

def getExt():
    # extensions under segmenation/
    return [
        Extension(
            name='vcg_connectomics.utils.seg.seg_dist',
            sources=['vcg_connectomics/utils/seg/seg_dist.pyx',
                     'vcg_connectomics/utils/seg/cpp/seg_dist/cpp-distance.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='vcg_connectomics.utils.seg.seg_core',
            sources=['vcg_connectomics/utils/seg/seg_core.pyx',
                     'vcg_connectomics/utils/seg/cpp/seg_core/cpp-seg2seg.cpp',
                     'vcg_connectomics/utils/seg/cpp/seg_core/cpp-seg2gold.cpp',
                     'vcg_connectomics/utils/seg/cpp/seg_core/cpp-seg_core.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='vcg_connectomics.utils.seg.seg_eval',
            sources=['vcg_connectomics/utils/seg/seg_eval.pyx',
                     'vcg_connectomics/utils/seg/cpp/seg_eval/cpp-comparestacks.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='vcg_connectomics.utils.seg.seg_malis',
            sources=['vcg_connectomics/utils/seg/seg_malis.pyx',
                     'vcg_connectomics/utils/seg/cpp/seg_malis/cpp-malis_core.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        )
    ]

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]

def setup_package():

    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/pytorch_connectomics'

    setup(name='vcg_connectomics',
        version=__version__,
        url=url,
        license='MIT',
        install_requires=['cython','scipy','boost'],
        cmdclass = {'build_ext': build_ext},
        include_dirs=getInclude(), 
        packages=['vcg_connectomics'],
        ext_modules = getExt()
    )

if __name__=='__main__':
    # pip install --editable .
    setup_package()