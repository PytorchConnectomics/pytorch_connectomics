import os, sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext

def getExt():
    # extensions under segmenation/
    return [
        Extension(
            name='torch_connectomics.utils.seg.seg_dist',
            sources=['torch_connectomics/utils/seg/seg_dist.pyx',
                     'torch_connectomics/utils/seg/cpp/seg_dist/cpp-distance.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='torch_connectomics.utils.seg.seg_core',
            sources=['torch_connectomics/utils/seg/seg_core.pyx',
                     'torch_connectomics/utils/seg/cpp/seg_core/cpp-seg2seg.cpp',
                     'torch_connectomics/utils/seg/cpp/seg_core/cpp-seg2gold.cpp',
                     'torch_connectomics/utils/seg/cpp/seg_core/cpp-seg_core.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='torch_connectomics.utils.seg.seg_eval',
            sources=['torch_connectomics/utils/seg/seg_eval.pyx',
                     'torch_connectomics/utils/seg/cpp/seg_eval/cpp-comparestacks.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        ),
        Extension(
            name='torch_connectomics.utils.seg.seg_malis',
            sources=['torch_connectomics/utils/seg/seg_malis.pyx',
                     'torch_connectomics/utils/seg/cpp/seg_malis/cpp-malis_core.cpp'],
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

    setup(name='torch_connectomics',
        description='Automatic Annotation of Connectomics with PyTorch',
        version=__version__,
        url=url,
        license='MIT',
        author='Zudi Lin',
        install_requires=['cython','scipy','boost'],
        cmdclass = {'build_ext': build_ext},
        include_dirs=getInclude(), 
        packages=find_packages(),
        ext_modules = getExt()
    )

if __name__=='__main__':
    # pip install --editable .
    setup_package()
