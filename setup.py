import os, sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]

def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/pytorch_connectomics'

    setup(name='connectomics',
        description='Automatic Reconstruction of Connectomics with PyTorch',
        version=__version__,
        url=url,
        license='MIT',
        author='Zudi Lin, Donglai Wei',
        install_requires=['scipy'],
        include_dirs=getInclude(), 
        packages=find_packages(),
    )

if __name__=='__main__':
    # pip install --editable .
    setup_package()
