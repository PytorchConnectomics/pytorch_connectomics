from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from setuptools import find_packages
import numpy as np

def setup_package():
    setup(name='vcg_connectomics',
       version='0.1.0',
       include_dirs=[np.get_include(), get_python_inc()], 
       packages=find_packages()
    )

if __name__=='__main__':
    # pip install --editable .
	setup_package()