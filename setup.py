import os
import sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
    'jupyter>=1.0',
    'scipy>=1.5',
    'scikit-learn>=0.23.1',
    'scikit-image>=0.17.2',
    'opencv-python>=4.3.0',
    'matplotlib>=3.3.0',
    'Cython==0.29.21',
    'yacs>=0.1.8',
    'h5py>=2.10.0',
    'gputil>=1.4.0',
    'imageio>=2.9.0',
    'tensorflow>=2.2.0',
    'tensorboard>=2.2.2',
    'einops>=0.3.0',
    'tqdm>=4.58.0',
    'monai>=0.9.1',
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/pytorch_connectomics'

    setup(name='connectomics',
          description='Semantic and instance segmentation toolbox for EM connectomics',
          version=__version__,
          url=url,
          license='MIT',
          author='PyTorch Connectomics Contributors',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
