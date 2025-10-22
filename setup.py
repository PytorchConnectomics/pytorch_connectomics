import os
import sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

# Core dependencies - always required
requirements = [
    # Deep learning frameworks
    'torch>=1.8.0',
    'numpy>=1.19.0',

    # PyTorch Lightning & MONAI (PRIMARY frameworks)
    'pytorch-lightning>=2.0.0',
    'monai>=0.9.1',
    'torchmetrics>=0.11.0',

    # Configuration management (Hydra/OmegaConf is PRIMARY, YACS is legacy)
    'omegaconf>=2.1.0',
    'yacs>=0.1.8',  # Legacy - being phased out

    # Scientific computing
    'scipy>=1.5',
    'scikit-learn>=0.23.1',
    'scikit-image>=0.17.2',

    # Image processing & I/O
    'opencv-python>=4.3.0',
    'h5py>=2.10.0',
    'imageio>=2.9.0',

    # Visualization & logging
    'matplotlib>=3.3.0',
    'tensorboard>=2.2.2',

    # Utilities
    'tqdm>=4.58.0',
    'einops>=0.3.0',
    'psutil>=5.8.0',

    # Post-processing (required for segmentation)
    'cc3d>=3.0.0',

    # Build tools
    'Cython>=0.29.22',
]

# Optional dependencies for specific features
extras_require = {
    # Full installation with all recommended features
    'full': [
        'gputil>=1.4.0',
        'jupyter>=1.0',
        'tifffile>=2021.11.2',
        'wandb>=0.13.0',
    ],

    # Hyperparameter optimization
    'optim': [
        'optuna>=2.10.0',
    ],

    # Advanced metrics (skeleton-based)
    'metrics': [
        # Install manually: pip install git+https://github.com/funkelab/funlib.evaluate.git
    ],

    # 3D visualization
    'viz': [
        'neuroglancer>=1.0.0',
    ],

    # Experiment tracking
    'wandb': [
        'wandb>=0.13.0',
    ],

    # TIFF file support
    'tiff': [
        'tifffile>=2021.11.2',
    ],

    # Documentation building
    'docs': [
        'sphinx==3.4.3',
        'sphinxcontrib-katex',
        'jinja2==3.0.3',
        'sphinxcontrib-applehelp==1.0.4',
        'sphinxcontrib-devhelp==1.0.2',
        'sphinxcontrib-htmlhelp==2.0.1',
        'sphinxcontrib-qthelp==1.0.3',
        'sphinxcontrib-serializinghtml==1.1.5',
    ],

    # Development and testing
    'dev': [
        'pytest>=6.0.0',
        'pytest-benchmark>=3.4.0',
    ],

    # MedNeXt models (external package)
    # Install separately: pip install -e /projects/weilab/weidf/lib/MedNeXt
    # Or from your local MedNeXt installation path
    'mednext': [
        # Placeholder - install manually as documented in .claude/MEDNEXT.md
    ],
}


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName)]


def setup_package():
    __version__ = '2.0.0'
    url = 'https://github.com/zudi-lin/pytorch_connectomics'

    setup(
        name='connectomics',
        description='Semantic and instance segmentation toolbox for EM connectomics',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        version=__version__,
        url=url,
        license='MIT',
        author='PyTorch Connectomics Contributors',
        python_requires='>=3.8',
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=getInclude(),
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Image Processing',
        ],
    )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
