# Troubleshooting

Here is a summary if common compilation issues that you 
might face while compiling / running this code:

## Compilation errors when compiling the library
If you encounter build errors like the following:
```
vcg_connectomics/utils/seg/cpp/seg_core/cpp-seg_core.cpp:3:43: fatal error: boost/pending/disjoint_sets.hpp: No such file or directory
     #include <boost/pending/disjoint_sets.hpp>
                                               ^
    compilation terminated.
    error: command 'gcc' failed with exit status 1

    ----------------------------------------
```
you can try
```
conda install -c statiskit libboost-dev
```
which shows the package information:
```
The following NEW packages will be INSTALLED:
     libboost-dev       statiskit/linux-64::libboost-dev-1.68.0-2748
     libedit            pkgs/main/linux-64::libedit-3.1.20181209-hc058e9b_0
     yaml               conda-forge/linux-64::yaml-0.1.7-h14c3975_1001
```
After installing those packages, run:
```
pip install --editable .
```
which should show:
```
Successfully built boost sqlalchemy
Installing collected packages: asn1crypto, cryptography, http-ece, Mastodon.py, sqlalchemy, boost, vcg-connectomics
  Running setup.py develop for vcg-connectomics
Successfully installed Mastodon.py-1.3.1 asn1crypto-0.24.0 boost-0.1 cryptography-2.6.1 http-ece-1.1.0 sqlalchemy-1.3.2 vcg-connectomics
```