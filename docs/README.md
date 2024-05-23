To update the documentation

1. Install connectomics package
2. Install docs package: `pip install -r requirements.txt`
3. Install correct version of pytorch_sphinx_theme: 
```
    git clone https://github.com/pytorch/pytorch_sphinx_theme
    pip install -e pytorch_sphinx_theme
```
4. Modify source files in `source/`
5. Compile the code: `make html`
