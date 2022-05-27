Acknowledgement: this document is adapted from [Pytorch](https://github.com/pytorch/pytorch/edit/master/CONTRIBUTING.md).

# Table of Contents
<!-- toc -->
- [Contributing to PyTorch Connectomics](#contributing-to-pytorch)
- [Developing PyTorch Connectomics](#developing-pytorch)
  - [Tips and Debugging](#tips-and-debugging)
- [Codebase Structure](#codebase-structure)
- [Unit Testing](#unit-testing)
  - [Python unit testing](#python-unit-testing)
  - [Better local unit tests with `pytest`](#better-local-unit-tests-with-pytest)
  - [Local linting](#local-linting)
- [Writing Documentation](#writing-documentation)
  - [Building documentation](#building-documentation)
    - [Tips](#tips)
  - [Previewing changes locally](#previewing-changes-locally)
  - [Adding documentation tests](#adding-documentation-tests)
<!-- tocstop -->

## Contributing to PyTorch Connectomics

Thank you for your interest in contributing to PyTorch Connectomics! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature in an [issue](https://github.com/zudi-lin/pytorch_connectomics/issues),
    and we shall discuss the design and implementation. Once we agree that the plan looks good,
    go ahead and implement it.

2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the [Pytorch Connectomics issue list]([https://github.com/zudi-lin/pytorch_connectomics/issues](https://github.com/zudi-lin/pytorch_connectomics/issues)).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please submit a **pull request (PR)** to
https://github.com/zudi-lin/pytorch_connectomics/pulls.


## Developing PyTorch Connectomics

To develop PyTorch Connectomics on your machine, here are some tips:

I. Clone a copy of PyTorch Connectomics from source:

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics
cd pytorch_connectomics
```

II. If you already have PyTorch Connectomics from source, update it:

```bash
git pull --rebase
git submodule sync --recursive
git submodule update --init --recursive --jobs 0
```

If you want to have no-op incremental rebuilds (which are fast), see the section below titled "Make no-op build fast."

III. Install PyTorch Connectomics in `develop` mode:

The change you have to make is to replace

```bash
python setup.py install
```

with

```bash
python setup.py develop
```

This mode will symlink the PyTorch Connectomics files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall PyTorch again and again.  This is especially useful if you are only changing Python files.

For example:
- Install local PyTorch Connectomics in `develop` mode
- Modify your PyTorch Connectomics file `connectomics/__init__.py` (for example)
- Test functionality

### Tips and Debugging
* Our `setup.py` requires Python >= 3.7
* If a commit is simple and doesn't affect any code (keep in mind that some docstrings contain code
  that is used in tests), you can add `[skip ci]` (case sensitive) somewhere in your commit message to
  [skip all build / test steps](https://github.blog/changelog/2021-02-08-github-actions-skip-pull-request-and-push-workflows-with-skip-ci/).
  Note that changing the pull request body or title on GitHub itself has no effect.
* If you run into errors when running `python setup.py develop`, here are some debugging steps:
  1. Run `printf '#include <stdio.h>\nint main() { printf("Hello World");}'|clang -x c -; ./a.out` to make sure
  your CMake works and can compile this simple Hello World program without errors.
  2. Nuke your `build` directory. The `setup.py` script compiles binaries into the `build` folder and caches many
  details along the way, which saves time the next time you build. If you're running into issues, you can always
  `rm -rf build` from the toplevel `pytorch_connectomics` directory and start over.
  3. If you have made edits to the PyTorchConnectomics repo, commit any change you'd like to keep and clean the repo with the
  following commands (note that clean _really_ removes all untracked files and changes.):
  ```bash
  git submodule deinit -f .
  git clean -xdf
  python setup.py clean
  git submodule update --init --recursive --jobs 0 # very important to sync the submodules
  python setup.py develop                          # then try running the command again
  ```
  4. The main step within `python setup.py develop` is running `make` from the `build` directory. If you want to
  experiment with some environment variables, you can pass them into the command:
  ```bash
  ENV_KEY1=ENV_VAL1[, ENV_KEY2=ENV_VAL2]* python setup.py develop
  ```

## Codebase structure

* [connectomics](connectomics) - The actual Pytorch Connectomics library.
    * [data](connectomics/data) - Dataset and dataloader for large-scale volumetric data
    * [model](connectomics/model) - Model zoo for 3D segmentation
    * [engine](connectomics/engine) - Training and inference routines
    * [config](connectomics/config) - Configuration files for training and inference

* [tests](tests) - Python unit tests for PytorchConnectomics Python frontend.
* [notebooks](notebooks) - Jupyter notebooks with visualization.
* [projects](projects) - Research projects that are built upon the repo.



## Writing documentation

So you want to write some documentation and don't know where to start?
Pytorchconnectomics has two main types of documentation:
- user-facing documentation.
These are the docs that you see over at [our docs website](https://pytorch.org/docs).
- developer facing documentation.
Developer facing documentation is spread around our READMEs in our codebase and in
the [PyTorch Developer Wiki](https://pytorch.org/wiki).
If you're interested in adding new developer docs, please read this [page on the wiki](https://github.com/pytorch/pytorch/wiki/Where-or-how-should-I-add-documentation%3F) on our best practices for where to put it.

The rest of this section is about user-facing documentation.

Pytorchconnectomics uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to
fit into Jupyter documentation popups.

### Building documentation

To build the documentation:

1. Build and install Pytorchconnectomics

2. Install the prerequisites

```bash
cd docs
pip install -r requirements.txt
# `katex` must also be available in your PATH.
# You can either install katex globally if you have properly configured npm:
# npm install -g katex
# Or if you prefer an uncontaminated global executable environment or do not want to go through the node configuration:
# npm install katex && export PATH="$PATH:$(pwd)/node_modules/.bin"
```

> Note that if you are a Facebook employee using a devserver, yarn may be more convenient to install katex:

```bash
yarn global add katex
```

3. Generate the documentation HTML files. The generated files will be in `docs/build/html`.

```bash
make html
```

#### Tips

The `.rst` source files live in [docs/source](docs/source). Some of the `.rst`
files pull in docstrings from Pytorchconnectomics Python code (for example, via
the `autofunction` or `autoclass` directives). To vastly shorten doc build times,
it is helpful to remove the files you are not working on, only keeping the base
`index.rst` file and the files you are editing. The Sphinx build will produce
missing file warnings but will still complete. For example, to work on `jit.rst`:

```bash
cd docs/source
find . -type f | grep rst | grep -v index | grep -v jit | xargs rm

# Make your changes, build the docs, etc.

# Don't commit the deletions!
git add index.rst jit.rst
...
```

### Previewing changes locally

To view HTML files locally, you can open the files in your web browser. For example,
navigate to `file:///your_pytorch_connectomics_folder/docs/build/html/index.html` in a web
browser.

If you are developing on a remote machine, you can set up an SSH tunnel so that
you can access the HTTP server on the remote machine from your local machine. To map
remote port 8000 to local port 8000, use either of the following commands.

```bash
# For SSH
ssh my_machine -L 8000:my_machine:8000

# For Eternal Terminal
et my_machine -t="8000:8000"
```

Then navigate to `localhost:8000` in your web browser.

**Tip:**
You can start a lightweight HTTP server on the remote machine with:

```bash
python -m http.server 8000 <path_to_html_output>
```

Alternatively, you can run `rsync` on your local machine to copy the files from
your remote machine:

```bash
mkdir -p build cpp/build
rsync -az me@my_machine:/path/to/pytorch_connectomics/docs/build/html build
rsync -az me@my_machine:/path/to/pytorch_connectomics/docs/cpp/build/html cpp/build
```

### Adding documentation tests

It is easy for code snippets in docstrings and `.rst` files to get out of date. The docs
build includes the [Sphinx Doctest Extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html),
which can run code in documentation as a unit test. To use the extension, use
the `.. testcode::` directive in your `.rst` and docstrings.

To manually run these tests, follow steps 1 and 2 above, then run:

```bash
cd docs
make doctest
```
