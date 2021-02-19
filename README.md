# Multi-Level Co-Surrogates

[![build status][build-badge]][actions-page]
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/256506176.svg)](https://zenodo.org/badge/latestdoi/256506176)


## Introduction

In simulation-based optimization, multiple (distinct) accuracy or fidelity levels are commonly
available, which make the trade-off between high accuracy and fast computation time. Hierarchical or
co-surrogate models present a way to combine the information from these multiple fidelity levels to
reduce computation cost, increase model accuracy or both simultaneously.

This work focusses on an empirical investigation of the behavior of this trade-off for various
multi-fidelity benchmark functions. This is done by training hierarchical surrogate models with
varying numbers of high- and low-fidelity input samples, and displaying the collection of model
accuracy values as 'Error Grids'.

This repository is the work of Sander van Rijn, PhD candidate at the Leiden Institute of Advanced
Computer Science, Leiden University, the Netherlands.


## Installation

The easiest way to work with this code is as follows:

* First clone clone the repository

```bash
git clone https://github.com/sjvrijn/multi-level-co-surrogates.git mlcs
cd mlcs
```

* Second, to make sure everything works as intended and nothing conflicts with any previously
  installed packages, create and activate a clean environment. To use the Python  built-in `venv`
  command, use the following commands:

Create the environment

```bash
python3 -m venv mlcs_env
```

And then activate it according to your operating system

```bash
# Linux/Unix bash
source mlcs_env/bin/activate

# Windows cmd.exe
mlcs_env\Scripts\activate.bat

# Windows PowerShell
mlcs_env\Scripts\Activate.ps1
```

* Finally install the requirements. If you wish to run the included tests, replace
  `requirements.txt` with the extended `requirements-dev.txt` in the following command:

```bash
python3 -m pip install -r requirements.txt
```

* Additionally, you may install `bottleneck` to speed up some computations, but this package can
  cause issues on Windows systems (see [this issue][bottleneck-issue])

```bash
python3 -m pip install bottleneck
```

To test whether installation has been succesful, you can run one of the standalone illustration
scripts, which should generate the figure
`mlcs/plots/2020-07-07-intercept-illustration/budget-extension-intercept.png`:

```bash
cd scripts/processing
python3 2020-07-07-intercept-illustration.py
```


## Usage

To run all experiments or all processing scripts, you can use the `runall.sh` bash scripts in the
scripts folders. **Note that most processing scripts depend on data being present, and that the
experiment scripts may take weeks to complete.**

```bash
cd scripts/experiments
./runall.sh
```

It is advised to adjust the scripts to only generate the exact data needed, or to first download the
data files available on [Zenodo] (see Related Work).

Alternatively you can run the scripts individually based on your needs, for example:

```bash
cd scripts/experiments
python3 2019-09-01-error-grids.py 1  # only run case index '1', will take ~1 day
```

For details on what each script does, please read the docstring at the top of each file. Any
optional commandline arguments are also listed there.


## Content

### multiLevelCoSurrogates

Contains the implementation of the hierarchical co-surrogates, CandidateArchive class, (initial)
multi-fidelity bayesian optimization implementation and further utility functions.


### notebooks

All notebooks that were interactively created during the research process.
They may not be reproducible and are either obsolete, or have been refactored into proper functions
or scripts elsewhere in this repository. Their main purpose is to serve as a historic archive.


### scripts

Experiment and post-processing scripts. The scripts in `experiments` create the data files and will
store them in the `files/` folder, while the scripts in `processing` use the data files from
`files/` and store the results in `plots/`. Both folders will be created automatically if they
do not yet exist. Details on each script can be found in the file's docstring.


### tests

Test file(s), specifically for the custom `CandidateArchive` class. Note that the packages in
`requirements-dev.txt` have to be installed to run them.


## Contact

If you encounter problems or have any questions about this code, please
[raise an issue][new-issue] in this
repository or send an [email](mailto-svrijn).


## Related Work

- [mf2]: collection of benchmark functions used in this package
- Data files archive on [Zenodo] [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4551287.svg)](https://doi.org/10.5281/zenodo.4551287)


## Citation

If this code has been useful in your research, please cite it:

```bibtex
@software{vanRijn2021-github,
    author = {van Rijn, Sander},
    title = {Github repository: Multi-Level Co-Surrogates},
    year = {2021},
    url = {https://github.com/sjvrijn/multi-level-co-surrogates},
}
```

## Acknowledgements

This work is part of the research program DAMIOSO with project number 628.006.002, which is partly
financed by the Netherlands Organisation for Scientific Research (NWO).



[actions-page]:     https://github.com/sjvrijn/multi-level-co-surrogates/actions
[bottleneck-issue]: https://github.com/pydata/bottleneck/issues/281
[build-badge]:	    https://github.com/sjvrijn/multi-level-co-surrogates/workflows/build/badge.svg
[mf2]:              https://github.com/sjvrijn/mf2
[mailto-svrijn]:    mailto:s.j.van.rijn@liacs.leidenuniv.nl
[new-issue]:        https://github.com/sjvrijn/multi-level-co-surrogates/issues/new
[Zenodo]:           https://doi.org/10.5281/zenodo.4551287

