
![](docs/_static/mutis.png)

# MUTIS: MUltiwavelength TIme Series

A Python package for the analysis of correlations of light curves and their statistical significance.

### Installation
- Install [Anaconda](https://www.anaconda.com/download/ )
- Clone this repository
- Run the commands

```
conda env create  -f environment.yml
conda activate mutis
pip install .
```

## Contribute
You are welcome to use and contribute to **MUTIS**! You can create a fork and do a PR from there. Before you do so, we suggest that you install `nbstripout` (`conda install -c conda-forge nbstripout`) and add the following lines to your repo configuration:

In file `.git/info/attributes` add:
```
*.ipynb filter=nbstripout
*.zpln filter=nbstripout
*.ipynb diff=ipynb
```
In file `.git/config` add:
```
[filter "nbstripout"]
        clean = nbstripout
        smudge = cat
        extrakeys = metadata.kernelspec
[diff "ipynb"]
        textconv = nbstripout -t
```
Now `git` will not show cell output in diffs, and will not include cell output to commits; it will make working working with the notebooks much easier.

This filters won't modify or clear the files, it will only be visible to git. If you want to clear the files also, you can use `nbstripout docs/recipes/*.ipynb --extra-keys "metadata.kernelspec"`.

### Documentation
- https://mutis.readthedocs.io/

### Status shields
- [![Docs Built](https://github.com/IAA-CSIC/MUTIS/workflows/CI/badge.svg)](https://github.com/IAA-CSIC/MUTIS/actions)
- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IAA-CSIC/MUTIS.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IAA-CSIC/MUTIS/context:python)
- [![codecov](https://codecov.io/gh/IAA-CSIC/MUTIS/branch/main/graph/badge.svg?token=8Q38S24P2J)](https://codecov.io/gh/IAA-CSIC/MUTIS)
- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
- [![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/IAA-CSIC/MUTIS/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/IAA-CSIC/MUTIS/?branch=main)