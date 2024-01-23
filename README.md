# bio-deep-ncpca

Biologically plausible, deep, non-negative, contrastive PCA

## Setup

Create the Conda environment using

    conda env create -f environment.yml

Then activate the environment

    conda activate ncpca

and make an editable install of the `ncpca` package using

    pip install -e .

Make sure that things work by running

    pytest tests
