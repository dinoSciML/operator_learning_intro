# Linear Elasticity Neural Operator Tutorial

## Installation

E.g., using `conda`

```
conda create -n torchfem -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter 
pip install torch
pip install tqdm
```

Use `pip` to install additional missing packages as needed


### `hippylib`

This tutorial uses the [`hippylib`](https://github.com/hippylib/hippylib/tree/master) package

Either `git clone` and set `HIPPYLIB_PATH` or use pip. I recommend `git`

### `hippyflow`

This tutorial uses the [`hippyflow`](https://github.com/hippylib/hippyflow/tree/master) package for data generation and dimension reduction

Use `git clone` and set `HIPPYFLOW_PATH`.

##

1. To see the setup for the parametric PDE map and deterministic inverse problem see `LinearElasticityIntro.ipynb`

2. To train neural operators (RBNO, FNO, DeepONet) use the following steps:

```
# Full-state
python generate_data.py # This generates the samples for the full-state training
python compute_coders.py # This computes the reduced bases (needed for RBNO only)
python reduce_data.py # This encodes the training data onto the reduced bases (needed for RBNO only)

# Observables only
python generate_data.py -output_type observable # This generates the samples for the RB observable networks
python compute_coders.py -data_dir data/observable/ # This computes the reduced bases (needed for RBNO only)
python reduce_data.py -data_dir data/observable/ -output_basis None # This encodes the training data onto the reduced bases (needed for RBNO only)

# To train all the networks:
python training_loop.py # This trains a bunch of neural operators
```

or simply execute

```
python run_everything.py
```

For more information on the training see the notebooks `LinearElasticityRBNO.ipynb`, `LinearElasticityFNO.ipynb`, `LinearElasticityDON.ipynb`

3. To visualize the results of the training for inverse problems, see the `LinearElasticityIPComparisons.ipynb` notebook. 