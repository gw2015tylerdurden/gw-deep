# GravitySpy Dataset
   - https://zenodo.org/record/1476156
     - trainingsetv1d0.tar.gz

   - https://www.kaggle.com/tentotheminus9/gravity-spy-gravitational-waves
     - archive.zip

---

# Setup dependencies

**Create new conda environment and install all dependencies**

```
$ conda create -n <new_env> python=3.7 --file requirements.txt -c conda-forge -c pytorch
$ conda install pandas seaborn; conda install -c conda-forge wandb; pip insrall hashids
```

---
or install into <existing_env>. (python >= 3.6)

```
$ conda activate <existing_env>; conda install --file requirements.txt -c conda-forge -c pytorch
```

# Dataset To HDF5 format

```
$ python scripts/setup_dataset.py [dataset dir] output_filename.h5
```

# config file
you should to configure {vae, iic}.yaml before run. 

# Run vae
```
$ python run_vae.py
```

## Evaluate vae
```
$ python eval_vae.py
```

# Run IIC
```
$ python run_iic.py
```

## Evaluate IIC
```
$ python eval_vae.py
```
