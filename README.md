# Repository to "Order Preservation in Dimensionality Reduction: Structure-dependent Embeddings in Real and Complex Space"

## Used datasets: 

- Russell2000 stock data: closing stock prices of (d=) 1795 stocks and (n=) 504 observations. After data cleaning there are 1549 stocks [Source](https://data.mendeley.com/datasets/ndxfrshm74/1)

- NASA MTS flight data: 10 flights with (d=) 29 features and minimally 4662 observations [Source](https://data.nasa.gov/dataset/Multivariate-Time-Series-Search/mrcc-b53g)
- Air Pollution in Japan : we used 4380 observations [Source](https://data.mendeley.com/datasets/phgrnvykmr/1)

## SLURM
1. Start a terminal of your choice
2. Copy ./src and ./pickled_results to your cip account
    * `scp -r ./src ./pickled_results <username>@remote.cip.ifi.lmu.de:~/<folder_name>`
3. Login to cip via SSH
    * `ssh <username>@remote.cip.ifi.lmu.de`
4. There are four important commands to use the slurm ressources:
    * `sinfo`: shows all computers that are available in the slurm engine
        * For now we should just concentrate on the ones with partition "All". Later we could specify a certain computer to use for training if we want to fine-tune.
    * `squeue`: shows all jobs
        * `squeue -u <username>`: shows all jobs for specified username
    * `sbatch`: runs a specified shell script. (**only shell scripts can be submitted!**)
        * `sbatch --partition=All --cpus-per-task=4 <script>.sh`: this is the usual usage of the command.
            * `--partition=All`: specifies that any available computer should be used
            * `--cpus-per-task=4`: specifies that for cpus should be used. If none is specified only one will be used
            * `--mem = 4600` : specifies the RAM memory needed
            * `<script>.sh`: the shell script that should be run
    * `scancel <process-ID>`: cancels the specified job
5. Get Logs: Log in to ssh and run `cat slurm-<slurm-jobID>.out`
6. Copy pickles of finished Slurm jobs to your own folder: 
    * `scp -r <username>@remote.cip.ifi.lmu.de:~/<folder_name>/pickled_results/* pickled_results/slurm/`

## Requirements
See and install requirements [requirements.txt](requirements.txt)
## Distance Measures:
- euclidean
- dtw
- scalar_product: k = 1, without variance
- multiple_scalar_product: up to k, with variance

## Orders
- zigzag
- one_line
- sep_lines -> use sep_measure == True in config, if lines should be measured separately
- random
- spiral
- staircase
- swiss_roll
- parallel_lines
- connected_parallel_lines
- connected_zigzag
- flights: Not generated -> Before first usage download data to folder "data" -> use sep_measure == True in config, if flights should be measured separately
- air_pollution: Not generated -> Before first usage download data to folder "data"
- russell2000_stock: Not generated -> Before first usage download data to folder "data"

## PCA options:
- **complex_pca**: Applies PCA to the whole complex data
- **complex_pca_svd_real**: Centres only the real part of the data and applies PCA only on the real part of the data. It uses the resulting principal components to transform the complex data
- **complex_pca_svd_complex**: Centres only real part of the data but applies PCA onto this new partially centred data. Only the real part of the principal components is being used for transformation
- **complex_pca_svd_complex_imag_pc**: Centres only real part of the data but applies PCA onto this new partially centred data. Only the imaginary part of the principal components is being used for transformation

## Scaling options:
- like_original_pca
- like_vector_pca
- low_imag

## Demo Plots
To obtain the plots for used for demonstration of the orders, for instance, uncomment wanted function in [src/demo_main.py](./src/demo_main.py).
Additionally, set parameters in [src/config.json](./src/config.json).
Run 
```console
    python src/demo_main.py
```
in your console.

## Experiments with CPCA variants
Modify [src/config.json](./src/config.json) and uncomment wanted function for plotting or computing of new results in [src/cpca_main.py](./src/cpca_main.py).
Run 
```console
    python src/cpca_main.py
```
in your console.

## Experiments with PCA* on real and generated data
All results are pickled. 

Modify [src/config.json](./src/config.json) and uncomment wanted function for plotting or computing of new results in [src/main.py](./src/main.py).
Run 
```console
    python src/main.py
```
in your console.