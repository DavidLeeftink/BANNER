# BANNER

<p align="center">    
<img src="https://user-images.githubusercontent.com/39411160/107499592-6db7e700-6b95-11eb-8acd-21979d91c82f.png" width="600">

    
### Bayesian Non-parametric Network Regression using variational Generalized Wishart Processes
This project consists of a gpflow 2 implementation of the [variational Generalized Wishart Process](https://arxiv.org/pdf/1906.09360.pdf)
, based on the [Generalized Wishart Process](https://arxiv.org/pdf/1101.0240.pdf). The implementation consists of the exact Wishart Process model and likelihood, as well as the factorized approximation. In addition, a multi output kernel is added which allows several input channels to share the same kernel (and thus learn the same lengthscale).


### Contact
- [Max Hinne](mailto:max.hinne@donders.ru.nl?subject=[GitHub]%BANNER)
- [Hester Huijsdens](mailto:hester.huijsdens@donders.ru.nl?subject=[GitHub]%BANNER)
- [David Leeftink](mailto:david.leeftink@ru.nl?subject=[GitHub]%BANNER)

### Package requirements
package | version
--------|----------
gpflow  | 2.1.4
tensorflow | 2.5.0
tensorflow_probability | 0.12.1
tensorboard | 2.5.0
matplotlib | 3.3.2
numpy | 1.19.2
h5py | 3.1.0
scikit-learn | 0.24.2
pandas | 1.3.2
tqdm | 4.62.3

### Project structure
    ├── data                    # Folder for offline data
    ├── logs                    # Saving trained models and training logs     
    ├── analyses                # Training scripts and jupyter notebook examples
    ├── src                     # Source files
    │   ├── models   
    |   ├── kernels   
    |   ├── likelihood   
    └── README.md	
    
    
### Tensorboard
Run "tensorboard --logdir logs/" in command line	
