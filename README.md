# active-client-selection-framework
Federated learning framework for active client selection method.

# Remain Works
- [] Implement non-iid dataset split and pytorch Data
- [] Implement FL algorithms
- [] Implement active client selection methods
- [] Tensorboard logging
- [] Result visualization 

# Requirements

CUDA 11.3 

```shell
    conda env create --file environment.yaml
```

# Experiment

## Preprocess

Make particion of each client

## Client selection experiment

### Arguments

# Visualization

# Reference

## LEAF

The main implementation based on LEAF project.

[Project page](https://leaf.cmu.edu/)
[Git repo](https://github.com/TalwalkarLab/leaf)

```
@article{DBLP:journals/corr/abs-1812-01097,
  author    = {Sebastian Caldas and
               Peter Wu and
               Tian Li and
               Jakub Kone{\v{c}}n{\'y} and
               H. Brendan McMahan and
               Virginia Smith and
               Ameet Talwalkar},
  title     = {{LEAF:} {A} Benchmark for Federated Settings},
  journal   = {CoRR},
  volume    = {abs/1812.01097},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.01097},
  eprinttype = {arXiv},
  eprint    = {1812.01097},
  timestamp = {Wed, 23 Dec 2020 09:35:18 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1812-01097.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## EMNIST

[The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
