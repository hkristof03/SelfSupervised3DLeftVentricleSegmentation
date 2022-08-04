# Volumetric left ventricle segmentation of MPI SPECT images with self-supervised learning
Master's Thesis

The quantification accuracy of the left ventricular myocardium volume from the Myocardial 
Perfusion SPECT (MPS) images directly relates to the fidelity of the left ventricular function 
assessment. The manual segmentation is tedious and depends on the observer’s expertise. 
Many supervised machine learning procedures require large number of labels, which further
require human involvement and in the medical field specialized experts. The hypothesis states 
that it is possible to train a general feature detector for left ventricle segmentation with 
Self-Supervised Learning (SSL) algorithms that reduce the label requirement of neural networks. 
Extended encoders of 3D U-Net architectures were trained on the Relative Patch Location and 
Jigsaw Puzzle pretext tasks, and the latter proved to be highly efficient to extract general 
feature representations from SPECT volumes. The pre-trained models were fine-tuned in a 
supervised manner (on data of 6 patients only) to provide accurate 3D segmentations for the 
left ventricle of the human heart.

Project Organization
------------

    ├── readme.md                      
    ├── Thesis                                  <- Master's thesis and presentations for defence
    ├── experiments                             <- YAML file based configuration files to run experiments
    ├── notebooks                               <- Jupyter playbooks for initial experiments and code writing
    ├── logger.py                               <- Basic logger for Neptune.ml
    ├── models.py                               <- Model definitions for both SSL and supervised models
    ├── ssl_jigsaw_puzzle.py                    <- SSL algorithm to solve jigsaw puzzles pretext tasks
    ├── ssl_relative_patch_location.py          <- SSL algorithm to solve the relative patch location pretext tasks
    ├── supervised_training.py                  <- algorithm for supervised training on the target task after SSL
    ├── utils.py                                <- utility functions
