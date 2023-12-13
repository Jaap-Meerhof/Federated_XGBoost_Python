# Federated Membership Inference on Federboost
This package implements 'Federboost' from Tian *et al* 
Which is a horizontally federated Gradient Boosted Decision Tree algorithm. With the regularisation terms of **XGBoost**. This is not a 1 to 1 conversion from XGBoost to python, the models perform remarkably close when regularisation parameters are used, but they differ more when it is not used. 

This is a multi-class classification implementation that uses softmax. Also different types of Membership Inference Attack can be tested, both using federated information or without.

[Get the paper here](https://essay.utwente.nl/97818/)

TODO: 

- Clear up for other researchers
   * code cleanup & explanation
- Change naming from old SFXGBoost to FXGBoost

---
Usage:
install conda environment:
```
conda env create -f environment.yml
```
Activate your environment:
```
conda activate fedxgboost_mpi
```

build and run the experiments using:
```
pip install . && mpiexec -np 3 python tests/main.py (experiment number 1 to 3)
```
here ```-np 3``` implies 1 server and two participants in the federated network.
for this you will have to change the ```POSSIBLE_PATHS``` parameter for the healthcare dataset.

or use "run.sh":
```
./run.sh myexperimentname (experiment number 1 to 3)
```
---

[Tian, Z., Zhang, R., Hou, X., Liu, J., & Ren, K. (2020). Federboost: Private federated learning for gbdt. arXiv preprint arXiv:2011.02796.](https://arxiv.org/abs/2011.02796)

---


Made in collaboration with the RIVM (Dutch National Institute for Public Health and the Environment) for my master thesis at the University of Twente. 

![Made in collaboration with the RIVM (Dutch National Institute for Public Health and the Environment)](https://github.com/Jaap-Meerhof/Federated_XGBoost_Python/blob/main/assets/RIVM_logo_big.png)
---

check https://jaap-meerhof.github.io for my contact information
