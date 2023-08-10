import logging

from mpi4py import MPI
from datetime import date
import time
import os
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

class Config:
    def __init__(self, nameTest:str, model:str, dataset:str, lam:float, gamma:float, max_depth:int, max_tree:int, nClasses:int, nFeatures:int, nBuckets:int):
        self.nameTest = nameTest
        self.model = model
        self.dataset = dataset
        self.lam = lam
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_tree = max_tree
        self.nClasses = nClasses
        self.nFeatures = nFeatures
        self.nBuckets = nBuckets

    def prettyprint(self):
        print(f"test: {self.nameTest}")
        print(f"model: {self.model}")
        print(f"dataset: {self.dataset}")
        print(f"nClasses: {self.nClasses}")
        print(f"nFeatures: {self.nFeatures}")
        print(f"lambda:{self.lam}")
        print(f"gamma: {self.gamma}")
        print(f"max_depth: {self.max_depth}")
        print(f"max_tree: {self.max_tree}")

class MyLogger:
    def __init__(self, config:Config):

        logger = logging.getLogger()
        day = date.today().strftime("%b-%d-%Y")

        curTime = round(time.time())

        logName = 'Log/{}/{}/{}_{}/Rank_{}.log'.format(config.nameTest, str(day), str(curTime), str(config.model), str(rank))
        os.makedirs(os.path.dirname(logName), exist_ok=True)

        file_handler = logging.FileHandler(logName, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
        formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.WARNING)
        self.logger = logger