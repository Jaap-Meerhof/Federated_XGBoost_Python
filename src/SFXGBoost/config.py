import logging

from mpi4py import MPI
from datetime import date
import time
import os
import numpy as np
from SFXGBoost.dataset.datasetRetrieval import getConfigParams

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

class Config:
    def __init__(self, experimentName:str, nameTest:str, model:str, dataset:str, lam:float, gamma:float, 
                 alpha:float, learning_rate:float, max_depth:int, max_tree:int, nBuckets:int, save:bool=True, 
                 target_rank:int=0, data_devision:list=[0.5, 0.5], train_size:int=2000):
        self.experimentName = experimentName
        self.nameTest = nameTest
        self.model = model
        self.dataset = dataset
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_tree = max_tree
        self.nClasses, self.nFeatures = getConfigParams(self.dataset)
        self.nBuckets = nBuckets
        self.save = save
        self.target_rank = target_rank  # Participant's ID to attack for experiment 2
        self.data_devision = data_devision
        # assert (0.99 <= sum(self.data_devision)) and (sum(self.data_devision) <= 1) and len(self.data_devision) == comm.Get_size -1
        self.train_size = train_size

        # self.save_location= "./Saves/" + nameTest + "_rank_" + str(rank)
        self.save_location= "/mnt/scratch_dir/meerhofj/Saves/" + nameTest + "_rank_" + str(rank)

    def prettyprint(self):
        print(f"experiment name = {self.experimentName}")
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

        # curTime = round(time.time())
        curTime = time.strftime("%H:%M", time.localtime())

        logName = 'Log/{}/{}/{}_{}/Rank_{}.log'.format(config.nameTest, str(day), str(curTime), str(config.model), str(rank))
        os.makedirs(os.path.dirname(logName), exist_ok=True)

        file_handler = logging.FileHandler(logName, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
        formatter = logging.Formatter('%(levelname)s - [%(filename)s:%(lineno)s - %(funcName)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.WARNING)
        self.logger = logger