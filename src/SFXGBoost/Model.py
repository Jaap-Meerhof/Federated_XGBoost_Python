from config import config
import numpy as np
from data_structure.treestructure import FLTreeNode
from copy import deepcopy

class SFXGBoostClassifierBase:
    def __init__(self, treeSet) -> None:
        self.trees: np.array(np.array(SFXGBoostTree)) = treeSet

class SFXGBoost(SFXGBoostClassifierBase):
    def __init__(self) -> None:
        trees = [[] for _ in range(config.nClasses)]
        for nClass in range(config.nClasses):
            trees[nClass] = [SFXGBoostTree(id) for id in range(config.max_tree)]
        trees = np.array(trees)
        super().__init__(trees)
    
    def boost(self, init_Probas):
        self.init_Probas = init_Probas
        orgData = deepcopy(self.dataBase)

class SFXGBoostTree:
    def __init__(self, id) -> None:
        self.id = id
        self.root = FLTreeNode()
        self.nNode = 0
        
        pass
