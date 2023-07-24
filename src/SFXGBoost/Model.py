from SFXGBoost.config import rank, comm, Config
from logging import Logger
import numpy as np
from SFXGBoost.data_structure.treestructure import FLTreeNode
from copy import deepcopy
from SFXGBoost.data_structure.databasestructure import QuantiledDataBase, DataBase
from SFXGBoost.common.XGBoostcommon import PARTY_ID
from SFXGBoost.loss.softmax import getGradientHessians
class MSG_ID:
    TREE_UPDATE = 69
    RESPONSE_GRADIENTS = 70
    SPLIT_UPDATE = 71

class SFXGBoostClassifierBase:
    def __init__(self, config:Config, logger:Logger, treeSet) -> None:
        self.trees: np.array(np.array(SFXGBoostTree)) = treeSet
        self.config = config
        self.logger = logger
    
    def setData(self, quantileDB:QuantiledDataBase=None, fName = None, original_data:DataBase=None, y=None):
        self.quantileDB = quantileDB
        self.original_data = original_data
        self.fName = fName
        self.nFeatures = len(fName)
        self.y = y
        nUsers = original_data.nUsers
        self.nUsers = nUsers
        for treesofclass in self.trees:
            for tree in treesofclass:
                tree.setInstances = np.full((nUsers, ), True) # sets all root nodes to have all instances set to True

class SFXGBoost(SFXGBoostClassifierBase):
    def __init__(self, config:Config, logger:Logger) -> None:
        trees = [[] for _ in range(config.nClasses)]
        for nClass in range(config.nClasses):
            trees[nClass] = [SFXGBoostTree(id) for id in range(config.max_tree)]
        trees = np.array(trees)
        super().__init__(config, logger, trees)
    
    def setSplits(self, splits):
        self.splits = splits

    def boost(self, init_Probas):
        self.init_Probas = init_Probas
        ####################
        ###### SERVER ######
        ####################
        if rank == PARTY_ID.SERVER:
            for t in range(self.config.max_tree):
                nodes = [[self.trees[c][t].root] for c in range(self.config.nClasses)] 
                for l in range(self.config.max_depth):
                    G = [[ [] for _ in range(len(nodes[c])) ] for c in range(self.config.nClasses)]
                    H = [None, None]
                    for Pi in range(comm.Get_size()):
                        GH = comm.recv(source=Pi, tag=MSG_ID.RESPONSE_GRADIENTS) # receive [nClasses][nodes][]
                        for c in range(self.config.nClasses):
                            for i, n in enumerate(nodes[c]):
                                Gpi = GH[0]
                                Hpi = GH[1]
                                if G[c][i] == []:
                                    G[c][i] =  Gpi[c][i]
                                    H[c][i] =  Hpi[c][i]
                                else:
                                    G[c][i] += Gpi[c][i]
                                    H[c][i] += Hpi[c][i]
                    for c in range(self.config.nClasses):
                        for i, n in enumerate(nodes[c]):
                            split_cn = find_split(self.spits, G[c][i], H[c][i])
                    
                    for Pi in range(comm.Get_size()):
                        comm.send(split_cn, Pi, tag=MSG_ID.SPLIT_UPDATE )
                    # update own my own trees to attack users
        
        ####################
        ### PARTICIPANTS ###
        ####################
        else:
            orgData = deepcopy(self.original_data)
            y_pred = np.tile(init_Probas, (self.nUsers, 1)) # nClasses, nUsers
            y = self.y
            instances = [np.full((self.original_data.nUsers,), True)]
            for t in range(self.config.max_tree):
                G, H = getGradientHessians(y, y_pred)
                nodes = [[self.trees[c][t].root] for c in range(self.config.nClasses)] 
                for l in range(self.config.max_depth):
                    Gnodes = [[] for _ in range(self.config.nClasses)]
                    Hnodes = [[] for _ in range(self.config.nClasses)]

                    for c in range(self.config.nClasses):
                        for node in nodes[c]:
                            instances = node.instances
                            gcn, hcn, dx = self.appendGradients(instances, G, H, orgData)
                            Gnodes[c].append(gcn)
                            Hnodes[c].append(hcn)

                    comm.send((Gnodes,Hnodes), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)
                    splits = comm.recv(source=PARTY_ID.SERVER, tag=MSG_ID.SPLIT_UPDATE)
                    nodes = self.update_trees(nodes, splits) # also update Instances
                
                update_pred = np.array([tree.predict(orgData) for tree in self.trees[:, t]]).T
                y_pred += update_pred
     
    
    def update_trees(self, nodes:list(list(FLTreeNode)), splits):
        for c in range(self.config.nClasses):
            for n, node in enumerate(nodes[c]):
                splitInfo = splits[c][n]
                node.splittingInfo = splitInfo
                node.leftBranch = FLTreeNode()
                node.rightBranch = FLTreeNode()
                fName = splitInfo.featureName
                sValue = splitInfo.splitValue

                node.leftBranch.instances = [self.original_data[fName][index] < sValue for index in self.nUsers]
                node.rightBranch.instances = [self.original_data[fName][index] >= sValue for index in self.nUsers]



    def appendGradients(self, instances, G, H, orgData):
        Gkv = []  #np.zeros((self.nClasses, amount_of_bins))
        Hkv = []
        Dx = []  # the split the data corresponds to, this means the split left of the value. If there is no split with a smaller value than take the smalles split value
        k = 0
        for fName, fData in self.qDataBase.featureDict.items():
            splits = self.qDataBase.featureDict[fName].splittingCandidates
            # Dxk = np.zeros((np.shape(splits)[0] + 1, ))
            Gk =  np.zeros((np.shape(splits)[0] + 1, ))
            Hk =  np.zeros((np.shape(splits)[0] + 1, ))

            data = orgData.featureDict[fName]
            gradients = G[k, :]
            hessians = H[k, :]

            # append gradient of corresponding data to the bin in which the data fits.                                 
            bin_indices = np.searchsorted(splits, data) -1 # indices 
            bin_indices = [0 if x==-1 else x for x in bin_indices] # replace -1 with 0 such that upper most left values gets assigned to the right split. 
            
            qData = [np.inf if x == len(splits) + 1 else x for x in np.searchsorted(splits, data)]

            Dx.append(qData)

            for index in range(np.shape(data)[0]):
                bin = bin_indices[index]
                if instances[index]: # if instances[index] is true, then it is still a valid row.
                    Gk[bin] += gradients[index]
                    Hk[bin] += hessians[index]
            
            Gkv.append(Gk)
            Hkv.append(Hk)

            k -=- 1
        return Gkv, Hkv, Dx
        # comm.send((Gkv, Hkv, Dx), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)

    def predict(self, X): # returns class number
        y_pred = self.predictweights(X) # get leaf node weights
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X): # returns probabilities of all classes
        y_pred = self.predict(X) # returns weights
        for rowid in range(np.shape(y_pred)[0]):
            row = y_pred[rowid, :]
            wmax = max(row)
            wsum = 0
            for y in row: wsum += np.exp(y-wmax)
            y_pred[rowid, :] = np.exp(row-wmax) / wsum
        return y_pred


    def predictweights(self, X): # returns weights
            
            if type(X) == np.ndarray:
                X = toDataBase(X)


            # y_pred = [None for n in range(self.nClasses)]
            y_pred = np.tile(self.init_Probas, (len(X) , 1)) #(Nclasses, xrows)
            data_num = X.shape[0]
            # Make predictions
            testDataBase = DataBase.data_matrix_to_database(X, self.fName)
            print(f"DEBUG: {np.shape(X)}, {np.shape(y_pred)}, {np.shape(self.init_Probas)}, {np.shape(self.trees)}")
            print(f"{self.init_Probas}")
            for treeID in range(self.nTree):
                for c in range(self.nClasses):
                    tree = self.trees[c][treeID]
                    # Estimate gradient and update prediction
                    self.logger.warning(f"PREDICTION id {treeID}")
                    b = FLVisNode(tree.root)
                    b.display(treeID)

                    update_pred = tree.predict(testDataBase)
                    if y_pred[c] is None:
                        y_pred[c] = self.init_Probas[c] # TODO replace with initprobas
                    if rank == 0: # hier gaat ie fout
                        # update_pred = np.reshape(update_pred, (data_num, 1))
                        # print(f"{np.shape(y_pred[:, c])}, {np.shape(update_pred)}")
                        y_pred[:, c] += update_pred
            return y_pred

             

class SFXGBoostTree:
    def __init__(self, id, nUsers) -> None:
        self.id = id
        self.root = FLTreeNode(nUsers=nUsers)
        self.nNode = 0
        
    def setInstances(self, instances):
        self.root.instances = instances
