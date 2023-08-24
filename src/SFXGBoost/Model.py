from SFXGBoost.config import rank, comm, Config
from logging import Logger
import numpy as np
from SFXGBoost.data_structure.treestructure import FLTreeNode, SplittingInfo
from copy import deepcopy
from SFXGBoost.data_structure.databasestructure import QuantiledDataBase, DataBase
from SFXGBoost.common.XGBoostcommon import PARTY_ID, L, Direction, weights_to_probas
from SFXGBoost.loss.softmax import getGradientHessians
from SFXGBoost.view.TreeRender import FLVisNode
from typing import List

class MSG_ID:
    TREE_UPDATE = 69
    RESPONSE_GRADIENTS = 70
    SPLIT_UPDATE = 71
    INITIAL_QUANTILE_SPLITS = 72
    Quantile_QJ = 86
    Quantile_nPrime_i = 89

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

        if rank != PARTY_ID.SERVER:
            nUsers = original_data.nUsers
            self.nUsers = nUsers
            for treesofclass in self.trees:
                for tree in treesofclass:
                    assert type(tree) == SFXGBoostTree
                    tree.setInstances(nUsers, np.full((nUsers, ), True)) # sets all root nodes to have all instances set to True
        
    def retrieve_differentials(self):
        """retrieves the G, H over the trees
        [G_tree1, G_tree2]
        where G_treei = [[root], [depth1, depth1], [depth2,..],...]
        H is done in the same fashion

        Returns:
            _type_: _description_
        """
        # self.trees take the gradients and hessians from all 
        return 

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
            splits = comm.recv(source=1, tag=MSG_ID.INITIAL_QUANTILE_SPLITS)
            self.logger.warning("splits:")
            for i, split_k in enumerate(splits):
                self.logger.warning(f"{self.fName[i]} = {split_k}")
            
            for t in range(self.config.max_tree):
                # print("#####################")
                print(f"> Busy with Tree {t} <")
                # print("#####################")

                nodes = [[self.trees[c][t].root] for c in range(self.config.nClasses)] 
                for l in range(self.config.max_depth+1):
                    G = [[ [] for _ in range(len(nodes[c])) ] for c in range(self.config.nClasses)]
                    H = [[ [] for _ in range(len(nodes[c])) ] for c in range(self.config.nClasses)]
                    for Pi in range(1, comm.Get_size()):
                        
                        GH = comm.recv(source=Pi, tag=MSG_ID.RESPONSE_GRADIENTS) # receive [nClasses][nodes][]

                        for c in range(self.config.nClasses):
                            for i, n in enumerate(nodes[c]):
                                Gpi = GH[0]
                                Hpi = GH[1]
                                if Pi == 1:                        
                                    G[c][i] =  Gpi[c][i]
                                    H[c][i] =  Hpi[c][i]  #TODO I should change the intial list to be zeros so I can just add
                                else:
                                    G[c][i] += Gpi[c][i]
                                    H[c][i] += Hpi[c][i]
                            # TODO save Gpi & Hpi for personalised attack
                            self.trees[c][t].gradients[l] = G[c] # keep track aggregated gradients
                            self.trees[c][t].hessians[l] = H[c] # keep track aggregated hessians

                    splittingInfos = [[] for _ in range(self.config.nClasses)] 
                    # print("got gradients")
                    for c in range(self.config.nClasses):
                        # def test(i):
                        #     print(f"working on node c={c} i={i}")
                        #     split_cn = self.find_split(splits, G[c][i], H[c][i], l+1 == self.config.max_depth)
                        #     splittingInfos[c].append(split_cn)
                        #     return None
                        # import concurrent.futures
                        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        #     results = list(executor.map(test, range(len(nodes[c]))))
                        # print(results)
                        for i, n in enumerate(nodes[c]): #TODO multithread!
                            split_cn = self.find_split(splits, G[c][i], H[c][i], l == self.config.max_depth)
                            splittingInfos[c].append(split_cn)
                    
                    for Pi in range(1, comm.Get_size()):
                        comm.send(splittingInfos, Pi, tag=MSG_ID.SPLIT_UPDATE )
                    nodes = self.update_trees(nodes, splittingInfos, l)
                    # update own my own trees to attack users
        
        ####################
        ### PARTICIPANTS ###
        ####################
        else:
            
            # new_splits = self.quantile_lookup()
            # print("DEBUG found splits")

            if rank == 1: # send found splits to server (doesn't need to be done)
                # def getSplits(quantileDB:QuantiledDataBase):
                splits = [ [] for _ in range(self.nFeatures)]
                i = 0
                for fName, quantileFeature in self.quantileDB.featureDict.items():
                    splits[i] = quantileFeature.splittingCandidates
                    i += 1

                comm.send(splits, PARTY_ID.SERVER, tag=MSG_ID.INITIAL_QUANTILE_SPLITS)
                self.logger.debug(f"splits are {splits}")

            orgData = deepcopy(self.original_data)
            # y_pred = np.tile(init_Probas, (self.nUsers, 1)) # nClasses, nUsers
            y_pred = np.zeros((self.nUsers, self.config.nClasses))
            y_pred = np.random.random(size=(self.nUsers, self.config.nClasses))

            y = self.y
            G, H = None, None
            for t in range(self.config.max_tree):
                G, H = getGradientHessians(np.argmax(y, axis=1), y_pred) # nUsers, nClasses
                G, H = np.array(G).T, np.array(H).T  # (nClasses, nUsers)
                nodes = [[self.trees[c][t].root] for c in range(self.config.nClasses)]
                
                for l in range(self.config.max_depth+1):
                    Gnodes = [[] for _ in range(self.config.nClasses)]
                    Hnodes = [[] for _ in range(self.config.nClasses)]

                    for c in range(self.config.nClasses):
                        for node in nodes[c]:
                            instances = node.instances
                            # print(instances)
                            gcn, hcn, dx = self.appendGradients(instances, G[c], H[c], orgData)
                            Gnodes[c].append(gcn)
                            Hnodes[c].append(hcn)
                    # send the gradients for every class's tree, the different nodes that have to be updated in that tree and the 
                    # print(f"sending gradients as rank {rank} on level {l}")
                    comm.send((Gnodes,Hnodes), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)
                    splits = comm.recv(source=PARTY_ID.SERVER, tag=MSG_ID.SPLIT_UPDATE)
                    nodes = self.update_trees(nodes, splits, l) # also update Instances
                for c in range(self.config.nClasses):
                    FLVisNode(self.logger, self.trees[c][t].root).display(t)
                update_pred = np.array([tree.predict(orgData) for tree in self.trees[:, t]]).T
                y_pred += update_pred #* self.config.learning_rate
    
    def quantile_lookup(self):
        """The different participants will run this algorithm to find the quantile splits.
        This algorithm DOES NOT actually run the secure aggregation. We simply assume it is secure for the research
        Outputs: Quantiles {Q_1,...,Q_{q-1}}
        """
        q = self.config.nBuckets
        X = self.original_data
        n = 50_000 # number of samples
        print(f"my users = {self.nUsers},  total = {n}")
        l = comm.Get_size()
        Pl = 1
        isPl = rank == Pl # active party 
        Q = np.zeros((self.nFeatures, q))
        other_users = [i for i in range(1, l) if i != rank]
        for featureid in range(self.config.nFeatures):
            features_i = deepcopy(X.featureDict[list(X.featureDict.keys())[featureid]])
            if len(np.unique(features_i)) < q:
                continue # not a continues feature to complete quantile_lookup.
            for j in range(1, q-1):
                Qj = None
                if isPl:
                    Qmin = np.min(features_i)
                    Qmax = np.max(features_i)
                nPrime = 0
                while np.abs(nPrime - (n/q)) > 0: # TODO catch loop
                    if isPl:
                        Qj = (Qmin + Qmax) / 2
                        for i in other_users:
                            comm.send(Qj, i, tag=MSG_ID.Quantile_QJ)
                    else:
                        Qj = comm.recv(source=Pl, tag=MSG_ID.Quantile_QJ)

                    nPrime_i = np.sum(features_i < Qj) #total nummber of local xs that are smaller than Qj
                    #secure aggregation part
                    nPrimes = np.zeros(l)
                    nPrimes[rank] = nPrime_i

                    for i in other_users: # 
                        comm.send(nPrime_i, i, MSG_ID.Quantile_nPrime_i)
                    for i in other_users: # 
                        nPrime_i = comm.recv(source=i, tag=MSG_ID.Quantile_nPrime_i)
                        nPrimes[i] = nPrime_i
                    nPrime = np.sum(nPrimes) # sum over recv
                    # print(f"nPrime = {nPrime} for user {rank}")
                    if isPl and nPrime > n/q:
                        Qmax = Qj
                    elif isPl and nPrime < n/q:
                        Qmin = Qj
                Q[featureid][j] = Qj
                print(f"my rank is {rank} with Qj = {Qj}")
                features_i = featureid[featureid < Qj] # remove the local xs that are smaller than Qj
        return Q


    def find_split(self, splits, gradient, hessian, is_last_level):
        """_summary_

        Args:
            splits (_type_): _description_
            gradient (_type_): _description_
            hessian (_type_): _description_
        """
        maxScore = -np.inf
        feature = np.inf
        value = np.inf
        featureName = "no split"
        
        if not is_last_level:
            for k in range(self.config.nFeatures):
                G = sum(gradient[k])
                H = sum(hessian[k])
                Gl, Hl = 0, 0
                for v in range(np.shape(splits[k])[0]):
                    Gl += gradient[k][v]
                    Hl += hessian[k][v]
                    Gr = G-Gl
                    Hr = H-Hl
                    score = L(G, H, Gl, Gr, Hl, Hr, self.config.lam, self.config.alpha, self.config.gamma)
                    if score > maxScore and Hl > 1 and Hr > 1: # TODO 1 = min_child_weight
                        value = splits[k][v]
                        featureId = k
                        featureName = self.fName[k]
                        maxScore = score

        # print(featureName)
        # print(value)
        # print(maxScore)
        # dice = np.random.randint(0, 1000)
        # for i in range(0, len(gradient)-1):
        #     print(f"i  = {i}:{np.sum(gradient[i])}, i+1 = {i+1}:{np.sum(gradient[i+1])} random -> {dice}")
        # assert all([np.sum(gradient[i]) == np.sum(gradient[i+1]) for i in range(0, len(gradient)-1)])
        weight, nodeScore = FLTreeNode.compute_leaf_param(gVec=gradient[0], hVec=hessian[0], lamb=self.config.lam, alpha=self.config.alpha) #TODO not done correctly should be done seperately!
        weight = self.config.learning_rate * weight
        return SplittingInfo(bestSplitScore=maxScore, featurId=featureId, featureName=featureName, splitValue=value, weight=weight, nodeScore=nodeScore)

    def update_trees(self, last_level_nodes:List[List[FLTreeNode]], splits:List[List[SplittingInfo]], depth):
        new_nodes = [[] for _ in range(self.config.nClasses)]
        for c in range(self.config.nClasses):
            for n, node in enumerate(last_level_nodes[c]):
                splitInfo = splits[c][n]
                if splitInfo.isValid and depth < self.config.max_depth:
                    node.splittingInfo = splitInfo
                    node.leftBranch = FLTreeNode(parent=node)
                    node.rightBranch = FLTreeNode(parent=node)
                    fName = splitInfo.featureName
                    sValue = splitInfo.splitValue
                    # print(self.original_data.featureDict.keys())
                    if rank != PARTY_ID.SERVER:
                        node.leftBranch.instances = np.logical_and([self.original_data.featureDict[fName][index] <= sValue for index in range(self.nUsers)], node.instances)
                        node.rightBranch.instances = np.logical_and([self.original_data.featureDict[fName][index] > sValue for index in range(self.nUsers)], node.instances)
                    new_nodes[c].append(node.leftBranch)
                    new_nodes[c].append(node.rightBranch)
                else:
                    node.weight = splitInfo.weight
                    node.score = splitInfo.nodeScore
                    node.leftBranch = None
                    node.rightBranch = None
        return new_nodes

    def appendGradients(self, instances, G, H, orgData):
        Gkv = []  #np.zeros((self.nClasses, amount_of_bins))
        Hkv = []
        # G, H = (nUsers,)
        # to return = (nFeatures, nBins)
        Dx = []  # the split the data corresponds to, this means the split left of the value. If there is no split with a smaller value than take the smalles split value
        k = 0
        for fName, fData in self.quantileDB.featureDict.items():
            splits = self.quantileDB.featureDict[fName].splittingCandidates
            # Dxk = np.zeros((np.shape(splits)[0] + 1, ))
            Gk =  np.zeros((np.shape(splits)[0] + 1, ))
            Hk =  np.zeros((np.shape(splits)[0] + 1, ))

            data = orgData.featureDict[fName]
            gradients = G
            hessians = H

            # append gradient of corresponding data to the bin in which the data fits.                                 
            bin_indices = np.searchsorted(splits, data) # indices 
            # bin_indices = [x-1 if x==len(splits) else x for x in bin_indices] # replace -1 with 0 such that upper most left values gets assigned to the right split. 
            
            # qData = [np.inf if x == len(splits) + 1 else x for x in np.searchsorted(splits, data)]

            # Dx.append(qData)

            for index in range(np.shape(data)[0]):
                bin = bin_indices[index]
                if instances[index]: # if instances[index] is true, then it is still a valid row.
                    Gk[bin] += gradients[index]
                    Hk[bin] += hessians[index]
            
            Gkv.append(Gk)
            Hkv.append(Hk)

            k -=- 1
        return Gkv, Hkv, None
        # comm.send((Gkv, Hkv, Dx), PARTY_ID.SERVER, tag=MSG_ID.RESPONSE_GRADIENTS)
    
    def fit(self, X_train, y_train, fName):
            quantile = QuantiledDataBase(DataBase.data_matrix_to_database(X_train, fName) )

            initprobability = (sum(y_train))/len(y_train)
            total_users = comm.Get_size() - 1
            total_lenght = len(X_train)
            elements_per_node = total_lenght//total_users
            start_end = [(i * elements_per_node, (i+1)* elements_per_node) for i in range(total_users)]

            if rank != PARTY_ID.SERVER:
                start = start_end[rank-1][0]
                end = start_end[rank-1][1]
                X_train_my = X_train[start:end, :]
                y_train_my = y_train[start:end]

            # split up the database between the users
            if rank == PARTY_ID.SERVER:
                pass
                self.setData(fName=fName)
            else:
                original = DataBase.data_matrix_to_database(X_train_my, fName)
                quantile = quantile.splitupHorizontal(start_end[rank-1][0], start_end[rank-1][1])
                self.setData(quantile, fName, original, y_train_my)
            
            self.boost(initprobability)
            return self
    
    def predict(self, X): # returns class number
        y_pred = self.predict_proba(X) # get leaf node weights
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X): # returns probabilities of all classes
        y_pred = self.predictweights(X) # returns weights
        for rowid in range(np.shape(y_pred)[0]):
            row = y_pred[rowid, :]
            wmax = max(row)
            wsum = 0
            for y in row: wsum += np.exp(y-wmax)
            y_pred[rowid, :] = np.exp(row-wmax) / wsum
        return y_pred
    
    def predictweights(self, X): # returns weights
            # if type(X) == np.ndarray:
            #     X = DataBase.data_matrix_to_database(X, self.fName)

            # y_pred = [None for n in range(self.nClasses)]
            # y_pred = np.tile(self.init_Probas, (len(X) , 1)) #(Nclasses, xrows)
            y_pred = np.zeros((len(X), self.config.nClasses))

            data_num = X.shape[0]
            # Make predictions
            testDataBase = DataBase.data_matrix_to_database(X, self.fName)
            # print(f"DEBUG: {np.shape(X)}, {np.shape(y_pred)}, {np.shape(self.init_Probas)}, {np.shape(self.trees)}")
            # print(f"init_probas = {self.init_Probas}")
            for treeID in range(self.config.max_tree):
                for c in range(self.config.nClasses):
                    tree = self.trees[c][treeID]
                    # Estimate gradient and update prediction
                    self.logger.warning(f"PREDICTION id {treeID}")
                    b = FLVisNode(self.logger, tree.root)
                    b.display(treeID)
                    update_pred = tree.predict(testDataBase)
                    y_pred[:, c] += update_pred #* self.config.learning_rate
            return y_pred

             

class SFXGBoostTree:
    def __init__(self, id) -> None:
        self.id = id
        self.root = FLTreeNode()
        self.nNode = 0
        self.G = [[] for _ in range(max_depth)]
        self.H = [[] for _ in range(max_depth)]
        
    def setInstances(self, nUsers, instances):
        self.root.nUsers = nUsers
        self.root.instances = instances

    def predict(self, data:DataBase):
        curNode = self.root
        outputs = np.empty(data.nUsers, dtype=float)

        for userId in range(data.nUsers):
            curNode = self.root
            while(not curNode.is_leaf()):
                direction = \
                    (data.featureDict[curNode.splittingInfo.featureName].data[userId] > curNode.splittingInfo.splitValue)
                if(direction == Direction.LEFT):
                    curNode =curNode.leftBranch
                elif(direction == Direction.RIGHT):
                    curNode = curNode.rightBranch
            outputs[userId] = curNode.weight
        return outputs