
import numpy as np
from SFXGBoost.common.XGBoostcommon import ThresholdL1
from SFXGBoost.config import rank, comm, Config

class SplittingInfo:
    def __init__(self, bestSplitScore = -np.Infinity, featurId = None, featureName=None, splitValue=0.0, weight = None, nodeScore = None) -> None:
        self.bestSplitScore = bestSplitScore
        self.bestSplitParty = None
        self.selectedFeatureID = 0
        self.selectedCandidate = 0
        # self.instances = []
        self.featureId = featurId
        self.featureName = featureName
        self.splitValue = splitValue
        
        self.isValid = bestSplitScore > 0

        self.weight = weight
        self.nodeScore = nodeScore

    def delSplittinVector(self): # performance attempt
        self.bestSplittingVector = None

    def log(self, logger):
        logger.debug("Best Splitting Score: L = %.2f, Selected Party %s",\
                self.bestSplitScore, str(self.bestSplitParty))
        logger.debug("%s", self.get_str_split_info())
        logger.debug("The optimal splitting vector: %s| Feature ID: %s| Candidate ID: %s",\
            str(self.bestSplittingVector), str(self.selectedFeatureID), str(self.selectedCandidate))


    def get_str_split_info(self):
        """
        
        """
        retStr = ''
        if(self.bestSplittingVector is not None):
            retStr = "[P: %s, N = %s, " % (str(self.bestSplitParty), str(len(self.bestSplittingVector)))
        else:
            return "Infeasible splitting option. The tree growing should be terminated..."

        
        if(self.featureName is not None): # implies the private splitting info is set by the owner party
            retStr += "F: %s, S: %.4f]" % (str(self.featureName), (self.splitValue))
        else:
            retStr += "F: Unknown, s: Unknown]" 
        return retStr

class TreeNode:
    def __init__(self, weight = 0.0, leftBranch=None, rightBranch=None):
        self.weight = weight
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.weightpi = [ 0 for _ in range(comm.Get_size() -1)]
        

    def logNode(self, logger):
        logger.info("Child Node Addresses: L %d| R %d", id(self.leftBranch), id(self.rightBranch))

    def get_string_recursive(self):
        str = ""
        if not self.is_leaf():
            str += "[Addr: {} Child L: {} Child R: {} Weight: {}]".format(id(self), id(self.leftBranch), id(self.rightBranch), self.weight)
            str += "{}".format(self.get_private_info())
            str += " \nChild Info \nLeft Node: {} \nRight Node: {}".format(self.leftBranch.get_string_recursive(), self.rightBranch.get_string_recursive())
        else:
            str += "[TreeLeaf| Addr: {} Weight: {}]".format(id(self), self.weight)
        return str

    def get_private_info(self):
        return

    def is_leaf(self):
        return (self.leftBranch is None) and (self.rightBranch is None)


class FLTreeNode(TreeNode):
    def __init__(self, FID = 0, weight=0, parent=None, leftBranch=None, rightBranch=None, ownerID = -1, instances=None):
        super().__init__(weight, leftBranch, rightBranch)
        self.FID = FID
        self.owner = ownerID
        self.splittingInfo = SplittingInfo()
        self.score = None
        self.parent = parent
        self.instances = instances
        self.G = None 
        self.H = None
        self.Gpi = [ [] for _ in range(comm.Get_size() - 1)]
        self.Hpi = [ [] for _ in range(comm.Get_size() - 1)]
        self.splitting_sgh = None  # NOT USED YET
        self.splitting_sgh = None  # store parent's splits, + your gradients

    def get_nodes_depth(self, l, cur_l=0):
        """retrieves all the nodes that are at depth l [0, max_depth]

        Args:
            l (_type_): _description_
            cur_l (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        ret = []
        if l == 0:
            return [self]
        else:
            for child in [self.leftBranch, self.rightBranch]:
                if child == None:
                    continue
                if cur_l+1 == l:
                    ret.append(child)
                else:
                    nodes = child.get_nodes_depth(l, cur_l+1)
                    if nodes == []:
                        continue
                    else:
                        ret.extend(nodes)
        return ret

    def get_private_info(self):
        return "\nOwner ID:{}".format(self.owner)

    def set_splitting_info(self, sInfo: SplittingInfo):
        self.owner = sInfo.bestSplitParty
        self.splittingInfo = sInfo

    def find_child_node(self, id):
        if (self.FID) is id:
            return self
        for child in [self.leftBranch, self.rightBranch]:
            if child is not None:
                ret = child.find_child_node(id)
                if ret:
                    #print("Yay")
                    return ret
        return None

  
    def compute_score(self, gamma):
        score = 0
        if self.is_leaf():
            return self.score + gamma
        else:
            for child in [self.leftBranch, self.rightBranch]:
                if child is not None:
                    score += child.compute_score() 
        return score

    @staticmethod
    def compute_leaf_param(gVec, hVec, lamb, alpha):
        # sum over one features's gradients and hessians. all sums should be equal of all different features
        # param.h use thresholdL1 if alpha != 0
        gI = np.sum(gVec) # TODO other sum
        hI = np.sum(hVec)
        # print(f"gI = {gI}")
        # print(f"hI = {hI}")
        weight = -1.0 * ThresholdL1(gI, alpha) / (hI + lamb)
        score = 1/2 * weight * gI
        return weight, score
