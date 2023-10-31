
from distutils.log import Log
import numpy as np
from SFXGBoost.common.BasicTypes import Direction
from SFXGBoost.config import rank


def ThresholdL1(g, alpha): # for the split function g is never < 0 
    if g > alpha:
        return g - alpha
    elif g < -alpha:
        return g + alpha
    else:
        return 0.0
    
# L = lambda G,H, GL, GR, HL, HR, lamb, gamma: 1/2 * ((ThresholdL1(GL*GL) / (HL + lamb)) + (ThresholdL1(GR*GR) / (HR + lamb)) - (ThresholdL1(G*G) / (H + lamb))) - gamma
L = lambda G,H, GL, GR, HL, HR, lamb, alpha, gamma: ((ThresholdL1(GL*GL, alpha) / (HL + lamb)) + (ThresholdL1(GR*GR, alpha) / (HR + lamb)) - (ThresholdL1(G*G, alpha) / (H + lamb))) - gamma

# L = lambda G,H, GL, GR, HL, HR, lamb, gamma, alpha: 1/2 * (ThresholdL1(GL, alpha) / (HL + lamb)) + (GR*GR / (HR + lamb)) - (G*G / (H + lamb))) - gamma

# def computeSplitScore(Gl, Gr)

class PARTY_ID:
    ACTIVE_PARTY = 1
    SERVER = 0


class XgboostLearningParam():
    #def __init__(self) -> None:
    LOSS_FUNC = None
    LOSS_TERMINATE = None
    LAMBDA = None
    ALPHA=None
    GAMMA = None
    N_TREES = None
    MAX_DEPTH = None

def compute_splitting_score_quantile(splits, GVec, HVec, lamb, alpha, gamma):
    # instances = (nFeature, depends) True if still applicalbe, False if not relevant for this node
    def inhomogenioussum(mylist):
        sum = 0
        for k in range(len(mylist)):
            for v in range(len(mylist[k])):
                sum += mylist[k][v]
        return sum
        
    G = inhomogenioussum(GVec)
    H = inhomogenioussum(HVec)

    # splits = self.qDataBase.featureDict[fName].splittingCandidates

    nFeature = len(splits)  # splits = (nFeature, depends on amount of splits of that feature) same as GVec and HVec!
    maxscore = -np.inf
    feature = np.inf
    value = np.inf

    for k in range(nFeature):
        Gl, Hl = 0, 0
        for v in range(len(splits[k])): # go over every split possible in this feature. 
            Gl += GVec[k][v] # dit is irritant
            Hl += HVec[k][v]
            Gr = G-Gl
            Hr = H-Hl
            score = L(G, H, Gl, Gr, Hl, Hr, lamb, alpha, gamma)
            if score > maxscore:
                value = splits[k][v]  # nogiets linker split value s_{k,v}
                feature = k
                maxscore = score
    
    from copy import deepcopy
    gradientleft = deepcopy(GVec)
    gradientleft[feature] = [g if i <= value else 0 for i, g in enumerate(gradientleft[feature])]

    hessleft = deepcopy(HVec)
    hessleft[feature] = [h if i <= value else 0 for i, h in enumerate(hessleft[feature])]
    # hessleft[feature, :] = [h for i, h in enumerate(hessleft[feature, :]) if i <= j]

    ghleft = (gradientleft, hessleft)
    
    gradientright = deepcopy(GVec)
    # gradientright[feature, :] = [g for i, g in enumerate(gradientright[feature, :]) if i > j] + [0]
    gradientright[feature] = [g if i > value else 0 for i, g in enumerate(gradientright[feature])]
    
    hessright = deepcopy(HVec)
    hessright[feature] = [h if i > value else 0 for i, h in enumerate(hessright[feature])]

    ghright = (gradientright, hessright)

    return value, feature, maxscore, ghleft, ghright

def weights_to_probas(y_pred):
    for rowid in range(np.shape(y_pred)[0]):
                row = y_pred[rowid, :]
                wmax = max(row)
                wsum = 0
                for y in row: wsum += np.exp(y-wmax)
                y_pred[rowid, :] = np.exp(row-wmax) / wsum
    return y_pred



def compute_splitting_score(SM, GVec, HVec, lamb, alpha, gamma):
    G = sum(GVec)
    H = sum(HVec)
    print(f"SM: {np.shape(SM)}")
    print(f"Gvec: {np.shape(GVec)}")

    GRVec = np.matmul(SM, GVec)
    HRVec = np.matmul(SM, HVec)
    GLVec = G - GRVec
    HLVec = H - HRVec
    score = L(G,H,GLVec,GRVec,HLVec,HRVec, lamb, alpha, gamma)

    bestSplitId = np.argmax(score)
    maxScore = score[bestSplitId]
    return score.reshape(-1), maxScore, bestSplitId

def get_splitting_score(G, H, GL, GR, HL, HR, lamb = XgboostLearningParam.LAMBDA, alpha=XgboostLearningParam.ALPHA, gamma = XgboostLearningParam.GAMMA):
    score = L(G,H,GL,GR,HL,HR,lamb, alpha, gamma)
    return score.reshape(-1)
 

# class SplittingInfo:
#     def __init__(self) -> None:
#         self.bestSplitScore = -np.Infinity
#         self.bestSplitParty = None
#         self.bestSplittingVector = None
#         self.selectedFeatureID = 0
#         self.selectedCandidate = 0
#         self.isValid = False

#         self.featureName = None
#         self.splitValue = 0.0

#     def delSplittinVector(self): # performance attempt
#         self.bestSplittingVector = None

#     def log(self, logger):
#         logger.debug("Best Splitting Score: L = %.2f, Selected Party %s",\
#                 self.bestSplitScore, str(self.bestSplitParty))
#         logger.debug("%s", self.get_str_split_info())
#         logger.debug("The optimal splitting vector: %s| Feature ID: %s| Candidate ID: %s",\
#             str(self.bestSplittingVector), str(self.selectedFeatureID), str(self.selectedCandidate))


#     def get_str_split_info(self):
#         """
        
#         """
#         retStr = ''
#         if(self.bestSplittingVector is not None):
#             retStr = "[P: %s, N = %s, " % (str(self.bestSplitParty), str(len(self.bestSplittingVector)))
#         else:
#             return "Infeasible splitting option. The tree growing should be terminated..."

        
#         if(self.featureName is not None): # implies the private splitting info is set by the owner party
#             retStr += "F: %s, S: %.4f]" % (str(self.featureName), (self.splitValue))
#         else:
#             retStr += "F: Unknown, s: Unknown]" 
#         return retStr


class FedQueryInfo:
    def __init__(self, userIdList = None) -> None:
        #self.nUsers = len(userIdList)
        self.nUsers = 1
        self.userIdList = userIdList

class FedDirRequestInfo(FedQueryInfo):
    def __init__(self, userIdList) -> None:
        super().__init__(userIdList)
        self.nodeFedId = None

    def log(self, logger):
        logger.debug("Inference Request| NodeFedID %d| nUsers: %d| Users: %s|", self.nodeFedId, self.nUsers, self.userIdList)


class FedDirResponseInfo(FedQueryInfo):
    def __init__(self, userIdList) -> None:
        super().__init__(userIdList)
        self.Direction = [Direction.DEFAULT for i in range(self.nUsers)]


