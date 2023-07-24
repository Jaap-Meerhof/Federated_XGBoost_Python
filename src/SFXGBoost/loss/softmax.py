import numpy as np

def diff(actual, predicted):
    return 

def getGradientHessians(y, y_pred):
    if case_weight is None: case_weight = np.ones(y.shape)
    grad = np.zeros((y_pred.shape), dtype=float) # for multi-class
    hess = np.zeros((y_pred.shape), dtype=float) # for multi-class
    for rowid in range(y_pred.shape[0]):
        wmax = max(y_pred[rowid]) # line 100 multiclass_obj.cu
        wsum =0.0
        for i in y_pred[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(y_pred.shape[1]):
            p = np.exp(y_pred[rowid][c]- wmax) / wsum
            target = y[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g
            hess[rowid][c] = h
    return grad, hess

def gradient(actual, predicted): # TODO make this not incredibly redundant
    actualtmp = np.argmax(actual, axis=1)
    grad = np.zeros((predicted.shape), dtype=float) # for multi-class
    hess = np.zeros((predicted.shape), dtype=float) # for multi-class
    for rowid in range(predicted.shape[0]):
        wmax = max(predicted[rowid]) # line 100 multiclass_obj.cu
        wsum =0.0
        for i in predicted[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(predicted.shape[1]):
            p = np.exp(predicted[rowid][c]- wmax) / wsum
            target = actualtmp[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g
            hess[rowid][c] = h
    return np.array(grad)

def hess(actual, predicted): # TODO make this not incredibly redundant
    actualtmp = np.argmax(actual, axis=1)
    grad = np.zeros((predicted.shape), dtype=float) # for multi-class
    hess = np.zeros((predicted.shape), dtype=float) # for multi-class
    for rowid in range(predicted.shape[0]):
        wmax = max(predicted[rowid]) # line 100 multiclass_obj.cu
        wsum =0.0
        for i in predicted[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(predicted.shape[1]):
            p = np.exp(predicted[rowid][c]- wmax) / wsum
            target = actualtmp[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g
            hess[rowid][c] = h
    return np.array(hess)