""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
from networks import configs as cfg
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def evaluate(labels, scores, metric, best_auc):
    if metric == 'roc':
        return roc(labels, scores, best_auc)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.50
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, best_auc, saveto='./outputs', ):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.3f, EER = %0.3f)' % (roc_auc, eer))
#         plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "Current_Epoch_ROC.pdf"))
        if roc_auc>best_auc:
            plt.savefig(os.path.join(saveto, "Best_ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap

def confuse_matrix(score, lb):
    lb = lb.cpu().numpy()
    score = np.array(score.detach().squeeze(0).cpu())
    threshold = 0.5
    score[score>threshold] = 1.0
    score[score<=threshold] = 0.0 
    lb[lb>threshold] = 1.0
    lb[lb<=threshold] = 0.0 
    lb = lb[0,:,:]
    lb = np.round(lb)
    
    tp = np.sum(lb*score)
    fn = lb-score
    fn[fn<0]=0
    fn = np.sum(fn)
    tn = lb+score
    tn[tn>0]=-1
    tn[tn>=0]=1
    tn[tn<0]=0
    tn = np.sum(tn)
    fp = score - lb
    fp[fp<0] = 0
    fp = np.sum(fp)
    
    return tp, fp, tn, fn
        
def eva_metrics(TP, FP, TN, FN):
    precision = TP/(TP+FP+1e-8)
    oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    P = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2+1e-8)
    kappa = (oa-P)/(1-P+1e-8)
    
    return [precision, oa, recall, f1, kappa]
       
