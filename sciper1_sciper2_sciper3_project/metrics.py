import numpy as np 

def accuracy_fn(pred_labels, gt_labels):
    '''
    '''
    return np.mean(pred_labels == gt_labels)*100

def macrof1_fn(pred_labels,gt_labels):
    '''
        Macro F1 score
        Arguments:
            pred_labels: N prediction labels
            gt_labels: N corresponding gt labels
        Returns:
            returns the computed macro f1 score
    '''
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        macrof1 += 2*(precision*recall)/(precision+recall)
    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss

    '''

    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss


