import pandas as pd
import numpy as np
from sklearn import metrics

def classification_report(y_true, y_pred, non_biol_classes=['detritus'], **kwargs):
    """
    Build a text report showing the main classification metrics.
    
    Args:
        y_true (1d array-like): Ground truth (correct) target values
        y_pred (1d array-like): Predicted target values
        non_biol_classes (list of strings): Classes to exclude to compute
            statistics on biological classes only.
        **kwargs: Passed to sklearn.metrics.classification_report

    Returns:
        cr (pd.DataFrame): The classification report, as a DataFrame
    """

    # compute the classification report
    cr = metrics.classification_report(y_true=y_true, y_pred=y_pred,
             output_dict=True, **kwargs)
             
    # convert to DataFrame for printing and computation
    cr = pd.DataFrame(cr).transpose()
    
    # get only biological classes
    stats = ['accuracy', 'macro avg', 'weighted avg']
    biol_cr = cr[~cr.index.isin(non_biol_classes + stats)]
    
    # compute stats for biological classes
    biol_macro_avg = biol_cr.apply(np.average)
    biol_weighted_avg = biol_cr.apply(np.average, weights=biol_cr.support)
    
    # reformat as DataFrame
    biol_stats = pd.concat([biol_macro_avg, biol_weighted_avg], axis=1)
    biol_stats.columns = ['biol macro avg', 'biol weighted avg']
    biol_stats = biol_stats.transpose()
    biol_stats.support = len(y_true)
    
    # add to the total classification report
    cr = cr.append(biol_stats)
    # and format it nicely
    cr['precision']['accuracy'] = np.float64('NaN')
    cr['recall']['accuracy'] = np.float64('NaN')
    cr['support']['accuracy'] = len(y_true)
    cr.support = cr.support.astype(int)
    cr = cr.round(2)
    
    return(cr)
