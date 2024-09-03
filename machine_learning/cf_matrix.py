'''Routines for displaying and calculating metrics for confusion matrices.'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          filename=None):
    """This function will make a pretty plot of an sklearn Confusion Matrix using a Seaborn heatmap visualization.
    
    Adapted from https://github.com/DTrimarchi10/confusion_matrix

    Args:
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html              
        title:         Title for the heatmap. Default is None.
        filename:      Filename to save figure to and the extension determines the file format. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = [f'{value}\n' for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f'{value:0.0f}\n' for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        # normalise across rows as done in confusion_matrix in sklearn if normalise is 'true'
        group_percentages = [f'{value:.2%}' for value in (cf / np.sum(cf, axis=1)[:, None]).flatten()]
    else:
        group_percentages = blanks

    box_labels = [f'{v1}{v2}{v3}'.strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS
    if sum_stats:
        precision_macro, recall_macro, f1_macro, accuracy, _, _, _ = calculate_metrics(cf)
        stats_text = f'\n\nRecall={recall_macro:0.3f} Precision={precision_macro:0.3f} F1 Score={f1_macro:0.3f} Accuracy={accuracy:0.3f}'
    else:
        stats_text = ''

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)

    sns.heatmap(cf / cf.sum(axis=1)[:, None],annot=box_labels,fmt='',cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories,vmin=0,vmax=1)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel(f'Predicted label{stats_text}')
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    if filename:
        plt.savefig(f'{filename}', dpi=1200, bbox_inches='tight', transparent=True)
    else:
        plt.show()
    plt.close()

def calculate_metrics(cm):
    """
    Calculate confusion matrix-based metrics, including precision, recall, F1-score, and accuracy.

    Note that as some of our participants don't do all of the tasks, we have to calculate the mean
    metrics using np.nanmean, which ignores NaN values.

    Args:
        cm (numpy.ndarray): Confusion matrix from which to calculate the metrics. 
                            Should be a 2D array where the rows are the true classes 
                            and the columns are the predicted classes.

    Returns:
        precision_macro (float): Macro average precision across all classes.
        recall_macro (float): Macro average recall across all classes.
        f1_macro (float): Macro average F1-score across all classes.
        accuracy (float): Overall accuracy of the predictions.
        precision (numpy.ndarray): Precision scores for each individual class.
        recall (numpy.ndarray): Recall scores for each individual class.
        f1 (numpy.ndarray): F1-scores for each individual class.
    """
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)

    # Macro Average Precision
    precision = tp / (tp + fp)
    precision_macro = np.nanmean(precision)

    # Macro Average Recall
    recall = tp / (tp + fn)
    recall_macro = np.nanmean(recall)

    # Macro Average F1
    f1 = 2 * (precision * recall) / (precision + recall)
    f1_macro = np.nanmean(f1)

    # Accuracy
    accuracy = np.sum(tp) / np.sum(cm)

    return precision_macro, recall_macro, f1_macro, accuracy, precision, recall, f1

def calculate_caret_metrics(cm):
    """
    Calculate confusion matrix-based metrics to mimic behaviour of the `confusionMatrix` function
    from the `caret` package in R in a multiclass setting.

    Args:
        cm (numpy array): Confusion matrix from which to calculate the metrics.

    Returns:
        metrics (dict): a dictionary with calculated metrics for each class.
    """

    total = np.sum(cm)

    # From the confusion matrix we can get tp, tn, fp, fn for each class.
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = total - (fp + fn + tp)

    # Sensitivity, tpr, Recall
    tpr = tp/(tp+fn)
    # Specificity, tnr
    tnr = tn/(tn+fp)
    # Positive Predictive Power
    ppv = tp/(tp+fp)
    # Negative Predictive Power
    npv = tn/(tn+fn)
    # Prevalence
    prevalence = (tp+fn) / total
    # Detection Rate
    detection_rate = tp / total
    # Detection Prevalence
    detection_prevalence = (tp + fp) / total
    # Balanced Accuracy
    balanced_accuracy = (tpr + tnr) / 2

    # Accuracy
    accuracy = np.trace(cm) / total

    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': tpr.tolist(),
        'Specificity': tnr.tolist(),
        'Positive Predictive Value': ppv.tolist(),
        'Negative Predictive Value': npv.tolist(),
        'Prevalence': prevalence.tolist(),
        'Detection Rate': detection_rate.tolist(),
        'Detection Prevalence': detection_prevalence.tolist(),
        'Balanced Accuracy': balanced_accuracy.tolist()
    }

    return metrics

def calculate_our_metrics(cm):
    """
    Calculate confusion matrix-based metrics to compare between ML and DT models

    Args:
        cm (numpy array): Confusion matrix from which to calculate the metrics.

    Returns:
        metrics (dict): a dictionary with calculated metrics for each class.
    """

    total = np.sum(cm)

    # From the confusion matrix we can get tp, tn, fp, fn for each class.
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = total - (fp + fn + tp)

    # Sensitivity, tpr, Recall
    tpr = tp/(tp+fn)
    # Specificity, tnr
    tnr = tn/(tn+fp)

    balanced_accuracy = (tpr + tnr) / 2

    precision = tp / (tp + fp)
    f1 = 2 * (precision * tpr) / (precision + tpr)

    metrics = {
        'Sensitivity': tpr.tolist(),
        'Specificity': tnr.tolist(),
        'Balanced Accuracy': balanced_accuracy.tolist(),
        'Precision': precision.tolist(),
        'F1': f1.tolist()
    }

    # convert metrics from fraction to a percentage
    for key, value in metrics.items():
        metrics[key] = [x * 100 for x in value]

    return metrics
