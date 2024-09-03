def eval_classification(y_test, y_pred_encoded):
    import pandas as pd
    from statistics import mean
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        cohen_kappa_score,
    )

    # Calculate the metrics
    accuracy = accuracy_score(y_test, y_pred_encoded)
    precision = precision_score(y_test, y_pred_encoded, average="macro")
    recall = recall_score(y_test, y_pred_encoded, average="macro")
    f1 = f1_score(y_test, y_pred_encoded, average="macro")
    cohen_kappa = cohen_kappa_score(y_test, y_pred_encoded)

    conf_matrix = confusion_matrix(y_test, y_pred_encoded)

    # Calculate specificity for each class
    specificity = []
    for i in range(conf_matrix.shape[0]):
        true_negatives = (
            sum(conf_matrix[j][j] for j in range(conf_matrix.shape[0]))
            - conf_matrix[i].sum()
        )
        false_positives = sum(conf_matrix[i]) - conf_matrix[i][i]
        specificity_i = true_negatives / (true_negatives + false_positives)
        specificity.append(specificity_i)

    specificity = mean(specificity)

    # Create a DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Kappa",
                "Specificity",
            ],
            "Value": [accuracy, precision, recall, f1, cohen_kappa, specificity],
        }
    )

    return metrics_df
