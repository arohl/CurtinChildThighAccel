## Supplementary Information

### Machine and Deep Learning Classifiers for Activity Recognition in Children from Thigh-Worn Accelerometer Data

#### Pure windows

|Model|Normalisation|Accuracy|Precision|Recall|F1|κ|
|-----|-------------|--------|---------|------|--|-|
|RF|none|0.907±0.052|0.755±0.110|0.841±0.098|0.782±0.100|0.850±0.093|
|BRF|none|0.910±0.052|0.755±0.114|0.854±0.093|0.787±0.100|0.855±0.094|
|XGBoost|none|0.937±0.038|0.847±0.105|0.816±0.104|0.827±0.085|0.894±0.076| 
|KNN|stddev|0.925±0.038|0.804±0.107|0.801±0.094|0.798±0.083|0.875±0.074| 
|SVM|stddev|0.908±0.054|0.752±0.114|0.847±0.107|0.792±0.098|0.850±0.095| 
|LSTM-CNN #1|stddev|0.883±0.052|0.773±0.111|0.791±0.115|0.758±0.118|0.809±0.082|
|LSTM-CNN #2|stddev|0.875±0.060|0.763±0.107|0.803±0.101|0.764±0.107|0.801±0.094|
|LSTM-CNN #3|stddev|0.885±0.050|0.762±0.118|0.799±0.108|0.764±0.112|0.813±0.085|
|LSTM-CNN #4|stddev|0.881±0.056|0.757±0.113|0.795±0.123|0.757±0.116|0.806±0.100|
|LSTM-CNN #5|stddev|0.877±0.046|0.749±0.121|0.788±0.108|0.747±0.116|0.801±0.077|
|LSTM-CNN #6|stddev|0.877±0.051|0.750±0.109|0.789±0.110|0.748±0.111|0.800±0.087|
|LSTM-CNN #7|stddev|0.881±0.054|0.752±0.108|0.803±0.103|0.756±0.108|0.806±0.092|
|KNN|minmax|0.925±0.036|0.788±0.104|0.786±0.091|0.785±0.075|0.876±0.069| 
|SVM|minmax|0.909±0.051|0.759±0.113|0.845±0.093|0.793±0.099|0.852±0.092| 
|LSTM-CNN #1|minmax|0.914±0.042|0.751±0.141|0.691±0.111|0.692±0.127|0.856±0.080|
|LSTM-CNN #2|minmax|0.924±0.048|0.785±0.098|0.807±0.123|0.774±0.111|0.871±0.112|
|LSTM-CNN #3|minmax|0.930±0.039|0.779±0.105|0.812±0.098|0.776±0.098|0.885±0.073|
|LSTM-CNN #4|minmax|0.930±0.037|0.784±0.105|0.817±0.100|0.780±0.100|0.885±0.070|
|LSTM-CNN #5|minmax|0.932±0.041|0.778±0.108|0.808±0.116|0.771±0.115|0.883±0.095|
|LSTM-CNN #6|minmax|0.926±0.050|0.775±0.105|0.808±0.099|0.770±0.104|0.878±0.089|
|LSTM-CNN #7|minmax|0.930±0.039|0.778±0.108|0.796±0.111|0.764±0.103|0.883±0.073|

**Table S1**  LOSO-CV performance using pure windows. Subject-wise accuracy, precision, recall, F1, and Cohen's kappa (κ) reported (mean ± SD). 

Focusing first on the accuracy, all the LSTM-CNN models with stddev normalisation were significantly lower (~0.88) than all of the rest of the models examined, which had accuracies > 0.9. This large impact of normalisation on the LSTM-CNN models was unexpected, given that it has negligible impact on the KNN and SVN models. The highest accuracy was observed for XGBoost, followed by the LSTM-CNN \#3–\#7 with minmax and KNN. However, the performance of models is often assessed by the F1 score or Cohen's kappa in ML and DL. XGBoost was the highest performing model on both of these metrics. All of the LSTM-CNN models had lower F1 scores than the ML models but Cohen's kappa ordering was very similar to that found for accuracy. In the medical sciences, balanced accuracy is often used. For the multi-class case, there are several definitions, but the one employed in scikit-learn is equivalent to the average recall used here. Table S1 shows that the BRF model performed best on this metric. Although the LSTM-CNN models didn't top any metric, we were interested in understanding their performance relative to the ML models in more detail. Consequently, the XGBoost, BRF, and LSTM-CNN \#3 models were trained on impure windows.

### Impure Windows

|Model|Accuracy|Precision|Recall|F1|κ|
|-----|--------|---------|------|--|-|
|XGBoost|0.892±0.043|0.787±0.091|0.736±0.088|0.750±0.085|0.833±0.077| 
|BRF|0.857±0.055|0.690±0.089|0.797±0.079|0.717±0.095|0.790±0.088|
|LSTM-CNN #3 (stddev)|0.861±0.049|0.731±0.098|0.768±0.097|0.732±0.099|0.787±0.080|
|LSTM-CNN #3 (minmax)|0.900±0.045|0.756±0.085|0.778±0.088|0.740±0.094|0.844±0.081|

**Table S2**  LOSO-CV performance using impure windows. Subject-wise accuracy, precision, recall, F1, and Cohen's kappa (κ) reported (mean ± SD).

