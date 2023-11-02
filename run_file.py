import pandas as pd
import numpy as np
from data_preparation import price_ratio, feature_enginnering, filtering
from model_training import train_LGBM
from sklearn.metrics import accuracy_score, classification_report

"""
Author: Zhihan Xu
Date: 2nd Nov 2023

Summary
- LightGBM model is utilised to fit the data.
- 6 quantitative indicators are utilised based on the close price and the volume of the stocks.
- Those indicators are treated as features into the model, feature values are normalised per stock before input.  
- The target label is chosen to be the categorised consecutive price change ratio. Three categories include gain, loss and stable.
- The performance of the model has the accuracy of 0.447, which outperforms the random guess of 0.333.
"""



if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv('data.csv')

    # To ensure that there are adequate samples per stock given rolling base of 50, filter out stocks if sample less than 500.
    data = filtering(data)

    pr_PerTicker = data.groupby('ticker').apply(price_ratio)
    pr_PerTicker.index = pr_PerTicker.index.get_level_values('ticker')

    # Considering the computation of quantitative index, set the rolling base to be 50.
    feature_PerTicker = data.groupby('ticker', group_keys=True).apply(feature_enginnering)
    feature_PerTicker.index = feature_PerTicker.index.get_level_values('ticker')

    features = feature_PerTicker.dropna()
    labels = pr_PerTicker.dropna()
    labels.index = range(len(labels))

    # Categorise the computed price ratio labels. Given that the mean of price ratio is 0 by chance,
    # partition the ratio into three categories by the its quantile, which are [1, 2, 0] representing [loss, gain, balance].
    cat_labels = pd.Series(np.zeros(len(labels)))
    bound1 = labels.quantile(0.4)
    bound2 = labels.quantile(0.6)
    cat_labels[labels < bound1] = 1
    cat_labels[labels > bound2] = 2

    # Standardise feature values across samples per stock
    norm_features = features.reset_index().groupby('ticker', group_keys=True).transform(lambda x: (x - x.mean()) / x.std())

    # Remove samples containing nan
    if any(norm_features.isna()):
        cat_labels = cat_labels[~norm_features.isna().any(axis=1)]
        norm_features = norm_features.dropna()

    # Fit the data into the model
    bst, [X_test, y_test] = train_LGBM(X = norm_features, y = cat_labels)

    # Make predictions on the test set
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

    # Convert probabilities to class predictions
    y_pred_classes = [list(pred).index(max(pred)) for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_classes)
    print('Accuracy: %.3f' %accuracy)

    # Print report, note that the label of [1, 2, 0] represents [loss, gain, balance]
    print(classification_report(y_test, y_pred_classes))


