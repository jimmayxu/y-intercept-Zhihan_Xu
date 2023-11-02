import lightgbm as lgb
from sklearn.model_selection import train_test_split


def train_LGBM(X, y):

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    num_classes = len(set(y))

    # Set hyperparameters for the LightGBM model
    params = {
        'objective': 'multiclass',  # Multi-class classification
        'num_class': num_classes,  # Specify the number of classes
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'num_leaves': 31,  # Number of leaves in each tree
        'learning_rate': 0.05,
        'feature_fraction': 0.9,  # Fraction of features to consider in each tree
        'bagging_fraction': 0.8,  # Fraction of data to bag in each tree
        'bagging_freq': 5,  # Frequency for bagging
        'verbose': 0  # Control the level of LightGBM's verbosity
    }

    # Train the LightGBM model
    num_round = 100  # Number of boosting rounds (adjust as needed)
    bst = lgb.train(params, train_data, num_round)


    return bst, [X_test, y_test]