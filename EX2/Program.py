import pandas as pd
from numpy import loadtxt
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from SVRGradientBoostingRegressor import SVRGradientBoostingRegressor
import timeit

DATASETS = {}

DATASETS[0] = [r'Data\airfoil_self_noise.csv', 5]
DATASETS[1] = [r'Data\Appliances_energy_prediction.csv', 26]
DATASETS[2] = [r'Data\Cycle_Power_Plant.csv', 4]
DATASETS[3] = [r'Data\Glass.csv', 9]
DATASETS[4] = [r'Data\concrete.csv', 8]


def read_csv(path, max):
    """
    Read datasets into array-like structure.
    :param path: Path of file
    :param max: Max number of input attributes
    :param delim: Delimiter for parsing the file
    :return: Training and testing sets splits
    """
    dataset = loadtxt(path, delimiter=',')

    # Fill missing values in datasets (if exist)
    df = pd.DataFrame.from_records(dataset)
    df = pd.DataFrame.from_records(dataset).fillna(df.mean())
    # Convert back to numpy ndarray
    dataset = df.values

    # Split the columns of the data to training attributes and output
    X = dataset[:, 0:max]
    Y = dataset[:, max]

    # Split data to training set and test set for evaluation (Default of 10-folt CV)
    seed = 7
    return train_test_split(X, Y,
                            test_size=0.1,
                            random_state=seed)


def evaluate(model, X, Y, kfold):
    """
    Evaluate given model on training set using k-fold cross validation
    :param model: Model to evaluate
    :param X: Training set
    :param Y: Output attribute of model
    :param kfold: Number of folds in K-fold CV
    :return: MSE
    """
    results = model_selection.cross_val_score(model, X, Y,
                                              cv=kfold,
                                              verbose=0,
                                              scoring='neg_mean_squared_error')
    print("mse {:10.3f}".format(results.mean()))
    return results.mean()


def main():
    n_est = [100, 200, 500, 1000]
    lr = [0.05, 0.1, 0.3, 0.5, 0.8, 1]
    loss = ['ls', 'lad', 'huber', 'quantile']
    min_leaf = [3, 5, 10, 20]
    max_depth = [3, 4, 5, 6]

    for i in range(0, 5):
        print("-------------------------")
        path = DATASETS[i][0]
        max_attr = DATASETS[i][1]
        X_train, X_test, y_train, y_test = read_csv(path, max_attr)
        scikit_model = GradientBoostingRegressor(loss=loss[0],
                                                 learning_rate=lr[0],
                                                 n_estimators=n_est[2],
                                                 max_depth=max_depth[3],
                                                 min_samples_leaf=min_leaf[2])

        # Calculate elapsed running time
        startRegularModel = timeit.default_timer()
        scikit_model.fit(X_train, y_train)
        stopRegularModel = timeit.default_timer()
        ansRegular = stopRegularModel - startRegularModel
        print("Scikit regressor on " + path + ": " + str(ansRegular))

        modified_model = SVRGradientBoostingRegressor(loss=loss[0],
                                                      learning_rate=lr[0],
                                                      n_estimators=n_est[2],
                                                      max_depth=max_depth[3],
                                                      min_samples_leaf=min_leaf[2])

        startNewModel = timeit.default_timer()
        modified_model.fit(X_train, y_train)
        stopNewModel = timeit.default_timer()
        ansNewModel = stopNewModel - startNewModel
        print("Modified regressor on " + path + ": " + str(ansNewModel))

        evaluate(scikit_model, X_train, y_train, 10)
        evaluate(modified_model, X_train, y_train, 10)


if __name__ == "__main__":
    main()
