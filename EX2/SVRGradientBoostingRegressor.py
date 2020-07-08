import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y

# Ignore Deprecation warning regarding future argument modification of scikit in versions 0.19
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SVRGradientBoostingRegressor(GradientBoostingRegressor):
    """
    Class implementing an modification to the Gradient Boosting regressor of Scikit package.
    Instead of choosing the mean/median of the instances in the leaf of each estimator,
    a SVR regression model is built to predict the output of each instance in the leaf.
    A a default a radial basis function kernel is used in the Scikit SVR implementation.
    """

    def __init__(self, loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=2,
                 min_weight_fraction_leaf=0.,
                 max_depth=9,
                 min_impurity_decrease=0.,
                 min_impurity_split=1e-7,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto'):

        self.__model = None

        super(GradientBoostingRegressor, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            presort=presort)

        self.__gbr = GradientBoostingRegressor(loss=loss,
                                               learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               criterion=criterion,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                                               max_depth=max_depth,
                                               min_impurity_decrease=min_impurity_decrease,
                                               init=init,
                                               subsample=subsample,
                                               max_features=max_features,
                                               min_impurity_split=min_impurity_split,
                                               random_state=random_state,
                                               alpha=alpha,
                                               verbose=verbose,
                                               max_leaf_nodes=max_leaf_nodes,
                                               warm_start=warm_start,
                                               presort=presort)

    @property
    def gbr(self):
        return self.__gbr

    @gbr.setter
    def gbr(self, v):
        self.__gbr = v

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, v):
        self.__model = v

    def fit(self, X, y, sample_weight=None, monitor=None):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.gbr.fit(X, y)

        df = pd.DataFrame(self.gbr.apply(X))

        self.model = SVR()
        self.model.fit(df, y)

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Returns :regression model of the ensamble results
        """
        nmodel = self.gbr.apply(X)
        return self.model.predict(nmodel)
