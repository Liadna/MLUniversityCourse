import random
import numpy as np
import pandas as pd
import sys
import os.path
import ast
from copy import copy
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CoTrainingClassifier(BaseEstimator):
    """
    CoTraining Classifier. Implemented on top the Base scikit-learn estimator and implements the fit and predict methods.
    Uses semi-supervised approach to train the labeled data and adds new records, previously unlabeled, to the model in each iteration.
    Uses basic learner as a base classifier to train the model.
    """

    def __init__(self, base_learner, n_iter, n_records_iter, v1_indxs, v2_indxs):
        """
        Constructor
        :param base_learner: Base classifier to fit the labeled records
        :param n_iter: Number of iterations to perform in the co-training approach
        :param n_records_iter: Number of records to add in each cross-over
        :param v1_indxs: First set of features to use as a V1 (subset)
        :param v2_indxs: Second set of features to used as V2 (subset)
        """
        self.base_learner = base_learner
        self.n_iter = n_iter
        self.n_records_iter = n_records_iter
        self.v1_indxs = v1_indxs
        self.v2_indxs = v2_indxs

        self.V1 = None
        self.V2 = None
        self.h1 = None
        self.h2 = None

    @staticmethod
    def split_data_features(features_length,  v1_indxs=None, v2_indxs=None, frac_set1=0.5):
        """
        Split the feature set to two subsets (views)
        :param features_length: Total number of features in the original dataset
        :param v1_indxs: Indexes of first view (Default=None)
        :param v2_indxs: Indexes of second view (Default=None)
        :param frac_set1: Fraction for splitting the feature set (Default=0.5)
        :return: Tuple containing two feature subsets (views)
        """
        if not v1_indxs and not v2_indxs:
            all_indxs = range(0, features_length)
            # v1_indxs = np.random.choice(all_indxs, int(frac_set1 * features_length))
            v1_indxs = random.sample(all_indxs, int(frac_set1 * features_length))
            v2_indxs = set(all_indxs) - set(v1_indxs)
        return set(sorted(v1_indxs)), set(sorted(list(v2_indxs)))

    @staticmethod
    def project_feature_set(X, feature_set):
        """
        Project the given view features onto the dataset records
        :param X: The dataset records
        :param feature_set: The set of features for projection
        :return: Projected records
        """
        # first_col_L = X.index.name
        # col_names = [X.columns[i] for i in feature_set]
        df_v1 = X.filter(feature_set, axis=1)

        df_v1['index'] = range(df_v1.shape[0])
        df_v1 = df_v1.reset_index()
        df_v1 = df_v1.set_index('index')
        # if not first_col_L in feature_set:
        del df_v1[df_v1.columns[0]]
        return df_v1

    @staticmethod
    def encode(X):
        """
        Encode the categorical attributes to numeric ones (since schikit-learn accepts only numeric values)
        :param X: Data to encode
        :return: Encoded data
        """
        X.convert_objects(convert_numeric=True)
        le = preprocessing.LabelEncoder()

        for i in range(X.shape[1]):
            if not np.issubdtype(X[X.columns[i]].dtype, np.number):
                X[X.columns[i]] = le.fit_transform(X[X.columns[i]])
        return X

    def get_top_proba(self, proba_array):
        """
        Get top records which were estimated as the most probable by the base learner
        :param proba_array: Array of label probabilities for each test record
        :return: Top most probable records
        """
        for indx, proba_pair in enumerate(proba_array):
            proba_array[indx] = [int(indx), max(proba_pair)]
        y_U2_proba = sorted(proba_array.tolist(), key=lambda x: x[1])
        top_G_indx = [int(r[0]) for r in y_U2_proba][self.n_records_iter * -1:]
        return top_G_indx

    def fit(self, X, y_L):
        """
        Fit the classification to the given data and label vector
        :param X: Data to be fitted in the classifier
        :param y_L: Label vector for supervised classification
        :return: Tuple of two fitted base learners, one for each view of the dataset
        """
        # print("Fitting X")
        X = self.encode(X)

        # X['Class'] = y_L
        X = X.assign(Class=y_L.values)
        X.loc[X.index[X['Class']].tolist()[0] + 0.2 * len(X.loc[:, X.columns == 'Class']):, X.columns == 'Class'] = np.nan

        X_L = X[~X.isnull().any(axis=1)]
        X_L = X_L.loc[:, X_L.columns != 'Class']
        X_U = X[X.isnull().any(axis=1)]
        X_U = X_U.loc[:, X_U.columns != 'Class']

        v1, v2 = self.split_data_features(X_L.shape[1], self.v1_indxs, self.v2_indxs)
        self.V1 = X.columns[list(v1)]
        self.V2 = X.columns[list(v2)]

        X_L1 = self.project_feature_set(X_L, feature_set=self.V1)
        X_L2 = self.project_feature_set(X_L, feature_set=self.V2)
        X_U1 = self.project_feature_set(X_U, feature_set=self.V1)
        X_U2 = self.project_feature_set(X_U, feature_set=self.V2)

        y_L1 = y_L.head(X_L1.shape[0])
        y_L2 = y_L.head(X_L1.shape[0])

        for iteration in range(self.n_iter):
            h1 = copy(self.base_learner)
            h2 = copy(self.base_learner)

            # Classifiers Part
            h1.fit(X_L1, y_L1)
            h2.fit(X_L2, y_L2)

            y_U1 = h1.predict(X_U1)
            y_U1_proba = h1.predict_proba(X_U1)
            # X_U1['Class'] = pd.Series(y_U1)

            y_U2 = h2.predict(X_U2)
            y_U2_proba = h2.predict_proba(X_U2)
            # X_U2['Class'] = pd.Series(y_U2)

            # Select the top-G most confident newly labeled records from U2,
            # and move them from U2 to L1
            top_G_indx = self.get_top_proba(y_U2_proba)
            top_G_records = X.iloc[top_G_indx, :]
            new_add = self.project_feature_set(top_G_records, self.V1)
            X_L1 = X_L1.append(new_add, ignore_index=True)
            X_U2 = X_U2.drop(X_U2.index[top_G_indx])
            X_U2 = X_U2.reset_index()
            X_U2 = X_U2.set_index('index')

            top_G_labels = y_U2[top_G_indx]
            for idx, value in enumerate(top_G_labels):
                y_L1 = y_L1.append({'Class': value}, ignore_index=True)

            # Select the top-G most confident newly labeled records from U1,
            # and move them from U1 to L2
            top_G_indx = self.get_top_proba(y_U1_proba)
            top_G_records = X.iloc[top_G_indx, :]
            new_add = self.project_feature_set(top_G_records, self.V2)
            X_L2 = X_L2.append(new_add, ignore_index=True)
            X_U1 = X_U1.drop(X_U1.index[top_G_indx])
            X_U1 = X_U1.reset_index()
            X_U1 = X_U1.set_index('index')

            top_G_labels = y_U1[top_G_indx]
            for idx, value in enumerate(top_G_labels):
                y_L2 = y_L2.append({'Class': value}, ignore_index=True)

        self.h1 = copy(self.base_learner)
        self.h2 = copy(self.base_learner)
        self.h1.fit(X_L1, y_L1)
        self.h2.fit(X_L2, y_L2)

        return self.h1, self.h2

    def predict(self, X):
        """
        Predict the label of new test records
        :param X: Record to be predicted
        :return: List of predicted labels
        """
        avg_proba = self.predict_proba(X)
        y = []
        for record in avg_proba:
            if record[0] < record[1]:
                y.append(self.h1.classes_[1])
            else:
                y.append(self.h1.classes_[0])
        return y

    def predict_proba(self, X):
        """
        Estimate the probability of each class for each test record
        :param X: Test records to be predicted
        :return: Array of estimated probabilities
        """
        X = self.encode(X)

        proj_V1 = self.project_feature_set(X, self.V1)
        pred_proba_h1 = self.h1.predict_proba(proj_V1)

        proj_V2 = self.project_feature_set(X, self.V2)
        pred_proba_h2 = self.h2.predict_proba(proj_V2)
        avg_proba = (np.array(pred_proba_h1) + np.array(pred_proba_h2)) / 2
        return avg_proba


def run(dataset, v1, v2, base_learner, k, g):
    """
    Perform classification using the Co-Training approach
    :param dataset: Dataset to be learned
    :param v1: The first view/set of features of the data
    :param v2: The second view/set of features of the data
    :param base_learner: Base learning algorithm, e.g. RandomForestClassifier
    :param k: The number of iterations
    :param g: The number of unlabeled records to add in each iteration
    :return: None
    """
    # datasets = [r'C:\Users\micha\Downloads\adult1.csv',
    #            r'C:\Users\micha\Downloads\Airlines_shuffled.csv',
    #            r'C:\Users\micha\Downloads\elecNormNew.csv',
    #            r'C:\Users\micha\Downloads\hyperplane.csv',
    #            r'C:\Users\micha\Downloads\sea.csv']
    #
    # v_sets = [([0, 5, 8, 9, 7, 13], [1, 2, 3, 4, 6, 10, 11, 12]),
    #           ([0, 4, 5, 6], [1, 2, 3]),
    #           ([0, 1, 2], [3, 4, 5, 6, 7]),
    #           ([0, 2, 4, 6, 8],[1, 3, 5, 7, 9]),
    #           ([0, 2], [1])]

    #for i, dataset in enumerate(datasets):
    df = pd.DataFrame.from_csv(dataset, index_col=None)
    # unlabeled_frac = 0.8

    X_train = df.loc[:, df.columns != 'Class']
    y_target = df.loc[:, df.columns == 'Class']

    # y_train = df.loc[:, df.columns == 'Class']
    # X_test = df.loc[:3, df.columns != 'Class']

    # dtc = DecisionTreeClassifier(max_depth=5)
    rfc = RandomForestClassifier(n_estimators=100,
                                 criterion='gini',
                                 max_depth=5,
                                 min_samples_split=2,
                                 min_samples_leaf=1)
    # svc = SVC(probability=True)
    lrc = LogisticRegression(penalty='l2',  C=1.0, max_iter=100)

    h = rfc if base_learner == "RandomForestClassifier" else lrc
    v1 = ast.literal_eval(v1)

    v2 = ast.literal_eval(v2)

    # CoTraining classifier
    cotrc = CoTrainingClassifier(#base_learner=rfc,
                                 #n_iter=100,
                                 #n_records_iter=20,
                                 base_learner=h,
                                 n_iter=int(k),
                                 n_records_iter=int(g),
                                 v1_indxs=v1,
                                 v2_indxs=v2)
                                 # v1_indxs=v_sets[i][0],
                                 # v2_indxs=v_sets[i][1])
    # h1, h2 = cotrc.fit(X_train, y_train)
    # pred_labels = cotrc.predict(X_test)
    y_target = cotrc.encode(y_target)
    encoded_X = cotrc.encode(X_train)

    # kfold = model_selection.KFold(n_splits=10, random_state=7)
    # scores = cross_val_score(cotrc, X_train, y_target, cv=kfold, scoring='roc_auc')
    scoring = ['roc_auc', 'f1_micro']
    scores = cross_validate(cotrc, X_train, y_target, cv=10, scoring=scoring)
    print("============================================\n"
          "Dataset={0}\n"
          "Base Learner={1}\n"
          "K={2}, G={3}\n"
          "Average training time= {4} seconds\n"
          "Average testing time= {5} seconds\n" 
          "Mean AUC= {6}\n"
          "Mean F1-score= {7}".format(dataset.split('\\')[-1],
                                      base_learner,
                                      cotrc.n_iter, cotrc.n_records_iter,
                                      np.mean(scores['fit_time']),
                                      np.mean(scores['score_time']),
                                      np.mean(scores['test_roc_auc']),
                                      np.mean(scores['test_f1_micro'])))


def main():
    if len(sys.argv) == 7:
        if os.path.exists(sys.argv[1]):
            if (sys.argv[2][0] == '[') and (sys.argv[2][len(sys.argv[2]) - 1] == ']') and \
                    (sys.argv[2][0] == '[' and sys.argv[2][1] != ']'):
                if (sys.argv[3][0] == '[') and (sys.argv[3][len(sys.argv[3]) - 1] == ']') and \
                        (sys.argv[3][0] == '[' and sys.argv[3][1] != ']'):
                    if sys.argv[4] in ['RandomForestClassifier', 'LogisticRegression']:
                        if str(sys.argv[5]).isdigit() and str(sys.argv[5]) > '0':
                            if str(sys.argv[6]).isdigit() and str(sys.argv[6]) > '0':
                                run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                                    sys.argv[6])
                            else:
                                print "Please enter a positive numeric value for G"
                        else:
                            print "Please enter a positive numeric value for K"
                    else:
                        print "Please chose one H algorithm name from this list: [RandomForestClassifier, LogisticRegression]"
                else:
                    print "Please enter V2 as a list structure ['Index_Feature_1','Index_Feature_2',...] without whitespaces. Index starts at 0"
            else:
                print "Please enter V1 as a list structure ['Index_Feature_1','Feature_2',...] without whitespaces. Index starts at 0"
        else:
            print "The dataset file doesn't Exist. Please enter legal path!"
    else:
        print "Number of inputted parameters: " + str(len(sys.argv))
        print "Please write the command in this order: CoTraining.py Dataset_path.csv [V1_list_feature_indexes] [V2_list_features_indexes] h_algorithm K_iterations G_records"


if __name__ == "__main__":
    main()
