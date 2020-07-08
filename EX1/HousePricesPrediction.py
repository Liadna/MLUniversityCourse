import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import decomposition, linear_model
import pandas
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

path1 =r'C:\Users\micha\Desktop\train.csv'
pathTest=r'C:\Users\micha\Desktop\test.csv'
dirOut=r'C:\Users\micha\Desktop\pred'
BEST_PCA_COOF=110


# split data into X and y
allCategories=['Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition','SalePrice']
numericNoNA=['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','GarageYrBlt','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']
categorialNoNA=['Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','Heating','HeatingQC','CentralAir','KitchenQual','Functional','PavedDrive','SaleType','SaleCondition']
allCategorial=['Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
allOrdinal=['MSSubClass','MSZoning','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','SaleType']
#array of all numeric columns
allNumericColumns = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']

allCategorial=[cat for cat in allCategorial if cat not in allOrdinal ]
#allCategorial=allCategorial.remove(allOrdinal)

#ordianl arrays by correct order
MSSubClass = [20,30,40,45,50,60,70,80,85,90,120,150,160,180,190]
MSZoning = ["A","C (all)","FV","I","RH","RL","RP","RM"]
ExterQual=["Ex","Gd","TA","Fa","Po"]
ExterCond = ["Ex","Gd","TA","Fa","Po"]
BsmtQual = ["Ex","Gd","TA","Fa","Po","NA"]
BsmtCond = ["Ex","Gd","TA","Fa","Po","NA"]
BsmtExposure = ["Gd","Av","Mn","No","NA"]
BsmtFinType1 = ["GLQ","ALQ","BLC","Rec","LwQ","Unf","NA"]
BsmtFinType2 = ["GLQ","ALQ","BLC","Rec","LwQ","Unf","NA"]
HeatingQC = ["Ex","Gd","TA","Fa","Po"]
CentralAir = ["N","Y"]
Electrical = ["SBrkr","FuseA","FuseF","FuseP","Mix"]
KitchenQual = ["Ex","Gd","TA","Fa","Po"]
Functional = ["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"]
FireplaceQu = ["Ex","Gd","TA","Fa","Po","NA"]
GarageType = ["2Types","Attchd","Basment","BuiltIn","CarPort","Detchd","NA"]
GarageFinish = ["Fin","RFn","Unf","NA"]
GarageQual = ["Ex","Gd","TA","Fa","Po","NA"]
GarageCond = ["Ex","Gd","TA","Fa","Po","NA"]
PavedDrive = ["Y","P","N"]
PoolQC = ["Ex","Gd","TA","Fa","NA"]
Fence = ["GdPrv","MnPrv","GdWo","MnWw","NA"]
SaleType = ["WD","CWD","VWD","New","COD","Con","ConLw","ConLI","ConLD","Oth"]
#end ordinal arrays

#<editor-fold desc="ordinalDict">
ordinalDict=dict()
ordinalDict['MSSubClass']=MSSubClass
ordinalDict['MSZoning']=MSZoning
ordinalDict['ExterQual']=ExterQual
ordinalDict['ExterCond']=ExterCond
ordinalDict['BsmtQual']=BsmtQual
ordinalDict['BsmtCond']=ExterCond
ordinalDict['BsmtExposure']=BsmtExposure
ordinalDict['BsmtFinType1']=BsmtFinType1
ordinalDict['BsmtFinType2']=BsmtFinType2
ordinalDict['HeatingQC']=HeatingQC
ordinalDict['CentralAir']=CentralAir
ordinalDict['Electrical']=Electrical
ordinalDict['KitchenQual']=KitchenQual
ordinalDict['Functional']=Functional
ordinalDict['FireplaceQu']=FireplaceQu
ordinalDict['GarageType']=GarageType
ordinalDict['GarageFinish']=GarageFinish
ordinalDict['GarageQual']=GarageQual
ordinalDict['GarageCond']=GarageCond
ordinalDict['PavedDrive']=PavedDrive
ordinalDict['PoolQC']=PoolQC
ordinalDict['Fence']=Fence
ordinalDict['SaleType']=SaleType

#replace ordinal
def replaceOrd(dataset):
    vecDict=dict()
    for k,v in ordinalDict.items():
        d = dict()
        for val in xrange(len(v)-1,-1,-1):
            d[v[len(v)-val-1]]=val
        vecDict[k]=d
    for column in dataset:
        if column in vecDict.keys():
            d=vecDict[column]
            dataset[column].replace(d,inplace=True)

def getMean(ser):
    count=0.0
    sm=0.0
    for val in ser:
        if val != 'NA':
            count+=1
            sm+=float(val)
    return sm/count

def fixNumericColumns(dataset):
    for numericCategory in allNumericColumns:
        tmpDType = dataset[numericCategory].dtype
        if tmpDType == object:
           dataset[numericCategory]=pandas.to_numeric(dataset[numericCategory],float)
    return dataset

def featureExtraction(dataset):
    for cat in dataset[numericNoNA]:
        mean=getMean(dataset[cat])
        dataset[cat]=dataset[cat].replace('NA', mean)
    fixNumericColumns(dataset)
    dataset[categorialNoNA]=dataset[categorialNoNA].apply(lambda row:row.replace('NA',max(row)),axis=1)
    id=dataset['Id']
    del dataset['Id']
    replaceOrd(dataset)
    dataset=pandas.get_dummies(dataset)
    return id,dataset


def pcaFeatureExtraction(dataset,models=None,rng=range(BEST_PCA_COOF,BEST_PCA_COOF+1,1)):
    dsList = []
    modelLst = []
    if models != None:
        rng = xrange(0, len(models), 1)
    for i in rng:
        if models == None:
            pca = decomposition.PCA(i)
            X = pca.fit_transform(dataset)
            modelLst.append(pca)
        else:
            pca = models[i]
            X=pca.transform(dataset)
            modelLst.append(models[i])
        dsList.append(X)
    return dsList , modelLst

def npToDfCsv(npArray, path,id,colNames):
    i = 1
    for m in npArray:
        df = pandas.DataFrame(data=m, index=id, columns=colNames)
        pandas.DataFrame.to_csv(df, path+str(i))
        i += 1

def getClassifierArray():
    classifiers = []

    classifiers.append(linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))
    classifiers.append(linear_model.Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
                                          normalize=False, random_state=None, solver='auto', tol=0.001))
    classifiers.append(linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
                                            normalize=False))
    classifiers.append(linear_model.Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                                          normalize=False, positive=False, precompute=False, random_state=None,
                                          selection='cyclic', tol=0.0001, warm_start=False))
    classifiers.append(linear_model.LassoLars(alpha=0.1, copy_X=True, fit_intercept=True,
                                              fit_path=True, max_iter=500, normalize=True, positive=False,
                                              precompute='auto', verbose=False))

    classifiers.append(DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=5))
    classifiers.append(KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30))

    classifiers.append(linear_model.ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True,
                                              max_iter=1000, normalize=True, positive=False,
                                              precompute=True))


    classifiers.append(BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
                                     fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
                                     normalize=False, tol=0.001, verbose=False))
    classifiers.append(GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_split=1e-07,
                                                 init=None, random_state=None, max_features=None, alpha=0.9, verbose=0,
                                                 max_leaf_nodes=None, warm_start=True, presort='auto'))
    classifiers.append(RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2))
    classifiers.append(ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2))

    classifiers.append(GridSearchCV(XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear'),
                       {'max_depth': [3, 4, 5],
                        # 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        'n_estimators': [50, 100, 200, 500, 1000]}, verbose=1))
    return classifiers


def applyNeuralNetwork(train_X, train_Y, test_X):
    # Keras Neural Network for regression
    # sh = train_X.shape
    model = Sequential()
    model.add(Dense(100, input_dim=110, activation='relu'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(train_X, train_Y, epochs=200, verbose=1)
    predictions = model.predict(test_X)
    return predictions.flatten()


def printResults(id_list, pred_set, file_num):
    with open('/home/tolik/PycharmProjects/kaggle_house/output/'+"output"+str(file_num)+".csv", 'wb') as resultFile:
        wr = csv.writer(resultFile, delimiter=',')
        wr.writerow(['Id', 'SalePrice'])
        wr.writerows([id_list, pred_set])

def runRegressionModels(classList,train_X,train_Y,test_X):
    predLst=[]
    for c in classList:
        c.fit(train_X, train_Y)
        predictions = c.predict(test_X)
        predLst.append(predictions)

    # Append the predictions of neural network
    predLst.append(applyNeuralNetwork(train_X, train_Y, test_X))
    return predLst

def applyModelsOnAllTestSets(all_trainSets, all_testSets, y_train, classifiers,
                             modelsFnc):
    sets=[]
    for ind in range(0, len(all_trainSets),1):
        mod = modelsFnc(classifiers, all_trainSets[ind], y_train, all_testSets[ind])
        sets.extend(mod)
    return sets

def printAllSets(indexes, predictionSetLst):
    ind = ('Id', indexes.tolist())
    i = 1
    for lst in predictionSetLst:
        sp = ('SalePrice', lst)
        df = pandas.DataFrame.from_items([ind, sp])
        df.to_csv(dirOut+'\pred'+str(i)+'.csv', index=False)
        i += 1


trainset = pandas.read_csv(path1, keep_default_na=False)
testSet = pandas.read_csv(pathTest, keep_default_na=False)

numOfRowsTrainSet = trainset.shape[0]
numOfRowsTestSet = testSet.shape[0]
y_value_train = trainset['SalePrice']
del trainset['SalePrice']
train_test_DataTAble = pandas.concat([trainset, testSet], axis=0)
all_table_index, train_test_DataTAble = featureExtraction(train_test_DataTAble)
testset_indexes = all_table_index.tail(numOfRowsTestSet)
trainset = train_test_DataTAble.head(numOfRowsTrainSet)
testSet = train_test_DataTAble.tail(numOfRowsTestSet)

pcaTrain, modelLst = pcaFeatureExtraction(trainset)
pcaTestSet, modelLst = pcaFeatureExtraction(testSet, modelLst)

classiffiersSet = getClassifierArray()
#classiffiersSet=[]
#classiffiersSet.append(linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))


sets=applyModelsOnAllTestSets(pcaTrain, pcaTestSet, y_value_train, classiffiersSet, runRegressionModels)

printAllSets(testset_indexes, sets)
