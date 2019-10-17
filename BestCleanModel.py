#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing all modules needed
import numpy as np

import pandas as pd
from pandas import Series, DataFrame

from sklearn import datasets
from sklearn import linear_model
from sklearn import neural_network
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn import decomposition
from sklearn.decomposition import FactorAnalysis, PCA

from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression

from scipy.stats import zscore
from IPython.display import Image
from IPython.core.display import HTML 
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb


# In[ ]:


#Import training/test data
trainingData ='D:/PythonWorkspace/Machine Learning/Competition 1/FirstModel/tcd ml 2019-20 income prediction training (with labels).csv'
#Import data to predict
submissionData ='D:/PythonWorkspace/Machine Learning/Competition 1/FirstModel/tcd ml 2019-20 income prediction test (without labels).csv'

#Read in (training/test) the data
instances = pd.read_csv(trainingData, engine='python')

#Read in the data to predict
submission = pd.read_csv(submissionData, engine='python')

#Column lables
features = ["Instance",
            "YearOfRecord",
            "Gender",
            "Age",
            "Country",
            "SizeOfCity",
            "Profession",
            "UniversityDegree",
            "WearsGlasses",
            "HairColor",
            "BodyHeight",
            "Income"]

#Lable columns
def lable_columns(Dataframe, features):
    Dataframe.columns = features
    return Dataframe
    

lable_columns(instances, features)
lable_columns(submission, features)


# In[ ]:


#Data preparation
#Filling in nan values

#Finds the mean of every column (Not using it as median is mroe accurate)
def meanOfAllColumns(Dataframe):
    meanOfAllColumns = Dataframe.mean(axis=0, skipna = True)
    return meanOfAllColumns

#meansOfInstances = meanOfAllColumns(instances)
#display(meansOfInstances)


# In[ ]:


#Finds the median of every column (Not using it as median is mroe accurate)
def medianOfAllColumns(Dataframe):
    medianOfAllColumns = Dataframe.median(axis=0, skipna = True)
    return medianOfAllColumns

medianOfInstances = medianOfAllColumns(instances)
medianIncome = medianOfInstances[6]


# In[ ]:


#Gender: other, unknown and zero, all have almost the same mean height therfore we can replace values
def fixGenderColumn(Dataframe):
    #Change zero to unknown as they mean the same thing
    Dataframe["Gender"] = Dataframe["Gender"].replace(["0"], "unknown")
    #Change all unknown to NaN so we can predict the gender based on height
    Dataframe["Gender"] = Dataframe["Gender"].replace(["unknown"], np.NaN )
    return Dataframe

# #Then fills the NaN values in Gender based on their hight
def fillGenderbasedOnHeight(Dataframe):
    #Find all Gender instances with NaN
    emptyGenders = Dataframe[Dataframe['Gender'].isnull()]
    femaleBins = (0,175, 220)
    groupFemaleBodyHeight = ['female', 'male' ]
    emptyGenders['Gender'] = pd.cut(emptyGenders['BodyHeight'], femaleBins, labels=groupFemaleBodyHeight)
    Dataframe = Dataframe.update(emptyGenders)
    return Dataframe
    
fixGenderColumn(instances)
fixGenderColumn(submission)    
fillGenderbasedOnHeight(instances)
fillGenderbasedOnHeight(submission)

instances.Instance = instances.Instance.astype(int)
submission.Instance = submission.Instance.astype(int)


# In[ ]:


#Finds the mean hight of a woman and a man. 
def shortWomenTallMen(Dataframe):
    meanHeightPerGender = Dataframe.groupby('Gender')['BodyHeight'].mean()
    return meanHeightPerGender

#Array with mean height per gender
meanHeightPerGenderI = shortWomenTallMen(instances)
meanHeightPerGenderS = shortWomenTallMen(submission)


# In[ ]:


#Fill in Nan Values based on the median and modes.
def fillNan(Dataframe, arrayOfMeans):
    meanYear = arrayOfMeans[1]
    meanGender = Dataframe.Gender.mode()
    meanAge = arrayOfMeans[2]
    meanCountry = Dataframe.Country.mode()
    meanSizeOfCity = arrayOfMeans[3]
    meanProfession = Dataframe.Profession.mode()
    meanUniversityDegree = Dataframe.UniversityDegree.mode()
    
    meanValues = ({"YearOfRecord":meanYear,
           'Gender':meanGender[0],
           "Age":meanAge,
           "Country":meanCountry[0],
           "SizeOfCity":meanSizeOfCity,
           "Profession":meanProfession[0],
           "UniversityDegree":meanUniversityDegree[0],
            })
    
    Dataframe = Dataframe.fillna(value=meanValues, inplace = True)
    return Dataframe

fillNan(instances, medianOfInstances)
fillNan(submission, medianOfInstances)


# In[ ]:


# Ordinal/LableEncoding would make sense here however mean income differen per degree is minimal
def ordinalLable_UniversityDegree(Dataframe):
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["No"], 0)
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["Bachelor"], 1)
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["Master"], 2)
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["PhD"], 3)
    return Dataframe

def UniversityHotLable(Dataframe):
    justIncome = Dataframe['Income']
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["0"], "No")
    genderDummies = pd.get_dummies(Dataframe['UniversityDegree'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["UniversityDegree","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe


#ordinalLable_UniversityDegree(instances)
#ordinalLable_UniversityDegree(submission)
#display(instances.head())
#display(submission.head())

# Target lable University degree
meanUD = instances.groupby('UniversityDegree')['Income'].mean()
def targetLable_UniversityDegree(Dataframe, mean):
    Dataframe["UniversityDegree"] = Dataframe["UniversityDegree"].replace(["0"], "No")
    Dataframe['UniversityDegree'] = Dataframe['UniversityDegree'].map(mean)
    return Dataframe

# instances = targetLable_UniversityDegree(instances, meanUD)
# submission = targetLable_UniversityDegree(submission, meanUD)

instances = UniversityHotLable(instances)
submission = UniversityHotLable(submission)


# In[ ]:


#Drop HairColor as correlation too low
def drop_Hair(Dataframe):
    Dataframe = Dataframe.drop(["HairColor"],axis=1,inplace=True)
    return Dataframe

#Drop Wears glasses as mean is 0.5 + low correlation
def drop_WearsGlasses(Dataframe):
    Dataframe = Dataframe.drop(["WearsGlasses"],axis=1, inplace=True)
    return Dataframe

drop_Hair(instances)
drop_Hair(submission)

drop_WearsGlasses(instances)
drop_WearsGlasses(submission)


# In[ ]:


def genderHotLable(Dataframe):
    justIncome = Dataframe['Income']
    Dataframe["Gender"] = Dataframe["Gender"].replace(["0"], "other")
    genderDummies = pd.get_dummies(Dataframe['Gender'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["Gender","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe


meanGenderIncome = instances.groupby('Gender')['Income'].mean()
def targetLable_Gender(Dataframe, mean):
    Dataframe['Gender'] = Dataframe['Gender'].map(meanGenderIncome)
    return Dataframe

instances = genderHotLable(instances)
submission = genderHotLable(submission)
# instances = targetLable_Gender(instances, meanGenderIncome)
# submission = targetLable_Gender(submission, meanGenderIncome)


# In[ ]:


def groupByAge(Dataframe):
    bins = (0,10,20,30,40,50,60,70,80,90,100,110,120,130)
    groupAge = ['0+','10+','20+','30+','40+','50+','60+','70+','80+','90+','100+','110+','120+']
    Dataframe['Age'] = pd.cut(Dataframe['Age'], bins, labels=groupAge)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['Age'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["Age","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe

def groupByAgeQuarter(Dataframe):
    bins = (0,24,35,48,120)
    groupAge = ['0-24','24-35','35-48','48-120']
    Dataframe['Age'] = pd.cut(Dataframe['Age'], bins, labels=groupAge)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['Age'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["Age","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe
    
instances = groupByAge(instances)
submission = groupByAge(submission)


# In[ ]:


def groupByYearOfRecord(Dataframe):
    bins = (1970,1980,1990,2000,2010,2020)
    groupYearOfRecord = ['1970+','1980+','1990+','2000+','2010+']
    Dataframe['YearOfRecord'] = pd.cut(Dataframe['YearOfRecord'], bins, labels=groupYearOfRecord)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['YearOfRecord'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["YearOfRecord","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe
    
instances = groupByYearOfRecord(instances)
submission = groupByYearOfRecord(submission)


# In[ ]:


def groupByBodyHeight(Dataframe):
    bins = (90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260)
    groupBodyHeight = ['90+','100+','110+','120+','130+','140+','150+','160+','170+','180+','190+','200+','210+','220+'
                      ,'230+','240+','250+']
    Dataframe['BodyHeight'] = pd.cut(Dataframe['BodyHeight'], bins, labels=groupBodyHeight)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['BodyHeight'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["BodyHeight","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe

def groupByBodyHeightQuarter(Dataframe):
    bins = (0,160,173,190,265)
    groupBodyHeight = ['0-169','169-173','173-190','190-265']
    Dataframe['BodyHeight'] = pd.cut(Dataframe['BodyHeight'], bins, labels=groupBodyHeight)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['BodyHeight'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["BodyHeight","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe


def drop_BodyHeight(Dataframe):
    Dataframe = Dataframe.drop(["BodyHeight"],axis=1, inplace=True)
    return Dataframe


instances = groupByBodyHeightQuarter(instances)
submission = groupByBodyHeightQuarter(submission)

# drop_BodyHeight(instances)
# drop_BodyHeight(submission)


# In[ ]:


def groupBySizeOfCity(Dataframe):
    bins = (0, 1000, 10000, 100000, 300000, 1000000, 3000000, 10000000, 50000000)
    groupBodyHeight = ['Village','Town','Large Town', 'City', 'Large City', 'Metropolis', 'Conurbation', 'Megalopolis']
    Dataframe['SizeOfCity'] = pd.cut(Dataframe['SizeOfCity'], bins, labels=groupBodyHeight)
    
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['SizeOfCity'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["SizeOfCity","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe
    
instances = groupBySizeOfCity(instances)
submission = groupBySizeOfCity(submission)


# In[ ]:


#Getting max and min of each colum to decide on grouping ranges

meanCountryIncome = instances.groupby('Country')['Income'].mean()
def targetLable_Country(Dataframe, mean):
    Dataframe['Country'] = Dataframe['Country'].map(mean)
    return Dataframe

def hotLableCountry(Dataframe):
    justIncome = Dataframe['Income']
    genderDummies = pd.get_dummies(Dataframe['Country'], prefix = 'dum')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["Country","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe


instances = targetLable_Country(instances, meanCountryIncome)
instances['Country'] = instances['Country'].fillna(value=medianIncome)
submission = targetLable_Country(submission, meanCountryIncome)
submission['Country'] = submission['Country'].fillna(value=medianIncome)

instancesUniqueCountry = pd.DataFrame({'Country': instances.Country.unique()})
submissionUniqueCountry = pd.DataFrame({'Country': submission.Country.unique()})
#countryDifference = (np.setdiff1d(instancesUniqueCountry, submissionUniqueCountry))
countryDataframe = pd.concat([instancesUniqueCountry,submissionUniqueCountry], axis= 0, join='outer')
allCountires = countryDataframe.Country.unique()
# instances = hotLableCountry(instances)
# submission = hotLableCountry(submission)


def hotLableCountry(Dataframe, allCountires):
    justIncome = Dataframe['Income']
    Dataframe['Country'] = Dataframe['Country'].astype('category', categories= allCountires)
    genderDummies=pd.get_dummies(Dataframe['Country'],prefix='dum').astype('int')
    Dataframe = pd.concat([Dataframe,genderDummies], axis= 1, join='outer')
    Dataframe = Dataframe.drop(["Country","Income"],axis=1)
    Dataframe = pd.concat([Dataframe,justIncome], axis= 1, join='outer')
    return Dataframe

# instances = hotLableCountry(instances, allCountires)
# submission = hotLableCountry(submission, allCountires)


# In[ ]:


meanProfessionIncome = instances.groupby('Profession')['Income'].mean()
def targetLable_Profession(Dataframe, mean):
    Dataframe['Profession'] = Dataframe['Profession'].map(mean)
    return Dataframe


instances = targetLable_Profession(instances, meanProfessionIncome)
instances['Profession'] = instances['Profession'].fillna(value=medianIncome)
submission = targetLable_Profession(submission, meanProfessionIncome)
submission['Profession'] = submission['Profession'].fillna(value=medianIncome)
#submission = submission.fillna(meanIncome)

#instances = groupByMeanProfessionIncome(instances, meanProfessionLable)
#submission = groupByMeanProfessionIncome(submission, meanProfessionLable)


# In[ ]:


# #Feature selection
# #Using Pearson Correlation
# plt.figure(figsize=(25,25))
#cor = instances.corr(method="spearman")
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()


# In[ ]:


# #Correlation with output variable
# cor_target = abs(cor["Income"])

# #Selecting highly correlated features
# relevant_features1 = cor_target[cor_target>0.05]


# display(relevant_features1)


# In[ ]:


# print(instances[["Country",
#           "Profession",
#           "dum_10+",
#           'dum_1980+',
#           'dum_2010+',
#           'dum_Large Town',
#           'dum_Metropolis',
#           'Income'
#                 ]].corr())


# In[ ]:


numberOfColumns = (len(instances.columns) - 1)


# In[ ]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_regression, k=10)

X = instances.iloc[:,1:numberOfColumns] #independent columns
Y= instances.iloc[:,-1]    #target column i.e price range
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
pd.options.display.max_rows = 220


# In[ ]:


featuresToUse = featureScores.nlargest(200,'Score').as_matrix(columns=featureScores.nlargest(200,'Score').columns[0:1])

#display(featuresToUse)
def featureSelectionBasedOnScore(Dataframe):
    
    DataframeInstances = Dataframe['Instance']
    DataframeIncome = Dataframe['Income']
    Dataframe = Dataframe[featuresToUse.flatten()]
    
    Dataframe = pd.concat([DataframeInstances,Dataframe], axis= 1, join='outer')
    Dataframe = pd.concat([Dataframe,DataframeIncome], axis= 1, join='outer')
    return Dataframe

instances = instances.drop(["dum_City","dum_Conurbation", "dum_Megalopolis" ],axis=1)
submission = submission.drop(["dum_City", "dum_Conurbation", "dum_Megalopolis"],axis=1)
instances = instances.drop(["dum_0+","dum_120+","dum_110+","dum_100+","dum_30+"],axis=1)
submission = submission.drop(["dum_0+","dum_120+","dum_110+","dum_100+","dum_30+"],axis=1)
instances = instances.drop(["dum_169-173"],axis=1)
submission = submission.drop(["dum_169-173"],axis=1)

# instances = featureSelectionBasedOnScore(instances)
# submission = featureSelectionBasedOnScore(submission)

instances.Instance = instances.Instance.astype(int)
submission.Instance = submission.Instance.astype(int)

numberOfColumns = (len(instances.columns) - 1)


# In[ ]:


# # #Feature selection
# # #Using Pearson Correlation
# # plt.figure(figsize=(25,25))
# # cor = instances.corr(method="spearman")
# # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# # plt.show()

# instances = instances.drop(["dum_City","dum_Conurbation", "dum_Megalopolis" ],axis=1)
# submission = submission.drop(["dum_City", "dum_Conurbation", "dum_Megalopolis"],axis=1)
# instances = instances.drop(["dum_110+","dum_100+","dum_30+"],axis=1)
# submission = submission.drop(["dum_110+","dum_100+","dum_30+"],axis=1)
# instances = instances.drop(["dum_169-173"],axis=1)
# submission = submission.drop(["dum_169-173"],axis=1)

# instances = instances.drop(["Profession"],axis=1)
# submission = submission.drop(["Profession"],axis=1)

numberOfColumns = (len(instances.columns) - 1)


# In[ ]:


#Separate

X_prime = instances.ix[:,1:numberOfColumns].values

y = instances.ix[:,numberOfColumns].values

X_primeSubmittion = submission.ix[:,1:numberOfColumns].values

ySubmittion = submission.ix[:,numberOfColumns].values

what = instances.iloc[:,1:numberOfColumns]    #target column i.e price range
who = instances.iloc[:,numberOfColumns]    #target column i.e price range


# In[ ]:


# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_prime, y, test_size=.33)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_primeSubmittionScaler = scaler.transform(X_primeSubmittion)

# Model Selection
regr = neural_network.MLPRegressor(solver='lbfgs',
                                   learning_rate='constant',
                                   activation ='relu',
                                   verbose = True,
                                   shuffle = False,
                                   hidden_layer_sizes=(100,100,100),
                                   alpha = 1,
                                   early_stopping = True
                                  )

models = [regr]

def get_cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=5, )
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
    
# loop through list of models
for model in models:
    print(model)
    #get_cv_scores(model)

# Train the model using the training sets
regr.fit(X_train, y_train)


# Make predictions using the testing set

y_pred = regr.predict(X_test)
y_predSub = regr.predict(X_primeSubmittionScaler)


# # # The coefficients
# print('Coefficients: \n', regr.coefs_)

# # The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


print("Best score using built-in MLP score: %f" %regr.score(X_train, y_train))


# In[ ]:


# activation=['identity', 'logistic', 'tanh', 'relu'] 
alpha = [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
beta_1 = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# beta_2 = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# epsilon = [  0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# early_stopping=[True]
# learning_rate=['constant','invscaling','adaptive']
# learning_rate_init = [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
# max_iter=[100,200,300,400,500]
# momentum = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# power_t = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# solver=['lbfgs', 'sgd', 'adam']
# tol=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# n_iter=[80,90,100,110,120,130,140]


param_grid = {
#                     'activation': activation,
#                     'hidden_layer_sizes':[(100,100,100)],
#                     'alpha' : alpha,
                     'beta_1' : beta_1,
#                     'beta_2' : beta_2,
#                     'epsilon' : epsilon,
#                      'early_stopping': early_stopping,
#                     'learning_rate': ['constant'],
#                      'learning_rate_init' : learning_rate,
#                     'max_iter' : max_iter,
#                     'momentum' : momentum,
#                     'power_t' : power_t,
#                     'shuffle':[False],
#                     'solver' : solver,
#                      'tol' : tol,
#                     'n_iter': n_iter
}


# In[ ]:


# parameters = {'solver': ['lbfgs'],
#               'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ],
#               'alpha': 10.0 ** -np.arange(1, 10),
#               'hidden_layer_sizes':np.arange(10, 15),
#               'random_state':[0,1,2,3,4,5,6,7,8,9]
#              }
clf = GridSearchCV(neural_network.MLPRegressor(), param_grid, n_jobs=-1, verbose=1)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.best_params_)


# In[ ]:


# random = RandomizedSearchCV(estimator=neural_network.MLPRegressor(), 
#                             param_distributions=param_grid,  
#                             verbose=1, 
#                             n_jobs=-1, 
#                             n_iter=10)

# random_result = random.fit(X_train, y_train)

# print('Best Score: ', random_result.best_score_)
# print('Best Params: ', random_result.best_params_)


# In[ ]:


df = pd.DataFrame({"Instance":submission["Instance"], "Income":y_predSub})


# In[ ]:


df.to_csv(r'D:/PythonWorkspace/Machine Learning/Competition 1/FirstModel/tcd ml 2019-20 income prediction submission file.csv.csv', index=False)


# In[ ]:


checkFile ='D:/PythonWorkspace/Machine Learning/Competition 1/FirstModel/tcd ml 2019-20 income prediction submission file.csv.csv'

#Read in training/test the data
checkFileDF = pd.read_csv(checkFile, engine='python')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




