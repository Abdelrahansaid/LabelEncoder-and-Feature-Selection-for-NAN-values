#!/usr/bin/env python
# coding: utf-8

# 1) [Discover & clean data:) ](#t1.)
# 
# 2) [Prepare data  <:](#t2.)
# 
#   ##### ....We tested 3 moduls:
# 3) [1. HistGradientBoostingRegressor ](#t3.)
# 
# 4) [2. BaggingRegressor ](#t4.)
# 
# 5) [3. DecisionTreeRegressor ](#t5.)

# In[59]:


import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats


# In[60]:


df_train = pd.read_csv('C:\\Users\\Abdelrahman\\projects\\Kaggle Projects\\Hous_prices\\Data\\train.csv')

df_test = pd.read_csv('C:\\Users\\Abdelrahman\\projects\\Kaggle Projects\\Hous_prices\\Data\\test.csv')

df_train.shape


# <a id="t1."></a>
# # Discover & clean data :)

# In[61]:


df_train.head()


# In[62]:


df_train.duplicated().sum()


# In[63]:


df_train.nunique()


# In[64]:


df_train.isna().sum()


# In[65]:


#cheak false values in salesprice 

indexNames = df_train[ df_train['SalePrice']<0  ].index
indexNames


# ### transform data  to nums using LabelEncoder

# ##### tranform (train data)

# In[66]:


import pandas as pd

data =df_train
string_columns = data.select_dtypes(include='object').columns

print(string_columns)


# In[67]:


# LabelEncder function

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_features(data, categorical_features):
    # Create a LabelEncoder instance
    encoder = LabelEncoder()

    # Transform each categorical feature
    for feature in categorical_features:
        # Encode the feature values
        encoded_values = encoder.fit_transform(data[feature])

        # Replace the original feature values with the encoded values
        data[feature] = encoded_values

    return data


# In[68]:


# Identify the categorical features
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']

# Encode the categorical features
encoded_data = label_encode_features(df_train, categorical_features)
df_train.head()


# In[69]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### remove( LotFrontage ,GarageYrBlt ,MasVnrArea)
# 

# In[70]:


#dealing with missing data
df_train = df_train.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index, axis=0)
df_train.isnull().sum().max()
   #just checking that there's no missing data missing...


# In[71]:


df_train.shape


# ## Split 

# In[72]:


X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

X.head()


# In[73]:


y.head()


# ### Tranform (test data)

# In[74]:


import pandas as pd

data =df_test
string_columns = data.select_dtypes(include='object').columns

print(string_columns)


# In[75]:


# Load the data
df_test = df_test

# Identify the categorical features
categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']

# Encode the categorical features
encoded_data = label_encode_features(df_test, categorical_features)


# <a id="t2."></a>
# # Prepare data <:

# ### Deal with null/NAN values for feature selection

# ### treat wiht NAN value  to use  feature selection 
# #####   does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
# #### we have three algo 
# *.*  BaggingRegressor
# 
# *.* DecisionTreeRegressor
# 
# *.* HistGradientBoostingRegressor

# In[76]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if df_train is not None or empty
if df_train is not None and len(df_train) > 0:
    # Create the correlation matrix
    corrmat = df_train.corr()

    # Create the heatmap
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()
else:
    print('df_train is either None or empty. Please load or create data before calculating the correlation matrix.')
    
#++NOTE++ the id colum have to be exist to make the corr graph


# ##### Another thing that got my attention was the 'SalePrice' correlations. We can see our well-known 'GrLivArea', 'TotalBsmtSF', and 'OverallQual' saying a big 'Hi!', but we can also see many other variables that should be taken into account. That's what we will do next. i will select some feature and see

# In[77]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Select the variables you want to include in the correlation matrix
cols = ['SalePrice','LotArea', 'Neighborhood', 'OverallQual', 'YearBuilt', 
        'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF','GrLivArea','GarageCars','GarageArea']

# Calculate the correlation matrix
cm = np.corrcoef(df_train[cols].values.T)

# Create a heatmap of the correlation matrix
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, yticklabels=cols, xticklabels=cols)
plt.show()


# In[78]:


#scatterplot
sns.set()
cols =  ['SalePrice','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF','GrLivArea','GarageCars','GarageArea']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# In[79]:


# Create the normal probability plots
for col in cols:
    stats.probplot(df_train[col], plot=plt)
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Sample quantiles')
    plt.title('Normal Probability Plot for Feature: {}'.format(col))
    plt.show()


# Ok, thier is some feature is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.
# 
# But everything's not lost. A simple data transformation can solve the problem. This is one of the awesome things you can learn in statistical books: in case of positive skewness, log transformations usually works well. When I discovered this, I felt like an Hogwarts' student discovering a new cool spell.

# # 1stFlrSF

# In[80]:


#histogram and normal probability plot
sns.distplot(df_train['1stFlrSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['1stFlrSF'], plot=plt)


# In[81]:


#applying log transformation for 1stFlrSF
df_train['1stFlrSF'] = np.log(df_train['1stFlrSF'])


# In[82]:


sns.distplot(df_train['1stFlrSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['1stFlrSF'], plot=plt)


# #### Ok, now we are dealing with the big boss. What do we have here?
# Something that, in general, presents skewness.
# A significant number of observations with value zero (houses without basement).
# A big problem because the value zero doesn't allow us to do log transformations.
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# # SalePrice

# In[83]:


#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[84]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[85]:


#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# # GrLivArea

# In[86]:


#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# In[87]:


#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


# In[88]:


#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# # TotalBsmtSF

# In[89]:


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)


# # HasBsmt
# ##### create column for new variable (one is enough because it's a binary categorical feature)      if area>0 it gets 1, for area==0 it gets 0

# In[90]:


df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


# In[91]:


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[92]:


#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


# In[93]:


df_train.head()


# In[94]:


X = df_train.drop(['SalePrice'], axis = 1)
y = df_train['SalePrice']
X.head()


# ## Do the same (log) to Test Data 

# In[95]:


#applying log transformation for 1stFlrSF
df_train['1stFlrSF'] = np.log(df_train['1stFlrSF'])

# transform GrLivArea
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

#-----------------------------------------

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0 
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1

#transform HasBsmt
df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])


# <a id="t3."></a>
# # HistGradientBoostingRegressor

# In[96]:


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# Create the estimator
estimator = HistGradientBoostingRegressor()

# Fit the estimator
estimator.fit(X, y)

# Define a custom importance_getter function
def my_importance_getter(estimator):
    # Compute feature importances using permutation importance
    importances = permutation_importance(estimator, X, y, n_repeats=10, random_state=0)

    # Extract permutation importance scores
    scores = importances['importances_mean']

    # Convert scores to a numerical array
    scores = np.asarray(scores)

    # Apply absolute value function
    scores = np.abs(scores)

    
    return scores

# Create the feature selector
selector = SelectFromModel(estimator=estimator, importance_getter=my_importance_getter, max_features=None)

# Transform the data
#X = selector.fit_transform(X, y)

# Show the X dimension and selected features
print('X Shape is:', X.shape)
print('Selected Features are:', selector.get_support())

X.columns


# In[97]:


# Get the selected features
selected_features = selector.get_support()

# Print the selected feature names
selected_feature_names = X.columns[selected_features]
print("Selected Feature Names:")
for name in selected_feature_names:

    print(name)


# In[98]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Select the variables you want to include in the correlation matrix
cols = ['SalePrice','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF','GrLivArea','GarageCars','GarageArea']

# Calculate the correlation matrix
cm = np.corrcoef(df_train[cols].values.T)

# Create a heatmap of the correlation matrix
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, yticklabels=cols, xticklabels=cols)
plt.show()


# In[99]:


Hist_X_train = df_train[ ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                        'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF','GrLivArea',
                          'GarageCars','GarageArea']]


y_df_train = df_train['SalePrice']

Hist_X_train.head()


# In[100]:


y_df_train.head()


# ### Split Test Data besd on selected features

# In[101]:


Hist_test=df_test[ ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                   'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF','GrLivArea',
                    'GarageCars','GarageArea']]


Hist_test.head()


# In[102]:


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error 
#----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(Hist_X_train , y_df_train , test_size=0.2, random_state=0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('scalar', StandardScaler()),
    ('model', HistGradientBoostingRegressor(max_iter=500))
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))


#Calculating Details
print('Pipeline Model Train Score is : ' , pipeline.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , pipeline.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = pipeline.predict(X_test)
print('Predicted Value for Pipeline Model is : ' , y_pred[:5])

print("---------------------------------------------------")

#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )

 
print('----------------------------------------------------------------------------------')
Hist_pred_data = pipeline.predict(Hist_test)
print('Predicted Value fpr Test data : ' , Hist_pred_data[:10])


model = HistGradientBoostingRegressor(max_iter=100)
# Train the model with different numbers of training iterations
training_errors = []
testing_errors = []
for num_iterations in range(1, 101):
    model.fit(X_train[:num_iterations], y_train[:num_iterations])
    training_errors.append(mean_squared_error(model.predict(X_train), y_train))
    testing_errors.append(mean_squared_error(model.predict(X_test), y_test))

# Plot the learning curves
plt.plot(range(1, 101), training_errors, label='Training Error')
plt.plot(range(1, 101), testing_errors, label='Testing Error')
plt.xlabel('Number of training iterations')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()


# ### Training and testing error: Compare the error on the training data to the error on the testing data.  if the error on the training data is much lower than the error on the testing data, this is a sign that the model is overfitting.

# In[103]:


#get exp to predicted SalePrice to invers the (log)
pred_prise= np.exp(Hist_pred_data)

print(pred_prise[:10])


# 
# <a id="t4."></a>
# # BaggingRegressor

# In[104]:


from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# Create the estimator
estimator = BaggingRegressor()

# Fit the estimator
estimator.fit(X, y)

# Define a custom importance_getter function
def my_importance_getter(estimator):
    # Compute feature importances using permutation importance
    importances = permutation_importance(estimator, X, y, n_repeats=10, random_state=0)

    # Extract permutation importance scores
    scores = importances['importances_mean']

    # Convert scores to a numerical array
    scores = np.asarray(scores)

    # Apply absolute value function
    scores = np.abs(scores)

    return scores

# Create the feature selector
selector = SelectFromModel(estimator=estimator, importance_getter=my_importance_getter, max_features=None)

# Transform the data
#X = selector.fit_transform(X, y)

# Show the X dimension and selected features
print('X Shape is:', X.shape)

# Get the selected features
selected_features = selector.get_support()

# Print the selected feature names
selected_feature_names = X.columns[selected_features]
print("Selected Feature Names:")
for name in selected_feature_names:

    print(name)


# In[105]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Select the variables you want to include in the correlation matrix
cols = ['SalePrice','LotArea', 'OverallQual', 'YearBuilt',
        'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF','CentralAir','1stFlrSF','GrLivArea','GarageCars','GarageArea']

# Calculate the correlation matrix
cm = np.corrcoef(df_train[cols].values.T)

# Create a heatmap of the correlation matrix
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols, xticklabels=cols)
plt.show()


# In[106]:


# selct cols in train data

Bag_X = df_train[[ 'LotArea', 'OverallQual', 'YearBuilt',
                  'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF','CentralAir',
                  '1stFlrSF','GrLivArea','GarageCars','GarageArea']]



Bag_X.head()


# In[107]:


# selct cols in test data

Bag_test=df_test[[ 'LotArea', 'OverallQual', 'YearBuilt',
                'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF','CentralAir','1stFlrSF',
                  'GrLivArea','GarageCars','GarageArea']]


Bag_test.head()


# In[108]:


from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error 
#----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(Bag_X, y_df_train, test_size=0.2, random_state=0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('scalar', StandardScaler()),
    ('model', BaggingRegressor(n_estimators=50))
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))


#Calculating Details
print('Pipeline Model Train Score is : ' , pipeline.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , pipeline.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = pipeline.predict(X_test)
print('Predicted Value for Pipeline Model is : ' , y_pred[:10])


#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )

 
print('----------------------------------------------------------------------------------')
Bag_pred_data = pipeline.predict(Bag_test)
print('Predicted Value fpr Test data : ' , Bag_pred_data[:10])


model = BaggingRegressor(n_estimators=50)
# Train the model with different numbers of training iterations
training_errors = []
testing_errors = []
for num_iterations in range(1, 101):
    model.fit(X_train[:num_iterations], y_train[:num_iterations])
    training_errors.append(mean_squared_error(model.predict(X_train), y_train))
    testing_errors.append(mean_squared_error(model.predict(X_test), y_test))

# Plot the learning curves
plt.plot(range(1, 101), training_errors, label='Training Error')
plt.plot(range(1, 101), testing_errors, label='Testing Error')
plt.xlabel('Number of training iterations')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()


# In[109]:


#get exp to predicted SalePrice to invers the (log)
pred_prise= np.exp(Bag_pred_data)


print(pred_prise[:10])


# <a id="t5."></a>
# # DecisionTreeRegressor

# In[110]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# Create the estimator
estimator = DecisionTreeRegressor()

# Fit the estimator
estimator.fit(X, y)

# Define a custom importance_getter function
def my_importance_getter(estimator):
    # Compute feature importances using permutation importance
    importances = permutation_importance(estimator, X, y, n_repeats=10, random_state=0)

    # Extract permutation importance scores
    scores = importances['importances_mean']

    # Convert scores to a numerical array
    scores = np.asarray(scores)

    # Apply absolute value function
    scores = np.abs(scores)

    return scores

# Create the feature selector
selector_tree = SelectFromModel(estimator=estimator, importance_getter=my_importance_getter, max_features=None)

# Transform the data
#X = selector.fit_transform(X, y)

# Show the X dimension and selected features
print('X Shape is:', X.shape)

# Get the selected features
selected_features = selector_tree.get_support()

# Print the selected feature names
selected_feature_names = X.columns[selected_features]
print("Selected Feature Names:")
for name in selected_feature_names:

    print(name)


# In[111]:


# selct cols in train data

tree_X = df_train[ ['Neighborhood','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                  'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', 'CentralAir','1stFlrSF','GrLivArea'
                   ,'GarageFinish','GarageArea']]



tree_X.head()


# In[112]:


# selct cols in test data

tree_test=df_test[ ['Neighborhood','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                  'YearRemodAdd','BsmtFinSF1', 'TotalBsmtSF', 'CentralAir','1stFlrSF','GrLivArea'
                   ,'GarageFinish','GarageArea']]


tree_test.head()


# In[113]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error 
#----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(tree_X, y_df_train, test_size=0.2, random_state=0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('scalar', StandardScaler()),
    ('model', DecisionTreeRegressor())#=,,
]                    

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))


#Calculating Details
print('Pipeline Model Train Score is : ' , pipeline.score(X_train, y_train))
print('Pipeline Model Test Score is : ' , pipeline.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = pipeline.predict(X_test)
print('Predicted Value for Pipeline Model is : ' , y_pred[:10])


#----------------------------------------------------
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)

#----------------------------------------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
#----------------------------------------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )

 
print('----------------------------------------------------------------------------------')
tree_pred_data = pipeline.predict(tree_test)
print('Predicted Value fpr Test data : ' , tree_pred_data[:10])


model = DecisionTreeRegressor()
# Train the model with different numbers of training iterations
training_errors = []
testing_errors = []
for num_iterations in range(1, 101):
    model.fit(X_train[:num_iterations], y_train[:num_iterations])
    training_errors.append(mean_squared_error(model.predict(X_train), y_train))
    testing_errors.append(mean_squared_error(model.predict(X_test), y_test))

# Plot the learning curves
plt.plot(range(1, 101), training_errors, label='Training Error')
plt.plot(range(1, 101), testing_errors, label='Testing Error')
plt.xlabel('Number of training iterations')
plt.ylabel('Mean squared error')
plt.legend()
plt.show()


# ### we see her over fit because the trainig error line under the testing error in many plesces

# In[114]:


#get exp to predicted SalePrice 
pred_prise= np.exp(tree_pred_data)


print(pred_prise[:10])


# # --------------------------------------------------------------------------------------------------------------

# # <b>References</b>
# * [kaggle](https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python#Want-to-learn-more?)
# * [scikit-learn.org]( https://scikit-learn.org/stable/modules/impute.html)
# * [Hair et al., 2013, Multivariate Data Analysis, 7th Edition](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)

# In[ ]:




