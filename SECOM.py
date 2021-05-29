#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install -c conda-forge pyreadstat


# In[2]:


conda install -c conda-forge pyreadstat


# In[3]:


import pandas as pd
import pyreadstat
import numpy as np
import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import plotly.express as px


# ## 1. Read Data

# In[5]:


df = pd.read_spss(r'C:\Users\s0576758\Desktop\HTW\Data Mining - Tilo\secom_mod.SAV')

df.set_index('ID',inplace=True)
df


# ## 2. Explore data

# #### 2.1. Shape and type

# In[6]:


df.shape


# In[7]:


df.info()


# #### 2.2. Pass/ Fail ratio

# In[8]:


df['class'].value_counts()


# In[9]:


labels = '0', '1'
sizes = df['class'].value_counts()
colors = ['lightblue', 'yellowgreen']

# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# #### 2.3. Histogram of features

# In[10]:


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

plotPerColumnDistribution(df, 10, 5)


# #### 2.4. Correlation

# In[11]:


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %0.9f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
plotScatterMatrix(df, 25, 10)


# In[12]:


# Correlation Histogram


cor_matrix = pd.DataFrame(df.corr())

# Heatmap of correlation
#cor_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)


# Remove reuntant values by selecting only the upper correlation triangle
upper_tri = pd.DataFrame(cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool)))

# Convert the upper correlation triangle to a Series
corlist = pd.Series(upper_tri.values.ravel('F'))

# Correlation Histogram
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':100})

# Plot Histogram on x
plt.hist(corlist, bins =50)
plt.gca().set(title='Frequency Histogram of Correlation', ylabel='Frequency', xlabel='Correlation coefficient');


# #### 2.5. Missing values

# In[13]:


#Total numbe rof Nan values in the dataframe
df.isna().sum().sum()


# In[14]:


#number of cols with Nan values
nan_cols = [i for i in df.columns if df[i].isnull().any()]
print('Nan_cols = ', len(nan_cols))

#number of rows with Nan values
nan_rows = df.apply(lambda x: sum(x.isnull().values), axis = 1) # For rows
print('Nan_rows = ', len(nan_rows))


# ## 3. Preprocessing

# In[15]:


# Change class labels
df.rename(columns ={'class':'Pass/Fail'},inplace=True)


# In[16]:


df


# In[17]:


# Give text labels to the training examples
df['Pass/Fail'] = df['Pass/Fail'].replace({0: "PASS", 1: "FAIL"})
df.tail()


# ## 4. Data Preparation

# ### 4.1.Splitting Traninig and Test Data

# In[18]:


# Split df into X and y
y = df['Pass/Fail']
X = df.drop('Pass/Fail', axis=1)


# In[19]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=40, stratify=y)


# In[20]:


#to check if the test and train data also has the same pass/fail ratio

fig = px.pie(
    y_train.value_counts(),
    values='Pass/Fail',
    names=["PASS", "FAIL"],
    title="Class Distribution of Train data",
    width=500
)

fig.show()



import plotly.express as px
fig = px.pie(
    y_test.value_counts(),
    values='Pass/Fail',
    names=["PASS", "FAIL"],
    title="Class Distribution of Test data",
    width=500
)

fig.show()


# Keeping test data aside. Using only X_train and y_train
# 

# ### 4.2 Reducing dimensionality of data by feature removal

# #### 4.2.1 Removal of timestamp and constants

# In[21]:


# removing timestamp
X_train= X_train.drop(columns='timestamp')


# In[22]:


#coloumns that have the same value excluding nan
print(len(X_train.columns[X_train.nunique() == 1]))

#column names with same value in it
X_train.columns[X_train.nunique() == 1]
print(list(X_train.columns[X_train.nunique() == 1]))

#remove these columns since it adds no value
single_value_coloumns = list(X_train.columns[X_train.nunique() == 1])
X_train=X_train.drop(columns=single_value_coloumns)
X_train


# #### 4.2.3. Checking for columns more than 55% nan values

# In[23]:


# count the number of missing values for each column
num_missing = (X_train == 0).sum()
# report the results
print(num_missing)


# In[24]:


#Missing Value histogram
nan_perc = pd.DataFrame(round((X_train.isnull().sum() * 100/ len(X_train)),2))
nan_perc = nan_perc.reset_index()
nan_perc = nan_perc.rename({'index': 'feature', 0: 'percentage'}, axis=1)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':100})
n,bins,patch = plt.hist(nan_perc['percentage'],bins=10, alpha=0.8, label='Value', edgecolor='black', linewidth=1)
plt.gca().set(title='Percentage of Missing Value Histogram', ylabel='No. of features', xlabel='Percentage of Nan values');
plt.show()


# In[25]:


# Defining a threshold to remove the values above it
threshold= 0.55
missing_value_columns = X_train.columns[X_train.isna().mean() >= threshold]
print(len(missing_value_columns))

X_train = X_train.drop(missing_value_columns, axis=1)
X_train


# #### 4.2.5. Less volatile

# In[26]:


#exploring  emaining features
summary = X_train.iloc[:,:].describe(include='all')
print(summary)


# In[27]:


#checking std of remaining features
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':100})

plt.hist(X_train.std(), bins = 50)
plt.gca().set(title='Standard deviation Histogram', ylabel='No. of features', xlabel='Standard deviation');
plt.show()


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,8), 'figure.dpi':100})

plt.hist(X_train.std()/X_train.mean(), bins = 50)
plt.gca().set(title='Coefficient of variation Histogram', ylabel='No. of features', xlabel='Coefficient of variation');
plt.show()


# In[29]:


# Checking for coeff of variation
feature_std = pd.DataFrame(summary.loc['std',:])
feature_coeffstd = pd.DataFrame(summary.loc['std',:]/summary.loc['mean',:])

#Max- min of stdev and coeff of variation
print("Min-max of stdev")
print('Min of stdev = ',min(feature_std.iloc[:,0]))
print('Max of stdev = ',max(feature_std.iloc[:,0]))

print ("\n")
print("Min-max of coeff of variation")
print('Min of coeff of variation = ', min(feature_coeffstd.iloc[:,0]))
print('Max of coeff of variation = ', max(feature_coeffstd.iloc[:,0]))


# ### 4.3 Outlier Identification

# In[30]:


# calculating the z score of the values
from scipy import stats
import stat
z_x_train= pd.DataFrame(stats.zscore(X_train,nan_policy='omit'))

# calculating the number of outliers
sum(z_x_train.apply(lambda x: sum(x.apply(lambda x: 1 if abs(x)>3 else 0))))


# In[31]:


#NA values in the dataset
X_train.isna().sum().sum()


# In[32]:


X_train = X_train.mask(X_train.sub(X_train.mean()).div(X_train.std()).abs().gt(3), np.nan)


# In[33]:


X_train.loc[:][:]


# In[34]:


X_train.isna().sum().sum()


# ### 4.4 Missing value Imputation

# #### KNN

# In[35]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_filled_knn = pd.DataFrame(imputer.fit_transform(X_train))


# In[36]:


X_filled_knn.isna().sum().sum()


# #### MICE

# In[37]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
X_filled_MICE = pd.DataFrame(imp.fit_transform(X_train))


# In[38]:


X_filled_MICE.isna().sum().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


# calculating the number of outliers
#Z TRANSFORM
scaler = StandardScaler()
scaler.fit(X_train)
a = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)


# In[40]:


demo = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)


# In[41]:


min(list(demo.std()))


# In[42]:


max(list(demo.std()))


# In[43]:


# MIN MAX SCALE
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = pd.DataFrame(min_max_scaler.fit_transform(X_train))
x_scaled


# In[44]:


min(list(x_scaled.std()))


# In[45]:


max(list(x_scaled.std()))


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





# In[ ]:




