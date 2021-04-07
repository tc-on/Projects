#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip


# # Importing Libraries

# In[ ]:


import sys ## used for functions that interact strongly with the interpreter (i.e. a program that reads and execute codes)
import copy ## to use copying functions
import pandas as pd ## library for dataset manipulation
import numpy as np ## mathematical function on arrays and matrices
import matplotlib.pyplot as plt ## for basic plotting
import seaborn as sns ## easier plotting and updated matplotlib library 
from sklearn.preprocessing import StandardScaler,PowerTransformer ## for standardization on numerical variables
from pandas_profiling import ProfileReport ## for a complete pandas DataFrame report
import plotly.express as px ## 'plotly.express' for scatter plotting on location data with a location map 
import statsmodels ## statsmodel module for using statistical models
import statsmodels.api as sm ## api module for OLS model from statsmodels, assigned as sm
import statsmodels.stats.api as sms ## api module for Gold-Feld Quandt Test from statsmodels.stats, assigned as sms
from sklearn.model_selection import train_test_split ## for splitting the dataset into train and test
from sklearn.metrics import mean_squared_error,r2_score ## for mse and r-squared
from sklearn.neighbors import KNeighborsRegressor ## for KNN-Regressor
from sklearn.tree import DecisionTreeRegressor ## for Decision Tree Regressor
import scipy.stats as stats ## for probability distribution and statistical functions
from scipy.stats import f_oneway,jarque_bera,shapiro ## for assumptions and statistical significance
from scipy.stats import norm,randint,skewnorm ## for normal continuous random variables and randomized integer variables
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif ## for variance inflation factor
from statsmodels.graphics.gofplots import qqplot ## for plotting q-q plots
import statsmodels.tsa.api as smt ## to use model classes and functions for time series analysis
from sklearn.model_selection import GridSearchCV ## for random cross-validation
from warnings import filterwarnings ## to remove warnings
filterwarnings('ignore') ## assigning warnings to 'ignore'
pd.set_option('display.max_columns', None) ## to show all columns
np.set_printoptions(threshold=sys.maxsize) ## to show the complete list or arrays
## will lead to static images of your plot embedded in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Dataset

# In[ ]:


home = pd.read_csv('homeperf.csv') ## reading the '.csv' file 
df = copy.deepcopy(home) ## copying to workstation dataset
home ## showing original DataFrame


# # Exploratory Data Analysis

# ## Data Preprocessing (Part-1)

# ### Features of the Original Dataset

# In[ ]:


## printing shape of the original DataFrame
print(home.shape)


# In[ ]:


## descriptive statistics of the original DataFrame
home.describe()


# In[ ]:


## printing information of the original DataFrame
print(home.info())


# In[ ]:


cat_cols_home = home.select_dtypes(include='object').columns ## selecting categorical columns into a list
num_cols_home = home.select_dtypes(include=np.number).columns ## selecting numeric columns into a list
print('Categorical:',cat_cols_home) ## printing categorical column names
print('\n') ## for spacing
print('Numerical:',num_cols_home) ## printing numerical column names


# ### Null Value Treatment

# In[ ]:


df.isna().sum()


# In[ ]:


## using for and if loops to gather null values in percentages for null value count greater than 0
for i in df:
  if ((df[i].isna().sum()/len(df[i]))*100)>0:
    print('Percentage of null values in',i,'is',round(((df[i].isna().sum()/len(df[i]))*100),3),'%')


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.isna())
plt.show


# - More than 10% null values in 'Gas Utility' and 'Year Home Built'. Hence, we drop both the columns.
# - Only 0.004% or 1 null value in 'Number of Units'. Hence, we drop the null value row.

# In[ ]:


df.drop(['Gas Utility','Year Home Built'],axis=1,inplace=True) ## dropping the extreme null value columns
df.dropna(inplace=True) ## dkropping the null value row
print(df.shape) ## printing the shape of dataframe


# In[ ]:


## checking for null values again after treatment
for i in df:
  if ((df[i].isna().sum()/len(df[i]))*100)>0:
    print('Percentage of null values in',i,'is',round(((df[i].isna().sum()/len(df[i]))*100),3),'%')
  else:
    print('No null values in the dataset.')


# - Checking for null values shows that null values have been treated.

# ### Fixing Duplicated Values

# In[ ]:


## check for duplicated rows 
df[df.duplicated()]


# - No duplicated values.
# - If duplicated values made after feature engineering, **ignore**

# ### Feature dType correction

# In[ ]:


## printing dataframe info to check wrongly specified feature dtype
print(df.info())


# - 'Size of Home' is wrongly specified to be 'object' type.
# - Need to be changed to 'numeric' type.

# #### Fixing 'Size of Home'

# In[ ]:


## printing unique values to understand values to fix
print(df['Size Of Home'].unique())


# In[ ]:


df['Size Of Home'] = df['Size Of Home'].str.replace(',', '') ## replacing ',' with empty field
df['Size Of Home'] = df['Size Of Home'].str.strip('Report SF less than ') ## stripping string value 
df['Size Of Home'] = df['Size Of Home'].str.strip('more than ') ## again stripping string value
df['Size Of Home'] = df['Size Of Home'].astype(np.number) ## changing the dtype by using astype


# In[ ]:


## printing DataFrame information to see if the correction has been made successfully
print(df.info())


# ### Feature Engineering

# #### Extracting only year from 'Project Completion Date'

# In[ ]:


## printing unique values to understand what values to strip
print(df['Project Completion Date'].unique())


# In[ ]:


df['Project Completion Year'] = df['Project Completion Date'] ## creating a new variable based on 'Project Completion Year' from  'Project Completion Date'
df['Project Completion Year'] = df['Project Completion Year'].str[6:] ## stripping values till we get only the year
print(df['Project Completion Year'].unique()) ## checking unique values again to see if stripping is successful


# #### Extracting only Month from 'Project Completion Date'

# In[ ]:


df['Billing Month'] = df['Project Completion Date'] ## creating a new variable based on the billing cycle from  'Project Completion Date'
df['Billing Month'] = df['Billing Month'].str[:2] ## stripping values till we get only the month
df['Billing Month'].replace({'01':'January','02':'February','03':'March',
                                 '04':'April','05':'May','06':'June','07':'July',
                                 '08':'August','09':'September','10':'October',
                                 '11':'November','12':'December'},inplace=True) ## Assigning month names
print(df['Billing Month'].unique()) ## checking unique values again to see if stripping is successful


# #### Extracting 'Latitude' and 'Longitude' from 'Location 1'

# In[ ]:


df['Location 1'] = df['Location 1'].str.split('(').str[1] ## stripping till '(' caharacter
df['Location 1'] = df['Location 1'].str.split(')').str[0] ## stripping the ')' character
df[['Latitude', 'Longitude']] = df['Location 1'].str.split(',', 1, expand=True) ## dividing 'Location 1' to 'Latitude' and 'Longitude' columns
df['Latitude'] = df['Latitude'].astype(np.number) ## fixing dType of 'Latitude' variable
df['Longitude'] = df['Longitude'].astype(np.number) ## fixing dType of 'Longitude' variable
df.drop('Location 1',axis=1,inplace=True) ## removing the unnecessary 'Location 1' variable


# #### Creating new feature from 'Project County'

# In[ ]:


## Checking unique county names
print(df['Project County'].unique())


# In[ ]:


## Creating new column 'Region' from 'Project County' and replacing unique counties with region names
df['Region'] = df['Project County']

## Central Region
df['Region'] = df['Region'].replace(['Oneida','Onondaga','Tioga','Cortland',
                                             'Chenango','Broome','Seneca',
                                             'Chemung','Madison','Otsego','Cayuga',
                                             'Tompkins','Schuyler'],'Central')
## East Region
df['Region'] = df['Region'].replace(['Albany','Rensselaer','Delaware','Columbia','Ulster',
                                             'Dutchess','Washington','Sullivan','Fulton','Schenectady',
                                             'Saratoga','Greene','Montgomery','Schoharie'],'East')
## West Region
df['Region'] = df['Region'].replace(['Niagara','Orleans','Steuben','Wyoming','Livingston',
                                             'Wayne','Chautauqua','Erie','Monroe','Allegany',
                                             'Genesee','Cattaraugus','Ontario','Yates'],'West')
## North Region
df['Region'] = df['Region'].replace(['Essex','Lewis','Oswego','Herkimer','Jefferson',
                                             'St. Lawrence','Clinton','Franklin','Warren',
                                             'Hamilton'],'North')
## South Region
df['Region'] = df['Region'].replace(['Suffolk','Kings','Bronx','New York','Nassau',
                                             'Queens','Orange','Putnam','Rockland','Westchester',
                                             'Richmond'],'South')


# #### New Variable based 'Type of Home Size' on 'Size of Home'

# In[ ]:


## checking descriptive statistics for size of home
df['Size Of Home'].describe()


# In[ ]:


## making a new variable 'Size of Residence' from 'Size of Home'
df['Type of Home Size'] = df['Size Of Home']
## label encoding 'Size of Residence' based on 'Large', 'Medium' and 'Small'
df['Type of Home Size'] = ['Large' if x>1803 
                           else 'Medium' if 1803>=x>1064 
                           else 'Small' 
                           for x in df['Type of Home Size']]


# In[ ]:


## checking value counts of Type of Home Size
df['Type of Home Size'].value_counts()


# #### Removing Unnecessary Features 

# In[ ]:


## 'Reporting Period', 'Project ID' and 'Project ZIP' are not needed for further analysis
df.drop(['Reporting Period','Project ID','Project ZIP'],axis=1,inplace=True)


# #### Cleaned Dataset

# In[ ]:


## showing cleaned DataFrame
df


# In[ ]:


## printing information of cleaned DataFrame
print(df.info())


# In[ ]:


df1 = copy.deepcopy(df)


# ## Data Preprocessing (Part-2)

# In[ ]:


cat_cols = df.select_dtypes(include='object').columns ## selecting categorical columns into a list
cat_cols1 = df1.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=np.number).columns ## selecting numeric columns into a list
num_cols1 = df1.select_dtypes(include=np.number).columns

cat_dat = df.select_dtypes(include='object') ## selecting categorical columns into a dataset
cat_dat1 = df1.select_dtypes(include='object')
num_dat = df.select_dtypes(include=np.number) ## selecting numeric columns into a dataset
num_dat1 = df1.select_dtypes(include=np.number)

print('Categorical:',cat_cols) ## printing categorical column names
print('\n') ## for spacing
print('Numerical:',num_cols) ## printing numerical column names


# ### Outlier Treatment

# #### Case 1: Outliers not removed

# In[ ]:


## checking for outliers and extreme values
for i in df.select_dtypes(np.number):
    plt.figure(figsize=(10,7))
    sns.boxplot(df[i])
    plt.show()


# - Outliers shouldn't be removed as the dataset will introduce bias towards projects.
# - We will further transform it to make the distributions more Gaussian-like.

# In[ ]:


# calculate interquartile range 

# compute the first quartile using quantile(0.25)
# use .drop() to drop the target variable 
# axis=1: specifies that the labels are dropped from the columns
Q1 = num_dat.drop(['First Year Modeled Project Energy Savings $ Estimate',
                   'Estimated Annual MMBtu Savings','Estimated Annual kWh Savings',
                   'Number Of Units','Longitude','Latitude'], axis=1).quantile(0.25)

# compute the first quartile using quantile(0.75)
# use .drop() to drop the target variable 
# axis=1: specifies that the labels are dropped from the columns
Q3 = num_dat.drop(['First Year Modeled Project Energy Savings $ Estimate',
                   'Estimated Annual MMBtu Savings','Estimated Annual kWh Savings',
                   'Number Of Units','Longitude','Latitude'], axis=1).quantile(0.75)

# calculate of interquartile range 
IQR = Q3 - Q1

# filter out the outlier values
# ~ : selects all rows which do not satisfy the condition
# |: bitwise operator OR in python
# any() : returns whether any element is True over the columns
# axis : "1" indicates columns should be altered (use "0" for 'index')
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


for i in df.select_dtypes(np.number):
    plt.figure(figsize=(10,7))
    sns.boxplot(df[i])
    plt.show()


# In[ ]:


df.shape


# #### Case 2: Outliers Removed

# In[ ]:


# calculate interquartile range 

# compute the first quartile using quantile(0.25)
# use .drop() to drop the target variable 
# axis=1: specifies that the labels are dropped from the columns
Q11 = num_dat1.drop(['Longitude','Latitude'], axis=1).quantile(0.25)

# compute the first quartile using quantile(0.75)
# use .drop() to drop the target variable 
# axis=1: specifies that the labels are dropped from the columns
Q31 = num_dat1.drop(['Longitude','Latitude'], axis=1).quantile(0.75)

# calculate of interquartile range 
IQR1 = Q31 - Q11

# filter out the outlier values
# ~ : selects all rows which do not satisfy the condition
# |: bitwise operator OR in python
# any() : returns whether any element is True over the columns
# axis : "1" indicates columns should be altered (use "0" for 'index')
df1 = df1[~((df1 < (Q1 - 1.5 * IQR1)) | (df1 > (Q31 + 1.5 * IQR1))).any(axis=1)]


# In[ ]:


for i in df1.select_dtypes(np.number):
    plt.figure(figsize=(10,7))
    sns.boxplot(df1[i])
    plt.show()


# ### Transformation of Numerical Variables

# #### Sqrt Transformation

# In[ ]:


df['sqr$'] = np.sqrt(df['First Year Modeled Project Energy Savings $ Estimate'])
df1['sqr$1'] = np.sqrt(df1['First Year Modeled Project Energy Savings $ Estimate'])

df.isnull().sum()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))
sns.distplot(df['sqr$'],ax=ax1)
sns.distplot(df1['sqr$1'],ax=ax2)
plt.tight_layout()
plt.show()


# #### Standard Scaling

# In[ ]:


ss = StandardScaler() ## assigning StandardScaler

num_t = ss.fit_transform(df.select_dtypes(np.number)) ## transforming numerical dataset with StandardScaler
num_t_dat = pd.DataFrame(data=num_t,columns=df.select_dtypes(np.number).columns) ## creating new encoded DataFrame

num_t1 = ss.fit_transform(df1.select_dtypes(np.number)) ## transforming numerical dataset with StandardScaler
num_t_dat1 = pd.DataFrame(data=num_t1,columns=df1.select_dtypes(np.number).columns) ## creating new encoded DataFrame


# In[ ]:


## checking transformed numerical dataset information
print(num_t_dat.info())


# ### Encoding Categorical Variables

# In[ ]:


## categorical dataset information to check for significant variables
print(cat_dat.info())


# In[ ]:


## removing unnecesary variables
cat_dat_dum = cat_dat.drop(['Project Completion Date','Project City','Project County'],axis=1)
num_t_dat = num_t_dat.drop(['Latitude','Longitude'],axis=1)

cat_dat_dum1 = cat_dat.drop(['Project Completion Date','Project City','Project County'],axis=1)
num_t_dat1 = num_t_dat1.drop(['Latitude','Longitude'],axis=1)


# In[ ]:


## creating dummy variables for categorical dataset
cat_dat_dumm = pd.get_dummies(cat_dat_dum,drop_first=True) ## one-hot encoding categorical dataset
cat_dat_dumm1 = pd.get_dummies(cat_dat_dum1,drop_first=True) ## one-hot encoding categorical dataset

print(cat_dat_dumm.shape) ## checking encoded categorical dataset shape
print(cat_dat_dumm1.shape) ## checking encoded categorical dataset shape


# In[ ]:


## creating new encoded dataset which will be used for analysis
df_en = cat_dat_dumm.join(num_t_dat,how='inner') ## using an inner-join function for creating a fully encoded dataset
df_en1 = cat_dat_dumm1.join(num_t_dat1,how='inner') ## using an inner-join function for creating a fully encoded dataset

df_noen = cat_dat_dumm.join(df.select_dtypes(np.number),how='inner')
df_noen1 = cat_dat_dumm1.join(df1.select_dtypes(np.number),how='inner')
df_noen = df_noen.drop(['Longitude','Latitude'],axis=1)
df_noen1 = df_noen1.drop(['Longitude','Latitude'],axis=1)

print(df_en.shape) ## checking final encoded dataset shape
print(df_en1.shape) ## checking final encoded dataset shape
print('\n')
print(df_noen.shape) ## checking final encoded dataset shape
print(df_noen1.shape) ## checking final encoded dataset shape


# ### Encoded Dataset

# In[ ]:


## Showing encoded DataFrame with outliers
df_en.info()


# In[ ]:


## Showing unencoded dataframe with outliers
df_noen.info()


# ## Pandas Profiling

# In[ ]:


profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_notebook_iframe()


# In[ ]:





# ## Plots
# 
# **Note:-**
# - Using 'df' or 'Cleaned' DataFrame for plotting exploration

# ### Correlation Heat Map

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# ### Pair Plot

# In[ ]:


plt.figure(figsize=(50,50))
pp = sns.pairplot(df,hue='Type of Home Size',height=10)
pp.add_legend()
plt.show()


# ### Location Scatter Plots

# In[ ]:


## plot a scatter plot on the world map
fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name =df["Project City"],
                        hover_data=df[["Project County","Electric Utility","Size Of Home"]],
                        color_discrete_sequence=["fuchsia"], zoom=5.7, height=600)

## set the layout of the map
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {   "below": 'traces',
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]
        }
      ])
## set magin width to 0 to avoid margins
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.scatterplot(df['Longitude'],df['Latitude'],hue = df['Region'],style = df['Pre-Retrofit Home Heating Fuel Type'])
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.show()


# ### Bar Plots

# In[ ]:


## defining function to plot against imortant features vs the target feature

def targetbar(f, ax1, ax2, ax3, ax4,x1,x2,x3,x4,dat):
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,15))
    
    ax1.bar(x=x1,height=x2,data=dat)
    ax1.set_title(label = x2+' '+'by'+' '+x1)
    ax1.set_xlabel(x1)
    ax1.set_ylabel(x2)
    ax1.tick_params(axis='x',labelrotation=45)
    ax1.grid()
    
    sns.countplot(x=x1,data=dat,ax=ax2)
    ax2.set_title('Count of'+' '+x1)
    ax2.tick_params(axis='x',labelrotation=45)
    ax2.grid()
    
    ax3.bar(x=x1,height=x3,data=dat)
    ax3.set_title(x3+' '+'by'+' '+x1)
    ax3.set_xlabel(x1)
    ax3.set_ylabel(x3)
    ax3.tick_params(axis='x',labelrotation=45)
    ax3.grid()
    
    ax4.bar(x=x1,height=x4,data=dat)
    ax4.set_title(x4+' '+'by'+' '+x1)
    ax4.set_xlabel(x1)
    ax4.set_ylabel(x4)
    ax4.tick_params(axis='x',labelrotation=45)
    ax4.grid()
    
    plt.tight_layout()
    plt.show()
    return f, ax1, ax2, ax3, ax4,x1,x2,x3,x4,dat


# In[ ]:


x1='Type Of Dwelling'
x2='First Year Modeled Project Energy Savings $ Estimate'
x3='Estimated Annual MMBtu Savings'
x4='Estimated Annual kWh Savings'
x5='Total Project Cost'
dat=df

f, ax = plt.subplots(3, 2,figsize=(20,25))
    
ax[0][0].bar(x=x1,height=x2,data=dat)
ax[0][0].set_title(label = x2+' '+'by'+' '+x1)
ax[0][0].set_xlabel(x1)
ax[0][0].set_ylabel(x2)
ax[0][0].tick_params(axis='x',labelrotation=45)
ax[0][0].grid()
    
sns.countplot(x=x1,data=dat,ax=ax[0][1])
ax[0][1].set_title('Count of'+' '+x1)
ax[0][1].tick_params(axis='x',labelrotation=45)
ax[0][1].grid()
    
ax[1][0].bar(x=x1,height=x3,data=dat)
ax[1][0].set_title(x3+' '+'by'+' '+x1)
ax[1][0].set_xlabel(x1)
ax[1][0].set_ylabel(x3)
ax[1][0].tick_params(axis='x',labelrotation=45)
ax[1][0].grid()
    
ax[1][1].bar(x=x1,height=x4,data=dat)
ax[1][1].set_title(x4+' '+'by'+' '+x1)
ax[1][1].set_xlabel(x1)
ax[1][1].set_ylabel(x4)
ax[1][1].tick_params(axis='x',labelrotation=45)
ax[1][1].grid()

ax[2][0].bar(x=x1,height=x5,data=dat)
ax[2][0].set_title(x5+' '+'by'+' '+x1)
ax[2][0].set_xlabel(x1)
ax[2][0].set_ylabel(x5)
ax[2][0].tick_params(axis='x',labelrotation=45)
ax[2][0].grid()
    
plt.tight_layout()
plt.show()


# In[ ]:


x1='Job Type'
x2='First Year Modeled Project Energy Savings $ Estimate'
x3='Estimated Annual MMBtu Savings'
x4='Estimated Annual kWh Savings'
x5='Total Project Cost'
dat=df

f, ax = plt.subplots(3, 2,figsize=(20,25))
    
ax[0][0].bar(x=x1,height=x2,data=dat)
ax[0][0].set_title(label = x2+' '+'by'+' '+x1)
ax[0][0].set_xlabel(x1)
ax[0][0].set_ylabel(x2)
ax[0][0].tick_params(axis='x',labelrotation=45)
ax[0][0].grid()
    
sns.countplot(x=x1,data=dat,ax=ax[0][1])
ax[0][1].set_title('Count of'+' '+x1)
ax[0][1].tick_params(axis='x',labelrotation=45)
ax[0][1].grid()
    
ax[1][0].bar(x=x1,height=x3,data=dat)
ax[1][0].set_title(x3+' '+'by'+' '+x1)
ax[1][0].set_xlabel(x1)
ax[1][0].set_ylabel(x3)
ax[1][0].tick_params(axis='x',labelrotation=45)
ax[1][0].grid()
    
ax[1][1].bar(x=x1,height=x4,data=dat)
ax[1][1].set_title(x4+' '+'by'+' '+x1)
ax[1][1].set_xlabel(x1)
ax[1][1].set_ylabel(x4)
ax[1][1].tick_params(axis='x',labelrotation=45)
ax[1][1].grid()

ax[2][0].bar(x=x1,height=x5,data=dat)
ax[2][0].set_title(x5+' '+'by'+' '+x1)
ax[2][0].set_xlabel(x1)
ax[2][0].set_ylabel(x5)
ax[2][0].tick_params(axis='x',labelrotation=45)
ax[2][0].grid()
    
plt.tight_layout()
plt.show()


# In[ ]:


x1='Pre-Retrofit Home Heating Fuel Type'
x2='First Year Modeled Project Energy Savings $ Estimate'
x3='Estimated Annual MMBtu Savings'
x4='Estimated Annual kWh Savings'
x5='Total Project Cost'
dat=df

f, ax = plt.subplots(3, 2,figsize=(20,25))
    
ax[0][0].bar(x=x1,height=x2,data=dat)
ax[0][0].set_title(label = x2+' '+'by'+' '+x1)
ax[0][0].set_xlabel(x1)
ax[0][0].set_ylabel(x2)
ax[0][0].tick_params(axis='x',labelrotation=45)
ax[0][0].grid()
    
sns.countplot(x=x1,data=dat,ax=ax[0][1])
ax[0][1].set_title('Count of'+' '+x1)
ax[0][1].tick_params(axis='x',labelrotation=45)
ax[0][1].grid()
    
ax[1][0].bar(x=x1,height=x3,data=dat)
ax[1][0].set_title(x3+' '+'by'+' '+x1)
ax[1][0].set_xlabel(x1)
ax[1][0].set_ylabel(x3)
ax[1][0].tick_params(axis='x',labelrotation=45)
ax[1][0].grid()
    
ax[1][1].bar(x=x1,height=x4,data=dat)
ax[1][1].set_title(x4+' '+'by'+' '+x1)
ax[1][1].set_xlabel(x1)
ax[1][1].set_ylabel(x4)
ax[1][1].tick_params(axis='x',labelrotation=45)
ax[1][1].grid()

ax[2][0].bar(x=x1,height=x5,data=dat)
ax[2][0].set_title(x5+' '+'by'+' '+x1)
ax[2][0].set_xlabel(x1)
ax[2][0].set_ylabel(x5)
ax[2][0].tick_params(axis='x',labelrotation=45)
ax[2][0].grid()
    
plt.tight_layout()
plt.show()


# In[ ]:


x1="Electric Utility"
x2='First Year Modeled Project Energy Savings $ Estimate'
x3='Estimated Annual MMBtu Savings'
x4='Estimated Annual kWh Savings'
x5='Total Project Cost'
dat=df

f, ax = plt.subplots(3, 2,figsize=(20,25))
    
ax[0][0].bar(x=x1,height=x2,data=dat)
ax[0][0].set_title(label = x2+' '+'by'+' '+x1)
ax[0][0].set_xlabel(x1)
ax[0][0].set_ylabel(x2)
ax[0][0].tick_params(axis='x',labelrotation=45)
ax[0][0].grid()
    
sns.countplot(x=x1,data=dat,ax=ax[0][1])
ax[0][1].set_title('Count of'+' '+x1)
ax[0][1].tick_params(axis='x',labelrotation=45)
ax[0][1].grid()
    
ax[1][0].bar(x=x1,height=x3,data=dat)
ax[1][0].set_title(x3+' '+'by'+' '+x1)
ax[1][0].set_xlabel(x1)
ax[1][0].set_ylabel(x3)
ax[1][0].tick_params(axis='x',labelrotation=45)
ax[1][0].grid()
    
ax[1][1].bar(x=x1,height=x4,data=dat)
ax[1][1].set_title(x4+' '+'by'+' '+x1)
ax[1][1].set_xlabel(x1)
ax[1][1].set_ylabel(x4)
ax[1][1].tick_params(axis='x',labelrotation=45)
ax[1][1].grid()

ax[2][0].bar(x=x1,height=x5,data=dat)
ax[2][0].set_title(x5+' '+'by'+' '+x1)
ax[2][0].set_xlabel(x1)
ax[2][0].set_ylabel(x5)
ax[2][0].tick_params(axis='x',labelrotation=45)
ax[2][0].grid()
    
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


print(df.shape)


# In[ ]:


dfToD = df[df['Type Of Dwelling']=='Single Family']
dfToD['Type Of Dwelling'].unique()
print(dfToD['Job Type'].unique())
print(dfToD.shape)
print('lossing',df.shape[0]-dfToD.shape[0],'values')


# In[ ]:


dfJT = dfToD[dfToD['Job Type']=='Home Performance']
print(dfJT['Job Type'].unique())
print(dfJT.shape)
print('lossing',df.shape[0]-dfJT.shape[0],'values')


# In[ ]:


dfPRF = dfJT[dfJT['Pre-Retrofit Home Heating Fuel Type']=='Natural Gas']
print(dfPRF['Pre-Retrofit Home Heating Fuel Type'].unique())
print(dfPRF.shape)
print('lossing',df.shape[0]-dfPRF.shape[0],'values')


# ##### Rejected:

# In[ ]:


dfEU = dfPRF[dfPRF["Electric Utility"]=='National Grid']
print(dfEU["Electric Utility"].unique())
print(dfEU.shape)
print('lossing',df.shape[0]-dfEU.shape[0],'values')


# In[ ]:


dfEUNG = dfJT[dfJT["Electric Utility"]=='National Grid']
print(dfEUNG["Electric Utility"].unique())
print(dfEUNG.shape)
print('lossing',df.shape[0]-dfEUNG.shape[0],'values')


# In[ ]:





# In[ ]:


dfPRF.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## bar plots and count plot with Respect to Year
x1='Project Completion Year'
x2='First Year Modeled Project Energy Savings $ Estimate'
x3='Estimated Annual MMBtu Savings'
x4='Estimated Annual kWh Savings'
dat=dfPRF

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,15))
    
ax1.bar(x=x1,height=x2,data=dat)
ax1.set_title(label = x2+' '+'by'+' '+x1)
ax1.set_xlabel(x1)
ax1.set_ylabel(x2)
ax1.tick_params(axis='x',labelrotation=45)
ax1.grid()
    
sns.countplot(x=x1,data=dat,ax=ax2)
ax2.set_title('Count of'+' '+x1)
ax2.tick_params(axis='x',labelrotation=45)
ax2.grid()
    
ax3.bar(x=x1,height=x3,data=dat)
ax3.set_title(x3+' '+'by'+' '+x1)
ax3.set_xlabel(x1)
ax3.set_ylabel(x3)
ax3.tick_params(axis='x',labelrotation=45)
ax3.grid()
    
ax4.bar(x=x1,height=x4,data=dat)
ax4.set_title(x4+' '+'by'+' '+x1)
ax4.set_xlabel(x1)
ax4.set_ylabel(x4)
ax4.tick_params(axis='x',labelrotation=45)
ax4.grid()
    
plt.tight_layout()
plt.show()


# In[ ]:


## bar plots and count plot with respect to Type of Home Size

targetbar (f,ax1,ax2,ax3,ax4,'Type of Home Size','First Year Modeled Project Energy Savings $ Estimate',
          'Estimated Annual MMBtu Savings','Estimated Annual kWh Savings',df)


# In[ ]:


## bar plots and count plot with respect to Region

targetbar (f,ax1,ax2,ax3,ax4,'Region','First Year Modeled Project Energy Savings $ Estimate',
          'Estimated Annual MMBtu Savings','Estimated Annual kWh Savings',df)


# In[ ]:


## bar plots and count plot with respect to Billing Month
targetbar(f,ax1,ax2,ax3,ax4,'Billing Month','First Year Modeled Project Energy Savings $ Estimate',
         'Estimated Annual MMBtu Savings','Estimated Annual kWh Savings',df)


# In[ ]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


s = df['Region']
counts = s.value_counts()
percent = s.value_counts(normalize=True)
percent100 = s.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
a = pd.DataFrame({'counts': counts, 'per': percent, 'per100': percent100})
b = pd.DataFrame (data = (a['per']*100),index=a.index,columns=['per'])

f,ax = plt.subplots(figsize=(10,6))
ax = sns.barplot(x=a.index,y=round((a['per']*100),2),data=df)
ax.set_title('Percentage of Home Participation by Region',fontsize=14)
ax.set_ylabel('Percentage %',fontsize=14)
ax.set_xlabel('Region',fontsize=14)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center',
                   color='white', 
                   size=15,
                   xytext = (0, -12), 
                   textcoords = 'offset points')
    
for p in ax.patches:
    ax.annotate(('%'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center',
                   color='white', 
                   size=15,
                   xytext = (23, -12), 
                   textcoords = 'offset points')

plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(df['Region'],hue = df['Electric Utility'])
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.show()


# ### Violin Plot

# In[ ]:


plt.figure(figsize=(15,5))
plt.bar(df['Electric Utility'],df['Estimated Annual kWh Savings'])
plt.xticks(rotation = 45)
plt.show()


# # Statistical Significance

# ## Descriptive Statistics

# In[ ]:


df.drop(['Latitude','Longitude'],axis=1).describe()


# ## Checking for Normality

# ### Distribution Plots

# In[ ]:


fig, ax = plt.subplots(3,2,figsize=(12,15))
sns.distplot(df['First Year Modeled Project Energy Savings $ Estimate'],hist=True,kde=True,bins=10,ax=ax[0,0])
qqplot(df['First Year Modeled Project Energy Savings $ Estimate'], line='s',ax=ax[0,1])
sns.distplot(df['Estimated Annual MMBtu Savings'],hist=True,kde=True,bins=10,ax=ax[1,0])
qqplot(df['Estimated Annual MMBtu Savings'], line='s',ax=ax[1,1])
sns.distplot(df['Estimated Annual kWh Savings'],hist=True,kde=True,bins=10,ax=ax[2,0])
qqplot(df['Estimated Annual kWh Savings'], line='s',ax=ax[2,1])
plt.tight_layout()
plt.show()


# - Heavily skewed towards right.
# - Not normally distributed

# In[ ]:


stat, p = shapiro(df["First Year Modeled Project Energy Savings $ Estimate"])

print(stat,p)


# ### Shapiro-Wilk Test

# **The null and alternate hypothesis of Shapiro test are as follows:**
# 
# H0: The data is normally distributed
# 
# H1: The data is not normally distributed

# In[ ]:


# normality test using shapiro()
# the test returns the the test statistics and the p-value of the test
stat, p = shapiro(df["First Year Modeled Project Energy Savings $ Estimate"])
stat1,p1 = shapiro(df["Estimated Annual MMBtu Savings"])
stat2,p2 = shapiro(df["Estimated Annual kWh Savings"])

# display the conclusion
# set the level of significance to 0.05
alpha = 0.05

# to print the numeric outputs of the Jarque-Bera test upto 3 decimal places
# %.3f: returns the a floating point with 3 decimal digit accuracy
# the '%' holds the place where the number is to be printed
# if the p-value is greater than alpha print we accept alpha 
# if the p-value is less than alpha print we reject alpha

print('Statistics=%.3f, P-Value=%.3f' % (stat, p))
if p > alpha:
    print('The data for First Year Modeled Project Energy Savings $ Estimate is normally distributed (fail to reject H0)')
else:
    print('The data for First Year Modeled Project Energy Savings $ Estimate is not normally distributed (reject H0)')
print('\n')
print('Statistics=%.3f, P-Value=%.3f' % (stat1, p1))
if p1 > alpha:
    print('The data for Estimated Annual MMBtu Savings is normally distributed (fail to reject H0)')
else:
    print('The data is for Estimated Annual MMBtu Savings not normally distributed (reject H0)')
print('\n')
print('Statistics=%.3f, P-Value=%.3f' % (stat2, p2))
if p2 > alpha:
    print('The data for Estimated Annual kWh Savings is normally distributed (fail to reject H0)')
else:
    print('The data for Estimated Annual kWh Savings is not normally distributed (reject H0)')


# # Initial Modelling

# In[ ]:


df_modelling=copy.deepcopy(df)
df_modelling.head()


# In[ ]:


# remove 'project county','project city','latitude','longitude','type of home size'

df_modelling.drop(['Project County','Project City','Latitude','Longitude','Type of Home Size'],axis=1,inplace=True)
df_modelling.head()


# In[ ]:


X = df_modelling.drop(['First Year Modeled Project Energy Savings $ Estimate'],axis=1)
y = df_modelling['First Year Modeled Project Energy Savings $ Estimate']

X.head()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[ ]:


# dropping 'project completion date' as we have 'project completion year' in place

X_train.drop(['Project Completion Date'],axis=1,inplace=True)
X_test.drop(['Project Completion Date'],axis=1,inplace=True)
X_train.head()


# In[ ]:


# reporting period for all the records is 2021.
# therefore, reporting_period-project_completion_year is the num_of_yrs_since_completion which makes more sense

X_train['num_yrs_since_proj_completion'] = X_train['Project Completion Year'].astype('int64').apply(lambda x: 2021-x)
X_train.drop(['Project Completion Year'],axis=1,inplace=True)

X_test['num_yrs_since_proj_completion'] = X_test['Project Completion Year'].astype('int64').apply(lambda x: 2021-x)
X_test.drop(['Project Completion Year'],axis=1,inplace=True)

print(X_train.head())
print(X_test.head())


# ## Feature Sets

# ### Feature Set 1: One Hot Encoded

# In[ ]:


import copy
X_train_ohe = copy.deepcopy(X_train)
X_test_ohe = copy.deepcopy(X_test)
print(X_train_ohe.shape)
print(X_test_ohe.shape)


# In[ ]:


cat_dat_tr = X_train_ohe.select_dtypes(include='object')
cat_dat_te = X_test_ohe.select_dtypes(include='object')

cat_dat_tr


# In[ ]:


train_objs_num = len(X_train)
dataset = pd.concat(objs=[cat_dat_tr,cat_dat_te], axis=0)

dataset = pd.get_dummies(dataset,drop_first=True)
X_tr_cat_dat = copy.deepcopy(dataset[:train_objs_num])
X_te_cat_dat = copy.deepcopy(dataset[train_objs_num:])

print(X_tr_cat_dat.shape)
print(X_te_cat_dat.shape)


# In[ ]:


##### encoding/scaling numerical data
X_tr_num_dat = X_train_ohe.select_dtypes(include=np.number)
X_te_num_dat = X_test_ohe.select_dtypes(include=np.number)


# In[ ]:


# scaling numerical data  using standard scaler

for i in X_tr_num_dat.columns:
    ss=StandardScaler()
    X_tr_num_dat[i] = ss.fit_transform(X_tr_num_dat[[i]])
    X_te_num_dat[i] = ss.transform(X_te_num_dat[[i]])
    
print(X_tr_num_dat.shape)
print(X_te_num_dat.shape)


# In[ ]:


X_train_ohe = X_tr_cat_dat.join(X_tr_num_dat,how='inner') 
X_test_ohe = X_te_cat_dat.join(X_te_num_dat,how='inner')

print(X_train_ohe.shape)
X_test_ohe.shape


# In[ ]:


print(y_train.shape)
print(y_test.shape)


# ### Feature Set 2: Frequency Encoded

# In[ ]:


X_train_freq = copy.deepcopy(X_train)
X_test_freq = copy.deepcopy(X_test)


# In[ ]:


cat_dat_tr = X_train_freq.select_dtypes(include='object')
cat_dat_te = X_test_freq.select_dtypes(include='object')

cat_dat_tr


# In[ ]:


c=copy.deepcopy(cat_dat_tr)
for i in cat_dat_tr.columns:
    d = dict(cat_dat_tr[i].value_counts(normalize=True))
    print(d)
    cat_dat_tr[i]=cat_dat_tr[i].apply(lambda x:d[x])

cat_dat_tr


# In[ ]:


def return_freq(d,x):
    try:
        return d[x]
    except:
        return 0
for i in cat_dat_te.columns:
    d = dict(c[i].value_counts(normalize=True))
    print(d)
    cat_dat_te[i]=cat_dat_te[i].apply(lambda x:return_freq(d,x))

cat_dat_te.head()


# In[ ]:


X_train_freq = cat_dat_tr.join(X_tr_num_dat,how='inner') 
X_test_freq = cat_dat_te.join(X_te_num_dat,how='inner')

print(X_train_freq.shape)
X_test_freq.shape


# ## Base Models

# ### OLS Base Model

# In[ ]:


X = df_en.drop('First Year Modeled Project Energy Savings $ Estimate',axis=1)
y = df_en['First Year Modeled Project Energy Savings $ Estimate']


# In[ ]:


Xc = sm.add_constant(X)

model = sm.OLS(y,Xc).fit()

# print the summary output
model.summary()


# #### Backward Elimination

# In[ ]:


cols = list(Xc.columns)
while len(cols)>1:
  X1 = Xc[cols]
  model1 = sm.OLS(y,X1).fit()
  pvalues = model1.pvalues
  pvalues = pvalues.drop('const')
  max_p = max(pvalues)
  feature_maxp = pvalues.idxmax()
  if max_p > 0.05:
    cols.remove(feature_maxp)
    print(feature_maxp, max_p)
  else:
    break

selected_features = cols
print(cols)


# #### Base OLS Model after Backward Elimination

# In[ ]:


Xc1 = Xc[selected_features]
model1 = sm.OLS(y,Xc1).fit()
model1.summary()


# In[ ]:


residuals = model1.resid
y_pred = model1.predict(Xc1)


# ### Testing for Assumptions for OLS Base Model

# #### Assumption 1: Multicollinearity

# In[ ]:


vf = [vif(Xc1.values,i) for i in range(Xc1.shape[1])]
vfdf = pd.DataFrame(vf, index= Xc1.columns, columns=['vif'])
vfdf


# #### Assumption 2: Normality of Residuals

# In[ ]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
stats.probplot(residuals,plot=plt)
plt.subplot(1,2,2)
#skewnorm.fit(residuals)
sns.distplot(residuals,fit=skewnorm,kde=True,hist=False,label='Skewed Normal')
#norm.fit(residuals)
#sns.distplot(residuals,fit=norm,kde=True,hist=False,color='red',label='Standard Normal')
#plt.legend()
plt.show()


# ##### Jarque Berra Test for checking goodness of fit

# In[ ]:


jarque_bera(residuals)


# ~~~
# H0: Residuals are normally distributed
# H1: Residuals are not normally distributed
# ~~~
# - Since P-Value(0.0) is less than significance level, we will reject H0 to conclude that residuals are not normally distributed.

# #### Assumption 3: Homoscedasticity

# In[ ]:


plt.figure(figsize=(8,5))
sns.regplot(x=y_pred,y=residuals, lowess=True, line_kws={'color':'red'})
plt.xlabel('y_pred')
plt.ylabel('residuals')
plt.show()


# ##### Goldfeld Quandt Test for checking Homoscedasticity

# In[ ]:


test = sms.het_goldfeldquandt(y=residuals,x=Xc1)
test


# ~~~
# H0: Variance of residuals is constant across the range of data
# H1: Variance of residuals is not constant across the range of  data
# ~~~
# - Since P-Value(2.0360055000948948e-07) is less than significance level, we will reject H0 to conclude that variance of residuals is not constant.

# #### Assumption 4: Auto-Correlation

# In[ ]:


fig, ax = plt.subplots(figsize=(8, 5))
acf = smt.graphics.plot_acf(residuals,lags=30,ax=ax)
acf.show()


# #### Assumption 5: Linearity of Relationship

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
sns.regplot(x=y_pred,y=y,lowess=True,line_kws={'color':'red'})
plt.subplot(2,1,2)
sns.regplot(x=y_pred,y=residuals, lowess=True, line_kws={'color':'red'})
plt.xlabel('y_pred')
plt.ylabel('residuals')
plt.show()


# In[ ]:


sm.stats.diagnostic.linear_rainbow(model1)


# ~~~
# H0: Fit of model using full sample = Fit of model using a central subset (linear relationship)
# H1: Fit of model using full sample is worse compared to fit of model using a central subset.
# ~~~
# - Since P-Value(7.385520188495804e-34) is lower than significance level, we will reject the H0 to conclude that Fit of model using full sample is worse compared to fit of model using a central subset. We need to improve our model.

# ### Model 1: K Nearest Neighbors Regression

# #### 1.1: KNN regression with One-Hot Encoded features

# In[ ]:


neigh = KNeighborsRegressor()
neigh.fit(X_train_ohe,y_train)

y_tr_pred = neigh.predict(X_train_ohe)
print('train rmse knn regression:',np.sqrt(mean_squared_error(y_train,y_tr_pred)))

y_te_pred = neigh.predict(X_test_ohe)
print('test rmse knn regression:',np.sqrt(mean_squared_error(y_test,y_te_pred)))


# In[ ]:


# finding best value of 'n' by manual tuning
train_mse=[]
test_mse=[]
for n in range(20):
  n = n+1
  model = KNeighborsRegressor(n_neighbors = n)

  model.fit(X_train_ohe, y_train)  #fit the model
  y_tr_pred=model.predict(X_train_ohe) #make prediction on test set
  tr_mse = mean_squared_error(y_train,y_tr_pred) #calculate rmse
  train_mse.append(tr_mse)

  y_te_pred=model.predict(X_test_ohe) #make prediction on test set
  te_mse = mean_squared_error(y_test,y_te_pred)
  test_mse.append(te_mse)
  print('test RMSE value for n =' , n , 'is:', np.sqrt(te_mse))


# In[ ]:


n = [i for i in range(1,21)]
plt.plot(n,test_mse,color='g',label='test mse')
plt.plot(n,train_mse,color='r',label='train mse')
plt.xticks([1,3,5,7,9,11,13,15,17,19])
plt.legend()
plt.show()


# - from the manual hyper parameter tuning, knn regressor with neighbours=7 gave the best mean squared error on test data

# In[ ]:


# Setup the parameters 
k_range = list(range(1, 21))
param_grid = dict(n_neighbors=k_range)


knn = KNeighborsRegressor()

# Instantiate the GridSearchCV object: knn_cv
knn_cv = GridSearchCV(knn,param_grid,scoring='r2',cv=5,n_jobs=-1)
knn_cv.fit(X_train_ohe,y_train)


print("Tuned Decision Tree Parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))


# In[ ]:


neigh = KNeighborsRegressor(**knn_cv.best_params_) # picking optimal k as 9 from random search
neigh.fit(X_train_ohe,y_train)

y_tr_pred = neigh.predict(X_train_ohe)
print('train rmse knn regression with best parameters:',np.sqrt(mean_squared_error(y_train,y_tr_pred)))

y_te_pred = neigh.predict(X_test_ohe)
print('test rmse knn regression with best parameters:',np.sqrt(mean_squared_error(y_test,y_te_pred)))


# #### 1.2: KNN regression with Frequency Encoded Feature Set

# In[ ]:


neigh = KNeighborsRegressor()
neigh.fit(X_train_freq,y_train)

y_tr_pred = neigh.predict(X_train_freq)
print('train rmse knn regression:',np.sqrt(mean_squared_error(y_train,y_tr_pred)))

y_te_pred = neigh.predict(X_test_freq)
print('test rmse knn regression:',np.sqrt(mean_squared_error(y_test,y_te_pred)))


# In[ ]:


train_mse=[]
test_mse=[]
for n in range(20):
  n = n+1
  model = KNeighborsRegressor(n_neighbors = n)

  model.fit(X_train_freq, y_train)  #fit the model
  y_tr_pred=model.predict(X_train_freq) #make prediction on test set
  tr_mse = mean_squared_error(y_train,y_tr_pred) #calculate rmse
  train_mse.append(tr_mse)

  y_te_pred=model.predict(X_test_freq) #make prediction on test set
  te_mse = mean_squared_error(y_test,y_te_pred)
  test_mse.append(te_mse)
  print('test RMSE value for n =' , n , 'is:', np.sqrt(te_mse))


# In[ ]:


n = [i for i in range(1,21)]
plt.plot(n,test_mse,color='g',label='test mse')
plt.plot(n,train_mse,color='r',label='train mse')
plt.xticks([1,3,5,7,9,11,13,15,17,19])
plt.legend()
plt.show()


# - n=10 looks optimal

# In[ ]:


# Setup the parameters 
k_range = list(range(1, 21))
param_grid = dict(n_neighbors=k_range)


knn = KNeighborsRegressor()

# Instantiate the GridSearchCV object: knn_cv
knn_cv = GridSearchCV(knn,param_grid,scoring='r2',cv=5,n_jobs=-1)
knn_cv.fit(X_train_freq,y_train)


print("Tuned Decision Tree Parameters: {}".format(knn_cv.best_params_))
print("Best score is {}".format(knn_cv.best_score_))


# In[ ]:


neigh = KNeighborsRegressor(**knn_cv.best_params_) # fitting with optimal 'n' picked
neigh.fit(X_train_freq,y_train)

y_tr_pred = neigh.predict(X_train_freq)
print('train rmse knn regression with best parameters:',np.sqrt(mean_squared_error(y_train,y_tr_pred)))

y_te_pred = neigh.predict(X_test_freq)
print('test rmse knn regression with best parameters:',np.sqrt(mean_squared_error(y_test,y_te_pred)))


# - From the  train and test root_mean_squared_error values of base model and knn regression, it is evident that knn regressor showed a significant improvement over the base model

# ### Model 2: Decision Tree Regression

# #### 2.1: DT regression with One-Hot Encoded Feature Set

# In[ ]:


dtreg = DecisionTreeRegressor(random_state=42) 
dtreg.fit(X_train_ohe,y_train) # fitting regressor with ohe encoded features

y_tr_pred = dtreg.predict(X_train_ohe)
tr_mse = mean_squared_error(y_train,y_tr_pred)
y_te_pred = dtreg.predict(X_test_ohe)
te_mse = mean_squared_error(y_test,y_te_pred)

print('train rmse decision tree regressor:',np.sqrt(tr_mse))
print('test rmse decision tree regressor:',np.sqrt(te_mse))


# In[ ]:


# Setup the parameters 
param_dist = {"max_depth": [3,4,5,6,7,8,9,10,11,15,19,20,None],
              "max_features": [5,7,9,10,14,15,19,20,22,25,27],
              "min_samples_split": [5,10,20,40,80,100,120,140],
              "min_samples_leaf": [10,20,40,50,60,70,80,90]
              }


tree = DecisionTreeRegressor()

# Instantiate the GridSearchCV object: tree_cv
tree_cv = GridSearchCV(tree,param_dist,scoring='r2',cv=5,n_jobs=-1)
tree_cv.fit(X_train_ohe,y_train)


print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[ ]:


dtreg = DecisionTreeRegressor(**tree_cv.best_params_) # fitting with optimal parameter values
dtreg.fit(X_train_ohe,y_train)

y_tr_pred = dtreg.predict(X_train_ohe)
tr_mse = mean_squared_error(y_train,y_tr_pred)
y_te_pred = dtreg.predict(X_test_ohe)
te_mse = mean_squared_error(y_test,y_te_pred)

print('train rmse decision tree regressor with best parameters:',np.sqrt(tr_mse))
print('test rmse decision tree regressor with best parameters:',np.sqrt(te_mse))


# #### 2.2: DT regression with Frequency Encoded Feature Set

# In[ ]:


dtreg = DecisionTreeRegressor(random_state=42)
dtreg.fit(X_train_freq,y_train) # fitting with frequency encoded features

y_tr_pred = dtreg.predict(X_train_freq)
tr_mse = mean_squared_error(y_train,y_tr_pred)
y_te_pred = dtreg.predict(X_test_freq)
te_mse = mean_squared_error(y_test,y_te_pred)

print('train rmse decision tree regressor:',np.sqrt(tr_mse))
print('test rmse decision tree regressor:',np.sqrt(te_mse))


# In[ ]:


# Setup the parameters 
param_dist = {"max_depth": [3,4,5,6,7,8,9,10,11,15,19,20,None],
              "max_features": [5,7,9,10,14,15,19,20,22,25,27],
              "min_samples_split": [5,10,20,40,80,100,120,140],
              "min_samples_leaf": [10,20,40,50,60,70,80,90]
              }


tree = DecisionTreeRegressor()


tree_cv = GridSearchCV(tree,param_dist,scoring='r2',cv=5,n_jobs=-1)
tree_cv.fit(X_train_freq,y_train)


print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[ ]:


dtreg = DecisionTreeRegressor(**tree_cv.best_params_) # fitting with optimal values
dtreg.fit(X_train_freq,y_train)

y_tr_pred = dtreg.predict(X_train_freq)
tr_mse = mean_squared_error(y_train,y_tr_pred)
y_te_pred = dtreg.predict(X_test_freq)
te_mse = mean_squared_error(y_test,y_te_pred)

print('train rmse decision tree regressor with best parameters:',np.sqrt(tr_mse))
print('test rmse decision tree regressor with best parameters:',np.sqrt(te_mse))


# In[ ]:


train_mse, test_mse = [],[]

# evaluate a decision tree for each depth
for n in range(20):
    n = n+1
    #configure the model
    dtreg = DecisionTreeRegressor(**tree_cv.best_params_)
    
    # fit model on the training dataset
    dtreg.fit(X_train_freq,y_train)
    
    # evaluate on the train dataset
    y_tr_pred = dtreg.predict(X_train_freq)
    tr_mse = mean_squared_error(y_train,y_tr_pred)
    train_mse.append(tr_mse)
    
    # evaluate on the test dataset
    y_te_pred = dtreg.predict(X_test_freq)
    te_mse = mean_squared_error(y_test,y_te_pred)
    test_mse.append(te_mse)
    
    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, tr_mse, te_mse))
    
#plot of train and test scores vs tree depth
n = [i for i in range(1,21)]
plt.plot(n,np.sqrt(test_mse),'-o',label='test rmse')
plt.plot(n,np.sqrt(train_mse),'-o',label='train rmse')
plt.xticks([1,3,5,7,9,11,13,15,17,19])
plt.legend()
plt.show()


# - Overall, decision tree regressor showed significant improvement over the base model very much in line with knn regressor

# ## Base Models Summary

# ### OLS Base Model Summary

# **Interpretation:**
# 
# The R-squared value obtained from this model is 0.761 which means that the above model explains 76.1% of the variation in the First Year Modeled Project Energy Savings $ Estimate.
# 
# **Durbin-Watson Test:**
# 
# The test is used to check the autocorrelation between the residuals.
# 
# - If the Durbin-Watson test statistic is near to 2: no autocorrelation
# - If the Durbin-Watson test statistic is between 0 and 2: positive autocorrelation
# - If the Durbin-Watson test statistic is between 2 and 4: negative autocorrelation
# 
# The summary output shows that the value of the test statistic is close to 2 (= 2.047) which means there is no autocorrelation.
# 
# **Jarque-Bera Test:**
# 
# The test is used to check the normality of the residuals. Here, the p-value of the test is less than 0.05; that implies the residuals are not normally distributed.
# 
# **'Cond. No':**
# 
# (= 1) represents the Condition Number (CN) which is used to check the multicollinearity.
# 
# - If CN < 100: no multicollinearity
# - If CN is between 100 and 1000: moderate multicollinearity
# - If CN > 1000: severe multicollinearity
# 
# Thus, it can be seen that there is no multicollinearity in the data.
# 

# ### ML Base Models Summary

# | Algorithm | Encoding | Test MSE |
# | --- | --- | --- |
# | Base Model | (Irrespective) | 92908 |
# | KNN Regressor | One hot Encoding | 19398 |
# | KNN Regressor | Frequency Encoding | 23563 |
# | Decision Tree Regressor | One hot Encoding | 22840 |
# | Decision Tree Regressor | Frequency Encoding | 20049 |

# - KNN regressor with one hot encoding has emerged the best so far with decision tree regressor+ frequency encoding giving the next best Mean Squared Error on test data
