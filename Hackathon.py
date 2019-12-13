import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,ShuffleSplit
import xgboost as xgb
from sklearn.metrics import confusion_matrix,classification_report
import pickle



def save_to_pickle(filename,model):
    pickle.dump(model,open(filename,'wb'))



df = pd.read_csv('census_income_dataset.csv',index_col = None)

# df.describe()
# df.dtypes
# df.head(1000)

#Here we are converting the income values which are less than 50k to '0' and greater than 50k to '1'
df['income_level']=df['income_level'].map({'<=50K': 0, '>50K': 1})

#Plotting workclass vs income_level for data analysis.
df_plot1 = df.groupby(['income_level','workclass']).size().reset_index().pivot(columns = 'income_level',index = 'workclass',values = 0)
df_plot1.plot(kind = 'bar',stacked = True)
plt.ylabel('count')

#Plotting education vs income_level for data analysis.
df_plot2 = df.groupby(['income_level','education']).size().reset_index().pivot(columns = 'income_level',index = 'education',values = 0)
df_plot2.plot(kind = 'bar',stacked = True)
plt.ylabel('count')


#Plotting occupation vs income_level for data analysis.
df_plot3 = df.groupby(['income_level','occupation']).size().reset_index().pivot(columns = 'income_level',index = 'occupation',values = 0)
df_plot3.plot(kind = 'bar',stacked = True)
plt.ylabel('count')


#Plotting marital_status vs income_level for data analysis.
df_plot4 = df.groupby(['income_level','marital_status']).size().reset_index().pivot(columns = 'income_level',index = 'marital_status',values = 0)
df_plot4.plot(kind = 'bar',stacked = True)
plt.ylabel('count')


#Plotting race vs income_level for data analysis.
df_plot5 = df.groupby(['income_level','race']).size().reset_index().pivot(columns = 'income_level',index = 'race',values = 0)
df_plot5.plot(kind = 'bar',stacked = True)
plt.ylabel('count')

#Plotting sex vs income_level for data analysis.
df_plot6 = df.groupby(['income_level','sex']).size().reset_index().pivot(columns = 'income_level',index = 'sex',values = 0)
df_plot6.plot(kind = 'bar',stacked = True)
plt.ylabel('count')


#countplot for income_level
# sns.countplot(x="income_level",data=df)

#since education and education_num are highly correlated, we are removing education_num from the dataframe
df = df.drop(["education_num"],axis = 1)


#Here, we are checking whether any column is containing '?' value.
# print(df['workclass'].isin(['?']).sum())
# print(df['occupation'].isin(['?']).sum())
# print(df['age'].isin(['?']).sum())
# print(df['relationship'].isin(['?']).sum())
# print(df['fnlwgt'].isin(['?']).sum())
# print(df['marital_status'].isin(['?']).sum())
# print(df['race'].isin(['?']).sum())
# print(df['sex'].isin(['?']).sum())
# print(df['capital_gain'].isin(['?']).sum())
# print(df['capital_loss'].isin(['?']).sum())
# print(df['hours_per_week'].isin(['?']).sum())
# print(df['native_country'].isin(['?']).sum())
# print(df['income_level'].isin(['?']).sum())


#Replacing the '?' with NaN. After this we plotted the graphs for each column by dropping the rows containing NaNs.
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)
df['native_country'] = df['native_country'].replace('?',np.nan)


#Removing the columns containing the NaN values for plotting.
df_fill = df.dropna()


#Checking the number of unique values in each column to check whether we can apply one hot encoding or not.
print("Number of unique values in workclass column : ",df['workclass'].nunique())
print("Number of unique values in occupation column : ",df['occupation'].nunique())
print("Number of unique values in marital_status column : ",df['marital_status'].nunique())
print("Number of unique values in relationship column : ",df['relationship'].nunique())
print("Number of unique values in race column : ",df['race'].nunique())
print("Number of unique values in sex column : ",df['sex'].nunique())
print("Number of unique values in native_country column : ",df['native_country'].nunique())


#Label encoding the data to get an idea of their distribution.
label_columns = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']
lencoder = LabelEncoder()
encoded = df_fill[label_columns].apply(lencoder.fit_transform)

pd.options.mode.chained_assignment = None
#replacing the string columns with their corresponding label encoderd columns.
df_fill.loc[:,['workclass','education','marital_status','occupation','relationship','race','sex','native_country']] = encoded.copy()


#Plotting the correlation matrix to get an idea of the relation between the columns
df_corr = df_fill[['age','workclass','education','occupation','race','sex','hours_per_week','native_country','income_level']]
corr = df_corr.corr()
corr.style.background_gradient(cmap = 'coolwarm')


#Distribution plot of occupation.
df_fill['occupation'].plot.kde()
plt.xlabel('occupation')


#Distribution plot of workclass.
df_fill['workclass'].plot.kde()
plt.xlabel('workclass_label')


# Distribution plot of native country.
df_fill['native_country'].plot.kde()
plt.xlabel("country_label")


#Replacing the NaN values in "native_country" with the mode (Here, 'United-States')
df['native_country'] = df['native_country'].fillna("United-States")


#Replacing the NaN values in "workclass" with the mode (Here,'Private')
df['workclass'] = df['workclass'].fillna("Private")


#Getting the number of NaN values in "occupation"
print("Number of missing values in occupation column : ",df['occupation'].isnull().sum())


#Imputing the values of occupation using other coulms and logistic regression.
log_reg = LogisticRegression(solver='liblinear',multi_class='auto')


x = df_corr.loc[:,['age','workclass','education','race','sex','hours_per_week','native_country','income_level']]
y = df_corr.loc[:,['occupation']]
log_reg.fit(x,y.values.ravel())


#Creating a new dataframe new_df for imputing the missing values in df.
df1 = pd.DataFrame(index = range(len(df)),columns = ['occupation'])


temp_data = df
temp_data['workclass'] = lencoder.fit_transform(temp_data['workclass'])
temp_data['education'] = lencoder.fit_transform(temp_data['education'])
temp_data['race'] = lencoder.fit_transform(temp_data['race'])
temp_data['sex'] = lencoder.fit_transform(temp_data['sex'])
temp_data['native_country'] = lencoder.fit_transform(temp_data['native_country'])


df1['occupation'] = pd.DataFrame(log_reg.predict(temp_data.loc[:,['age','workclass','education','race','sex','hours_per_week','native_country','income_level']]))


#replacing the NaN values with the predicted values.
df['occupation'].fillna(df1['occupation'],inplace=True)


#converting the labels back to their names for applying one-hot encoding
df = df.replace({'occupation': {0:'Adm-clerical' ,1:'Armed-Forces',2:'Craft-repair',3:'Exec-managerial',
                          4:'Farming-fishing',5:"Handlers-cleaners",6:"Machine-op-inspct",7:"Other-service",
                          8:"Priv-house-serv",9:"Prof-specialty",10:"Protective-serv",11:"Sales",
                          12:"Tech-support",13:"Transport-moving"}})


#Appling One Hot Encoding to the dataframe df and storing it in new_df.
new_df = pd.DataFrame()
for feature in label_columns:
    temp = pd.get_dummies(df[feature],prefix = feature)
    new_df = pd.concat([new_df,temp],axis=1)


new_df = pd.concat([new_df,df],axis=1)
# as we encoded label columns, we are removing original label columns
new_df = new_df.drop(label_columns,axis=1)
# new_df.describe()


#Here, we are normalizing the columns (['age','fnlwgt','capital_gain','capital_loss','hours_per_week']).
norm_df = new_df.loc[:,['age','fnlwgt','capital_gain','capital_loss','hours_per_week']]
norm_df = (norm_df-norm_df.min())/(norm_df.max() - norm_df.min())


#Replacing the normalized columns with corresponding columns in new_df.
new_df.loc[:,['age','fnlwgt','capital_gain','capital_loss','hours_per_week']] = norm_df


# new_df.describe()

#Splitting the data into train and test data for training a model and testing it.
X_train, X_test, y_train, y_test = train_test_split(new_df.drop('income_level',axis=1),new_df['income_level'], test_size=0.30,random_state=1, shuffle = True)

# predicting using LogisticRegression
logmodel = LogisticRegression(solver='liblinear', multi_class='auto')
logmodel.fit(X_train,y_train)
predict_log = logmodel.predict(X_test)
# logmodel.score(X_test,y_test)

# predicting using xgboost
# model_xgb=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
# model_xgb.fit(X_train, y_train)
# predict_xgb = model_xgb.predict(X_test)
# model_xgb.score(X_test,y_test)

# predicting using RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, y_train)
# predict_rf = rf.predict(X_test)
# rf.score(X_test, y_test)

# saving RandomForestClassifier to pickle file and running model using file
# save_to_pickle("Rf_model.sav",rf)
load_rf = pickle.load(open("Rf_model.sav","rb"))
load_rf.fit(X_train,y_train)
predict_log = load_rf.predict(X_test)
print("Accuracy using RandomForestClassifier : " ,load_rf.score(X_test,y_test))

# predicting using DecisionTreeClassifier
# model_tree=tree.DecisionTreeClassifier()
# model_tree.fit(X_train, y_train)
# predict_tree = model_tree.predict(X_test)
# model_tree.score(X_test,y_test)



# Below code is only for demonstration of different models
models = []
models.append(('logistic', LogisticRegression(solver='liblinear',multi_class='auto')))
models.append(('decision tree', tree.DecisionTreeClassifier()))
models.append(('Xgboost' ,xgb.XGBClassifier()))
models.append(('random forest', load_rf))

results = []
names = []
for name, model in models:
    cv = ShuffleSplit()
    cv_results = cross_val_score(model, X_train, y_train, scoring='accuracy',cv = cv)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " % (name, cv_results.mean())
    # print(msg)


fig = plt.figure()
fig.suptitle('Algorith Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# to run this, uncomment algorithms of LogisticRegression,xgboost,DecisionTreeClassifier

# cfm = confusion_matrix(predict_rf, y_test)
# sns.heatmap(cfm, annot=True)
# plt.xlabel('Predicted classes')
# plt.ylabel('Actual classes')
# plt.show()

# print("\nLogistic Regression")
# print(classification_report(y_test, predict_log))
# print("\nXgboost")
# print(classification_report(y_test, predict_xgb))
# print("\nRandom Forest Classifier")
# print(classification_report(y_test, predict_rf))
# print("\nDescision tree")
# print(classification_report(y_test, predict_tree))
