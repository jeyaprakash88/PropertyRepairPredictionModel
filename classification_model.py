#!/usr/bin/env python
# coding: utf-8

# 
# ## Predicting Repair demand
# 
# [1. EXPLORATORY DATA ANALYSIS](#1)  
#   
# [2. DATA VISUALIZATION](#5)
# 
# [3. STATISTICAL ANALYSIS](#2)
# 
# [4. HANDLING MISSING VALUES](#3)
# 
# [5. CORRELATION ](#4)
# > [4.1. Pearson's correlation](#41)
# 
# > [4.2. Spearman's correlation](#42)
# 
# [6. FEATURE SELECTION](#6)
# 
# [7. MODEL DEVELOPMENT](#7)
#    >> [7.1. Model with manual preprocessing](#71)   
#    >> [7.2. Model with selected feature](#72)  
#    >> [7.3. Final Model](#73)
# 
# [8. HYPERPARAMETER](#8)
#    >> [8.1. Hyperparameter optimisation using grid search](#81)   
#    >> [8.2. Hyperparameter optimisation using Random search](#82)  
#    >> [8.3. Feature Importance](#83)
#    _______________________________
# 

import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")

from sklearn.exceptions import FitFailedWarning 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression


# ## Load the data :



# Data path
#load and store data into a dataframe
df = pd.read_excel('data.xlsx')


# ## 1.Exploratory Data Analysis by analyzing the dataset<a name="1"></a>

df.info()

# Results
print(f'Data dimension: {df.shape}')
print (df.columns)

# Display the first few rows of the DataFrame
df.head()

# Check the summary statistics of numeric columns
df.describe()

df.dtypes

# 5.Descriptive summary of the dataset:
df_num = df[df.describe().columns]
display(df_num.describe().round(2))
df_num.boxplot()

pd.DataFrame(df.nunique())


# Set the figure size
plt.figure(figsize=(13, 6))

# Create the count plot
ax = sns.countplot(x='pty_classification', data=df)

# Set labels and title
plt.xlabel('Priority Classification')
plt.ylabel('Count')
plt.title('Priority Classification Count')

# Add count labels on top of each bar
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Display the plot
plt.show()

# Count plot for categorical columns
categorical_cols = ['job_type', 'jobsts-cde', 'postcode', 'loc-nam-2', 'Building_type', 'tenu_cde', 'rntpaymd_cde', 'ttncytyp_cde', 'pty_classification_subtype']
for col in categorical_cols:
    plt.figure(figsize=(15, 6))
    sns.countplot(x=col, data=df)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Count Plot of {col}')
    plt.show()

import matplotlib.pyplot as plt

# Get data
categories = df['job_type'].unique()
values = df['job_type'].value_counts()

# Calculate percentages
values_percent = (values / values.sum()) * 100

# Plot pie 
fig, ax = plt.subplots(figsize=(8,8))
wedges, texts, autotext = ax.pie(values, autopct='%1.1f%%', startangle=90)

# Add category labels
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center", size=10)

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    
    ax.annotate(f"{categories[i]} ({values_percent[i].round(1)}%)", xy=(x,y), xytext=(1.4*np.sign(x), 1.4*y),
                horizontalalignment = horizontalalignment, **kw)

plt.show()

categories = df['pty_classification'].unique()
values = df['pty_classification'].value_counts()

# Create donut chart
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal")) 

wedges, texts = ax.pie(values, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"), 
          bbox=bbox_props, zorder=0, va="center")

# Get percent values 
values_percent = (values/sum(values)) * 100

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    # Add percent to annotation
    percent = values_percent[i].round(1)
    ax.annotate("{0} ({1}%)".format(categories[i], percent), 
                xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), 
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Donut Chart")
plt.show()


building_types = df['Building_type'] 
building_type_counts = building_types.value_counts()

# Create a custom color palette with a unique color for each building type
color_palette = sns.color_palette("Set3", n_colors=len(building_type_counts))

# Create bar chart
plt.figure(figsize=(10, 6))  # Set plot size

bars = plt.bar(building_type_counts.index, building_type_counts.values, color=color_palette)

plt.xlabel('Building Type')
plt.ylabel('Count')
plt.title('Distribution of Building Types')

# Add labels to the bars
for bar in bars:
    height = bar.get_height()
    plt.annotate('{}'.format(height),
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Stacked Bar Plot
stacked_counts = df.groupby(['job_type', 'pty_classification']).size().unstack(fill_value=0)
stacked_counts.plot(kind='bar', stacked=True, figsize=(15, 6))
plt.title('Job Type vs Priority Classification')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.legend(title='Priority Classification')
plt.show()

# Histograms for numeric columns
numeric_cols = ['pr-seq-no', 'void-num', 'str-cde', 'Age', 'ownership-%', 'No_of_bedroom', 'Days_to_complete']
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.show()


# In[16]:


# Checking for outliers using boxing only on Numerical features
for cat in numeric_cols:
    plt.figure(figsize=(20,1))    
    sns.boxplot(data=df, x=cat,palette='Blues')
 
#Pearson's correlation
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="Blues")


# ### 4.2. Spearman's correlation <a name="42"></a>

#Spearman's correlation
corr = df.corr(method='spearman')
g = sns.heatmap(corr, annot=True, cmap="Blues")
sns.despine()
g.figure.set_size_inches(10,10)

# Matches the above value counts
print(df['pty_classification'].value_counts())
print(df['sortrd-cde-1'].unique())
print(df['jobsts-cde'].unique())
print(df['postcode'].unique())
print(df['loc-nam-2'].unique())
print(df['Building_type'].unique())
print(df['tenu_cde'].unique())
print(df['rntpaymd_cde'].unique())
print(df['ttncytyp_cde'].unique())
print(df['pty_classification_subtype'].unique())

# Find the missing data by percentage
total = df.isnull().sum()
percentage = (total/df.isnull().count()).round(4)*100
NAs = pd.concat([total,percentage],axis=1,keys=('Total','Percentage'))
NAs[NAs.Total>0].sort_values(by='Total',ascending=False)

df.dropna(subset = ['postcode'], inplace=True)
df = df.drop(columns=['num', 'job_type', 'rtb-dat', 'demolish-dat', 'prtyp-cde', 'construction-yr'])
df['Days_to_complete'].fillna(int(df['Days_to_complete'].mean()), inplace=True)
df['No_of_bedroom'] = df['No_of_bedroom'].fillna(0)
df['ownership-%'] = df['ownership-%'].fillna(100)
df['jobsts-cde'] = df['jobsts-cde'].astype(str)
df['postcode'] = df['postcode'].replace(0, method='ffill')

# Split the postcode column into two columns
df[['postcode_part1', 'postcode_part2']] = df['postcode'].str.split(' ', 1, expand=True)

# Define a mapping dictionary for town names
town_mapping = {
    'GL1': 'Gloucester',
    'GL10': 'Stonehouse',
    'GL2': 'Gloucester',
    'GL20': 'Ashchurch',
    'GL3': 'Hucclecote',
    'GL4': 'Gloucester',
    'GL5': 'Rodborough',
    'GL51': 'Cheltenham',
    'GL6': 'Thrupp'
}

# Map the values in the postcode_part1 column to the corresponding town names
df['town'] = df['postcode_part1'].map(town_mapping)

df['pty_classification'].unique()

# Splitting the data into features and target
df1 = df
X = df1.drop(['pty_classification'], axis=1)
y = df1['pty_classification']

# Encoding the labels for classification problems
label_encode = LabelEncoder()
labels = label_encode.fit_transform(y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), ['pr-seq-no', 'void-num', 'str-cde', 'Age', 'ownership-%', 'No_of_bedroom', 'Days_to_complete']),
    
    ('categorical', OneHotEncoder(), ['sortrd-cde-1', 'jobsts-cde', 'loc-nam-2', 'town', 'Building_type', 'tenu_cde', 'rntpaymd_cde', 'right-to-repair', 'ttncytyp_cde', 'pty_classification_subtype'])
])

print(X.shape)
print(y.shape)
print("X_train shape {} and size {}".format(X_train.shape,X_train.size))
print("X_test shape {} and size {}".format(X_test.shape,X_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_models(models, X_train, X_test, y_train, y_test, preprocessor):
    results, cm, report = [], [], []

    for name, model in models:
        # Create a pipeline with preprocessing and the classifier
        myClassifier = Pipeline([('preprocessing', preprocessor), ('classifier', model)])
        # Fit the pipeline on the training data
        myClassifier.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = myClassifier.predict(X_test)

        # Calculate classification report and confusion matrix
        report.append(classification_report(y_test, y_pred))
        cm.append(metrics.confusion_matrix(y_test, y_pred))

        # Calculate and store the accuracy score
        score = accuracy_score(y_test, y_pred)
        results.append("The Score of %s is: %2.2f" % (name, score))
    
    for result, confusion_matrix, classification_reports in zip(results, cm, report):
        print(result, '\n', classification_reports)
        
        unique_labels = sorted(set(y_test))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=unique_labels)
        
        # Set the figure size to display all label names
        plt.figure(figsize=(len(unique_labels), len(unique_labels)))
        
        cm_display.plot(xticks_rotation=90)
        plt.show()
    
    return results, cm, report


# ### Logistic Regression

# Define the models
models = [('LogisticRegression', LogisticRegression())]

# Call the function to evaluate models
results, cm, report = evaluate_models(models, X_train, X_test, y_train, y_test, preprocessor)


# ### Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Define the models
models = [('DecisionTreeClassifier', DecisionTreeClassifier())]

# Call the function to evaluate models
results, cm, report = evaluate_models(models, X_train, X_test, y_train, y_test, preprocessor)


# ### BaggingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def bagging_models(models, X_train, X_test, y_train, y_test, preprocessor):
    results, cm, report = [], [], []
    for name, model in models:
        # Create a pipeline with preprocessing and the classifier
        myClassifier = Pipeline([('preprocessing', preprocessor), ('classifier', model)])

        # Fit the pipeline on the training data
        myClassifier.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = myClassifier.predict(X_test)

        # Calculate classification report and confusion matrix
        report.append(classification_report(y_test, y_pred))
        cm.append(confusion_matrix(y_test, y_pred))

        # Calculate and store the accuracy score
        score = accuracy_score(y_test, y_pred)
        results.append("The Score of %s is: %2.2f" % (name, score))

    return results, cm, report
models = [('BaggingClassifier', BaggingClassifier())]

results, cm, report = bagging_models(models, X_train, X_test, y_train, y_test, preprocessor)

for result, confusion_matrix, classification_report in zip(results, cm, report):
    print(result, '\n', classification_report)
    # Create the confusion matrix display with the correct labels
    unique_labels = sorted(set(y_test))  # Get unique class labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=unique_labels)

    # Set the figure size to display all label names
    plt.figure(figsize=(len(unique_labels), len(unique_labels)))

    cm_display.plot(xticks_rotation=90)
    plt.show()


# ### GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder, OrdinalEncoder

def gradient_model(models, X_train, X_test, y_train, y_test, preprocessor):
    results, cm, report = [], [], []
    for name, model in models:
        # Create a pipeline with preprocessing and the classifier
        myClassifier = Pipeline([('preprocessing', preprocessor), ('classifier', model)])
        # Fit the pipeline on the training data
        myClassifier.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = myClassifier.predict(X_test)

        # Calculate classification report and confusion matrix
        report.append(classification_report(y_test, y_pred))
        cm.append(confusion_matrix(y_test, y_pred))

        # Calculate and store the accuracy score
        score = accuracy_score(y_test, y_pred)
        results.append("The Score of %s is: %2.2f" % (name, score))

    return results, cm, report
models = [('GradientBoostingClassifier', GradientBoostingClassifier())]

results, cm, report = gradient_model(models, X_train, X_test, y_train, y_test, preprocessor)

y_test_decoded = label_encoder.inverse_transform(y_test_encoded)

for result, confusion_matrix, classification_report in zip(results, cm, report):
    print(result, '\n', classification_report)
    unique_labels = sorted(set(y_test_decoded))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=unique_labels)

    # Set the figure size to display all label names
    plt.figure(figsize=(len(unique_labels), len(unique_labels)))

    cm_display.plot(xticks_rotation=90)
    plt.show()


# ### XGBClassifier

# Import the necessary libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# Assuming 'df' is the DataFrame containing the data
X = df.drop(['pty_classification', 'postcode', 'postcode_part1', 'postcode_part2', 'right-to-repair'], axis=1)
y = df['pty_classification']

# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Define the column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), ['pr-seq-no', 'void-num', 'str-cde', 'Age', 'ownership-%', 'No_of_bedroom', 'Days_to_complete']),
    ('categorical', OneHotEncoder(), ['sortrd-cde-1', 'jobsts-cde', 'loc-nam-2', 'Building_type', 'tenu_cde', 'rntpaymd_cde', 'ttncytyp_cde', 'pty_classification_subtype', 'town'])
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Fit the preprocessor on the training data and transform both train and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Create the XGBoost classifier
model = XGBClassifier()

# Fit the model on the preprocessed training data
model.fit(X_train_preprocessed, y_train)

# Make predictions on the preprocessed test data
y_pred = model.predict(X_test_preprocessed)

# Calculate classification report and confusion matrix
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Calculate and store the accuracy score
accuracy = accuracy_score(y_test, y_pred)
result = "The Score of XGBClassifier is: %2.2f" % accuracy

# Display the classification report and confusion matrix
print(result)
print(classification_report_result)
unique_labels = label_encoder.inverse_transform(sorted(set(y_test)))
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=unique_labels)
plt.figure(figsize=(len(unique_labels), len(unique_labels)))
cm_display.plot(xticks_rotation=90)
plt.show()


# ### Export the model as pickle

import joblib
import pickle

data = {"model" : model, "preprocessor":preprocessor, "label_encoder": label_encoder}
with open ('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# ## XGB Classifier with SMOTE
# 
# SMOTE addresses class imbalance in real-world datasets by generating synthetic examples for the minority class, balancing class distribution and enabling the model to learn from it. This improves generalization performance by preventing overfitting on the majority class and providing additional diversity and complexity in training data.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Assuming 'df' is the DataFrame containing the data
X = df.drop(['pty_classification', 'postcode', 'postcode_part1', 'postcode_part2', 'right-to-repair'], axis=1)
y = df['pty_classification']


# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Define the column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('numeric', StandardScaler(), ['pr-seq-no', 'void-num', 'str-cde', 'Age', 'ownership-%', 'No_of_bedroom', 'Days_to_complete']),
    ('categorical', OneHotEncoder(), ['sortrd-cde-1', 'jobsts-cde', 'loc-nam-2', 'Building_type', 'tenu_cde', 'rntpaymd_cde', 'ttncytyp_cde', 'pty_classification_subtype', 'town'])
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to balance the class distribution in the training data
smote = SMOTE(random_state=42)
X_train_preprocessed, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Transform the test data using the preprocessor
X_test_preprocessed = preprocessor.transform(X_test)

# Create the XGBoost classifier
model = XGBClassifier()

# Fit the model on the resampled training data
model.fit(X_train_preprocessed, y_train_resampled)

# Make predictions on the preprocessed test data
y_pred = model.predict(X_test_preprocessed)

# Calculate classification report and confusion matrix
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Calculate and store the accuracy score
accuracy = accuracy_score(y_test, y_pred)
result = "The Score of XGBClassifier with smote is: %2.2f" % accuracy

# Display the classification report and confusion matrix
print(result)
print(classification_report_result)
unique_labels = label_encoder.inverse_transform(sorted(set(y_test)))
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=unique_labels)
plt.figure(figsize=(len(unique_labels), len(unique_labels)))
cm_display.plot(xticks_rotation=90)
plt.show()

# The comparison between an XGBClassifier model with and without using the SMOTE (Synthetic Minority Over-sampling Technique) technique for handling imbalanced classes in our dataset. From the classification report, we are drawing the conclusion that the XGBClassifier without SMOTE is performing well, especially in predicting the minor classes.

# ### ROC curve and AUC
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Binarize the labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_bin = lb.transform(y_pred)

# Calculate ROC curve and ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-Class Classification')
plt.legend(loc="lower right")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Create the XGBoost classifier
model = XGBClassifier()

# Define the number of folds for cross-validation
n_splits = 5

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store results
accuracy_scores = []

# Perform k-fold cross-validation
for train_index, val_index in kf.split(X_train_preprocessed, y_train):
    X_train_fold, X_val_fold = X_train_preprocessed[train_index], X_train_preprocessed[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Fit the model on the training fold
    model.fit(X_train_fold, y_train_fold)
    
    # Make predictions on the validation fold
    y_val_pred = model.predict(X_val_fold)
    
    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_val_fold, y_val_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)

print(f'Average Accuracy: {average_accuracy:.4f}')

# Create a boxplot to visualize the cross-validation results
plt.figure(figsize=(8, 6))
plt.boxplot(accuracy_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy Distribution')
plt.xticks(np.arange(1, n_splits+1), labels=np.arange(1, n_splits+1))
plt.show()


# ### **Hyperparameter search using random search**<a name="82"></a>
# Imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.1, 0.01, 0.05],
    'subsample': [0.5, 0.8, 1.0]
}

# Random Search CV
random_search = RandomizedSearchCV(estimator=XGBClassifier(),
                                   param_distributions=param_grid,
                                   n_iter=10,
                                   scoring='roc_auc',
                                   n_jobs=-1, 
                                   cv=5,
                                   verbose=1)
                                   
# Fit on pipeline                                   
random_search.fit(X_train_preprocessed, y_train)

# Best hyperparameters
print(random_search.best_params_)

# Extract best model 
best_model = random_search.best_estimator_
best_model


# # Model with Hyperparameters
# without hyper parameter tuning our model is performing well hence we select the base model with change in thresh hold.
# Initialize XGBoost classifier with the best hyperparameters
best_model = XGBClassifier(subsample=0.8, n_estimators=500, max_depth=6, learning_rate=0.05)
sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(X_train_processed, y_train)

# Fit the model on the preprocessed training data
best_model.fit(X_train_res, y_train_res)

# Make predictions on the preprocessed test data
y_pred = myClassfier.predict(X_test_preprocessed)

# Calculate classification report and confusion matrix
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Calculate and store the accuracy score
accuracy = accuracy_score(y_test, y_pred)
result = "The Score of XGBClassifier is: %2.2f" % accuracy

# Display the classification report and confusion matrix
print(result)
print(classification_report_result)
unique_labels = label_encoder.inverse_transform(sorted(set(y_test)))
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result, display_labels=unique_labels)
plt.figure(figsize=(len(unique_labels), len(unique_labels)))
cm_display.plot(xticks_rotation=90)
plt.show()


# ## Feature Importance<a name="83"></a>
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Create the XGBoost classifier
model = XGBClassifier()

# Fit model on the preprocessed training data
model.fit(X_train_preprocessed, y_train)

# Plot feature importance
plot_importance(model, max_num_features=15, importance_type='weight')  # You can also use 'gain' or 'cover'
plt.xticks(rotation='vertical')  # Rotate x-axis labels for better visibility
plt.show()
