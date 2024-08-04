#!/usr/bin/env python
# coding: utf-8

# # Project Name: Differentiated Thyroid Cancer Recurrence

# ### Project Overview: 
# This project focuses on analyzing a dataset related to thyroid conditions. The type of problem is a supervised learning task, where the goal is to predict the recurrence of thyroid issues based on various patient characteristics and medical history.
# 
# ### Project Goal : 
# The primary objective of this project is to determine the factors that contribute to the recurrence of thyroid conditions. Understanding these factors is crucial for improving patient outcomes and optimizing treatment strategies. By identifying key predictors, the project aims to assist medical professionals in making informed decisions regarding patient care.
# 
# ### Data Source:  
# The data is imported from https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence  in a CSV file format named "Thyroid_Diff.csv". This data set contains 13 clinicopathologic features aiming to predict recurrence of well differentiated thyroid cancer. The data set was collected over a duration of 15 years and each patient was followed for at least 10 years. The dataset contains various attributes related to patient demographics, medical history, thyroid function, physical examination results, pathology, focality, risk assessment, stage, response to treatment, and recurrence status.
# 
# ### Data Description: 
# 
# #### Size:The Dataset contains 383 rows and 17 Columns
# #### Dataset Characteristics: Tabular
# 
# #### Features:
# Age: Age of the patient
# 
# Gender: Gender of the patient (F for female, M for male)
# 
# Smoking Hx: Smoking history (Yes/No)
# 
# Smoking Hx Radiotherapy: History of smoking and radiotherapy (Yes/No)
# 
# Thyroid Function: Thyroid function status (e.g., Euthyroid) 
# 
# Physical Examination: Results of physical examination (e.g., Single nodular goiter) 
# 
# Adenopathy: Presence of adenopathy (Yes/No) 
# 
# Pathology: Pathological findings (e.g., Micropapillary) 
# 
# Focality: Focality of the condition (e.g., Uni-Focal, Multi-Focal) 
# 
# Risk: Risk level (e.g., Low) T: Tumor size (e.g., T1a) 
# 
# N: Lymph node involvement (e.g., N0) 
# 
# M: Metastasis (e.g., M0) Stage: Stage of the condition (e.g., I) 
# 
# Response: Response to treatment (e.g., Excellent, Indeterminate) 
# 
# Recurred: Recurrence status (Yes/No) 
# 
# 

# In[1]:


#Import neccessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
# Set color map to have light blue background
sns.set()
import statsmodels.formula.api as smf
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the data from the provided CSV file
file_path = 'data/Thyroid_Diff.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


missing_values = data.isnull().sum()
missing_values


# In[6]:


data.shape


# In[7]:


# Check the distribution of the target variable
recurrence_distribution = data['Recurred'].value_counts()

recurrence_distribution


# ## Exploratory Data Analysis (EDA):
# #### Analyze the relationships between the features and the target variable. 

# In[8]:


# Display the column names to check for any discrepancies
data.columns


# In[9]:


# Plot the distribution of recurrence by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Gender', hue='Recurred')
plt.title('Distribution of Recurrence by Gender')
plt.show()


# In[10]:


# Plot the distribution of recurrence by smoking history
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Smoking', hue='Recurred')
plt.title('Distribution of Recurrence by Smoking History')
plt.show()


# In[24]:


# Plot the distribution of recurrence by thyroid function
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Thyroid Function', hue='Recurred')
plt.title('Distribution of Recurrence by Thyroid Function')
plt.show()


# In[12]:


# Plot the distribution of recurrence by physical examination
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Physical Examination', hue='Recurred')
plt.title('Distribution of Recurrence by Physical Examination')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[13]:


# Plot the distribution of recurrence by pathology
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Pathology', hue='Recurred')
plt.title('Distribution of Recurrence by Pathology')
plt.show()


# ### Summary of Exploratory Data Analysis (EDA) Gender:
# Both males and females have instances of recurrence, with a higher proportion of females in the dataset. The recurrence rate seems relatively balanced between genders. 
# 
# Smoking History: Patients with a history of smoking show a slightly higher rate of recurrence compared to non-smokers. 
# 
# Thyroid Function: Most patients are euthyroid (normal thyroid function). The recurrence rate appears similar across different thyroid function statuses. 
# 
# Physical Examination: Various physical examination findings show different recurrence rates. Multinodular goiters might have a slightly higher recurrence rate. 
# 
# Pathology: Micropapillary is the most common pathology type. The recurrence rate varies, but there are visible patterns that need further analysis.
# 

# ### Data Preprocessing:
# 
# #### Handle missing values. Encode categorical variables. Normalize numerical features
# 

# In[14]:


# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function',
                       'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk',
                       'T', 'N', 'M', 'Stage', 'Response']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the dataset into training and testing sets
X = data.drop('Recurred', axis=1)
y = data['Recurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()


# In[15]:


# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# The data preprocessing step has successfully encoded the categorical variables and split the dataset into training and testing sets

#  
# ### Modeling: 
# #### Apply various machine learning models to predict recurrence and evaluate their performance.

# In[16]:


# Initialize the models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
#     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Parameter grids for hyperparameter tuning
param_grids = {
    'Decision Tree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    },
#     'XGBoost': {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 6, 10],
#         'learning_rate': [0.01, 0.1, 0.2]
#     },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
}
le_recurred = LabelEncoder()
y_train = le_recurred.fit_transform(y_train)
y_test = le_recurred.transform(y_test)

#Dictionary to store the best models and their performances
best_models = {}
best_params = {}
performance_metrics = {}

# Loop through each model and perform GridSearchCV
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    best_models[model_name] = best_model
    best_params[model_name] = grid_search.best_params_
    performance_metrics[model_name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': conf_matrix,
        'Classification Report': class_report
    }

best_params, performance_metrics


# In[17]:


import pandas as pd
from IPython.display import display, HTML

# Create a DataFrame for best parameters
best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
best_params_df.index.name = 'Model'
best_params_df.columns = ['Parameter ' + str(i) for i in range(1, len(best_params_df.columns) + 1)]

# Display best parameters DataFrame
display(HTML(best_params_df.to_html()))

# Create a DataFrame for performance metrics
performance_metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')

# Transpose the DataFrame and rename columns for better readability
performance_metrics_df = performance_metrics_df.T.rename(
    columns={
        'Accuracy': 'Accuracy',
        'Confusion Matrix': 'Confusion Matrix',
        'Classification Report': 'Classification Report'
    }
)


# In[18]:


# prompt: compare all the models

# Compare model performance
model_names = list(performance_metrics.keys())
accuracies = [metrics['Accuracy'] for metrics in performance_metrics.values()]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies)
plt.title('Model Comparison - Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[29]:



# Print detailed performance metrics for each model
for model_name, metrics in performance_metrics.items():
  print(f"Model: {model_name}")
  print(f"Accuracy: {metrics['Accuracy']:.4f}")
  print("Confusion Matrix:")
  print(metrics['Confusion Matrix'])
  print("Classification Report:")
  print(metrics['Classification Report'])
  print("-" * 40)

# Find the best model based on accuracy
best_model_name = max(performance_metrics, key=lambda k: performance_metrics[k]['Accuracy'])
best_model_accuracy = performance_metrics[best_model_name]['Accuracy']


# In[28]:


print(f"Best Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")


# In[27]:



conclusion = """
Based on the comprehensive model comparison, the Decision Tree emerged as the best performing model with an impressive accuracy of 0.9870. This indicates its strong ability to accurately predict thyroid cancer recurrence based on the provided features.

While other models like Random Forest, Gradient Boosting, and SVM also demonstrated high accuracy, the Decision Tree's simplicity and interpretability make it a particularly attractive choice for this application. Its clear decision rules can provide valuable insights into the factors driving recurrence, aiding clinicians in making informed decisions.

Further investigation into feature importance and potential hyperparameter tuning could further enhance the Decision Tree's performance and solidify its role as a reliable tool for predicting thyroid cancer recurrence.
"""

print(conclusion)


# In[26]:


# Display performance metrics DataFrame
for model_name, metrics in performance_metrics.items():
    display(HTML(f'<h3>{model_name}</h3>'))
    for metric_name, metric_value in metrics.items():
        if metric_name == 'Confusion Matrix':
            cm_df = pd.DataFrame(metric_value, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
            display(HTML(cm_df.to_html()))
        else:
            display(HTML(f'<b>{metric_name}:</b> {metric_value}'))


# In[31]:


from sklearn.metrics import precision_score, recall_score, f1_score
# Create a list to store the results
results = []

# Evaluate each model on both training and testing sets
for model_name, model in best_models.items():
  # Training set evaluation
  y_train_pred = model.predict(X_train)
  train_accuracy = accuracy_score(y_train, y_train_pred)
  train_precision = precision_score(y_train, y_train_pred)
  train_recall = recall_score(y_train, y_train_pred)
  train_f1 = f1_score(y_train, y_train_pred)

  # Testing set evaluation
  y_test_pred = model.predict(X_test)
  test_accuracy = accuracy_score(y_test, y_test_pred)
  test_precision = precision_score(y_test, y_test_pred)
  test_recall = recall_score(y_test, y_test_pred)
  test_f1 = f1_score(y_test, y_test_pred)

  # Append results to the list
  results.append([model_name, train_accuracy, train_precision, train_recall, train_f1, 
                  test_accuracy, test_precision, test_recall, test_f1])

# Create a Pandas DataFrame from the results
results_df = pd.DataFrame(results, columns=['Model', 'Train Accuracy', 'Train Precision', 'Train Recall',
                                           'Train F1', 'Test Accuracy', 'Test Precision',
                                           'Test Recall', 'Test F1'])

# Display the results in a table format
print(results_df)

# Plotting the results
results_df.plot(x='Model', y=['Train Accuracy', 'Test Accuracy', 'Train Precision', 'Test Precision', 
                              'Train Recall', 'Test Recall', 'Train F1', 'Test F1'], kind='bar', figsize=(16, 8))
plt.title('Model Comparison - Key Features')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[32]:



# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy comparison plot
accuracy_df = performance_metrics_df.loc['Accuracy']  # Access 'Accuracy' row
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_df.index, y=accuracy_df.values)  # Use .values to get the accuracy values
plt.title('Accuracy Comparison of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[33]:



# Confusion matrix plots
for model_name, metrics in performance_metrics.items():
    cm = metrics['Confusion Matrix']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[34]:


# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Extract feature importances
rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_

# Create a DataFrame for better visualization
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'RandomForestImportance': rf_importances,
    'GradientBoostingImportance': gb_importances
})

# Sort the DataFrame by feature importance in descending order
importance_df = importance_df.sort_values(by='RandomForestImportance', ascending=False)

importace_df_display = importance_df.head(10)  # Displaying top 10 features for better clarity
importace_df_display


# In[35]:


# Plot feature importances for Random Forest
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.barh(importance_df['Feature'], importance_df['RandomForestImportance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importances')
plt.gca().invert_yaxis()

# Plot feature importances for Gradient Boosting
plt.subplot(1, 2, 2)
plt.barh(importance_df['Feature'], importance_df['GradientBoostingImportance'], color='lightgreen')
plt.xlabel('Importance')
plt.title('Gradient Boosting Feature Importances')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# Visualization of Key Features The bar charts above display the feature importances as determined by the Random Forest and Gradient Boosting models.
# 
# Random Forest Feature Importances: The feature "Response" is the most important, followed by "Risk", "N (Lymph node involvement)", "T (Tumor size)", and "Age".
# 
# Gradient Boosting Feature Importances: The feature "Response" is overwhelmingly the most important, with other features like "Risk", "Age", and "N (Lymph node involvement)" having comparatively lower importance.
# 
# These visualizations help in understanding which features have the most significant impact on predicting the recurrence of thyroid conditions.

# ### Limitations of these models
# 1.	Overfitting: Random Forest, Gradient Boosting, AdaBoost: These models are complex and can potentially overfit the training data, especially when hyperparameters are not properly tuned. This might lead to excellent performance on the training data but poorer generalization to new, unseen data. KNN: KNN can overfit if the number of neighbors (k) is too low.
# 2.	Model Interpretability: Random Forest, Gradient Boosting, AdaBoost: These ensemble models are often seen as "black boxes" because their decision-making processes are complex and not easily interpretable. Understanding the contribution of each feature to the final decision is challenging. SVM: Similarly, SVM with non-linear kernels (like RBF) is less interpretable.
# 3.	Training Time and Computational Resources:   Gradient Boosting: These models can be computationally intensive and time-consuming to train, particularly with large datasets and complex hyperparameter tuning. Random Forest: While generally faster than boosting algorithms, Random Forests can still be resource-intensive with many trees and large datasets.
# 4.	Imbalanced Data: Despite achieving high accuracy, the models might still struggle with imbalanced data where one class is underrepresented. This can lead to biased predictions towards the majority class. Example: If the dataset had a significantly higher number of non-recurrence cases than recurrence cases, the models might predict "No recurrence" more often, even when "Yes recurrence" is true.
# 5.	Hyperparameter Sensitivity: SVM: The performance of SVM highly depends on the choice of kernel and its parameters (e.g., C, gamma). Poorly chosen parameters can lead to suboptimal performance. KNN: The choice of the number of neighbors (k) can significantly affect the performance. Too few neighbors can lead to overfitting, while too many can lead to underfitting.
# 6.	Feature Engineering and Preprocessing: The performance of these models is highly dependent on the quality of the input features. If important features are missed or if irrelevant features are included, the model's performance can degrade. Proper handling of categorical variables, missing values, and feature scaling is crucial.
# 7.	Generalization to Different Datasets: The models' performance is specific to the dataset used for training and testing. They might not generalize well to different datasets with different characteristics, even if they are related to thyroid conditions.
# 

# ### Key Takeaways:
# 
# #### Overall Performance: 
# 
# Decision Tree and Random Forest show exceptional performance with near perfect accuracy on both training and test sets.
# 
# Gradient Boosting also performs well, but slightly lower than Decision Tree and Random Forest.
# 
# SVM, AdaBoost, and KNN have decent accuracy but lag behind the top performers.
# 
# #### Overfitting:
# 
# Decision Tree might be slightly overfitting the training data as its training accuracy is 1.0. This could indicate it's memorizing the training data rather than generalizing well.
# 
# Random Forest, due to its ensemble nature, generalizes better and shows less overfitting.
# 
# #### Best Model:
# 
# Based on the overall performance and generalization ability, Decision Tree appears to be the most suitable model for this task.
# It boasts high accuracy, precision, recall, and F1-score on both training and testing sets.
# 
# #### Further Analysis:
# 
# 
# Feature importance analysis can be conducted on the Random Forest model to identify the most influential features in predicting recurrence.
# 
# Hyperparameter tuning can be explored further to potentially improve the performance of the models, especially for those with lower accuracy. 
# 

# ## Results and Analysis:
# The analysis involved predicting the recurrence of thyroid conditions using various machine learning models. The models trained and evaluated include Decision Tree, Random Forest, SVM, Gradient Boosting, AdaBoost, and KNN. Among these models, the Decision Tree achieved the highest accuracy at 98.7%, followed closely by Random Forest model , Gradient Boosting, and AdaBoost with an accuracy of 97.4%.

# In[ ]:




