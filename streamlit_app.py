import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

### Understanding the data set we will be working with

df = pd.read_csv("data/bank-additional.csv",sep = ';')

df.head()

df.info()

df.describe()

# From the results of ``describe()``, there might be need of scaling for some of the features. For example ``age`` has a mean of 40.11, ranging from 18 to 88 whereas ``euribor3m``has a mean of 3 and it ranges from 0.63 to 5.04. Data scaling is beneficial for models like __Logistic Regression__, __K-nearest Neighbours__ and __Neural Networks__.

print(df.isnull().sum())

# No missing values in the data set.

# Target variable distribution
df['y'].value_counts().plot(kind='bar', title='Subscription Distribution')
plt.show()

# Class Imbalance Visualization
df['y'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#E74C3C', '#2ECC71'])
plt.title('Class Distribution of Target Variable (y)')
plt.ylabel('')  # Hide the y-label
plt.show()

# We can see that there is a clear imbalancy in the target variable. We should handle this before training our model.

# List of numerical columns to plot
numerical_cols = ['age', 'campaign', 'pdays', 'previous']

# Plot histograms
df[numerical_cols].hist(bins=15, figsize=(15, 6), layout=(2, 3))
plt.tight_layout()
plt.show()

# Boxplots for numeric features by target variable
for col in ['age', 'campaign', 'pdays', 'previous']:
    df.boxplot(column=col, by='y', figsize=(8, 4))
    plt.title(f"{col} by Subscription")
    plt.suptitle('')  # That's to remove the default 'Boxplot grouped by y' title
    plt.ylabel(col)
    plt.show()

sns.pairplot(df, vars=['age', 'campaign', 'pdays', 'previous'], hue='y', palette='bwr')
plt.show()

# Heatmap for the numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f")
plt.show()

# Bar charts for key categorical features
key_categorical_cols = ['job', 'education', 'contact']
for col in key_categorical_cols:
    pd.crosstab(df[col], df['y']).plot(kind='bar', stacked=True, figsize=(8, 4))
    plt.title(f"Subscription by {col}")
    plt.ylabel('Count')
    plt.show()

## Do Feature Engineering

### Data Scaling:
# These are candidates for data scaling:
# - age
# - duration
# - campaign
# - pdays
# - previous
# - emp.var.rate
# - cons.price.idx
# - cons.conf.idx
# - euribor3m
# - nr.employed  
# Categorical columns (usually of object type) should not be scaled.  
# Here is a template for scaling:

from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame

# Select only the numerical columns for scaling
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
df_numerical = df[numerical_cols]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the numerical data and then transform it
df_scaled = scaler.fit_transform(df_numerical)

# Create a new DataFrame with the scaled data
# Note that this will result in a NumPy array, so we convert it back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=numerical_cols)

# If you want to add the scaled data back into the original DataFrame,
# you can drop the original columns and then concatenate the new scaled DataFrame
df = df.drop(columns=numerical_cols, axis=1)
df = pd.concat([df, df_scaled], axis=1)

# __Note that, you should perform scaling after splitting your data into training and testing sets to prevent data leakage.__  
# This means you should fit the StandardScaler on the training data and then use it to transform both the training and testing sets:

from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df.drop('y', axis=1)  # Assuming 'y' is the target column
y = df['y']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled = scaler.transform(X_test[numerical_cols])

# Overwrite the numerical columns in the original X_train and X_test DataFrames
X_train[numerical_cols] = X_train_scaled
X_test[numerical_cols] = X_test_scaled

### Template for Model Training

# Import necessary libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset (assuming you've loaded the data into a DataFrame df)
# df = pd.read_csv('path_to_your_data.csv', sep=';')

# Drop the 'duration' column
df = df.drop('duration', axis=1)

# Define the categorical columns that you will need to encode
categorical_cols = df.select_dtypes(include=['object']).drop(['y'], axis=1).columns.tolist()

# Define the numerical columns (already numeric in the dataset)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model (replace with your chosen model)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Remove the output target from the feature set
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})  # Encode the target variable

# Split the dataset into training and validation datasets
# from sklearn.model_selection import train_test_split
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model
# my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
# preds = my_pipeline.predict(X_valid)

# Evaluate the model
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y_valid, preds)
# print('Accuracy:', score)


# Exploratory Data Analysis

df.info()

df.duplicated().sum()

df.isnull().sum()

df.describe()

df['y'] = df['y'].map({'yes': 1, 'no': 0})

df["y"].value_counts()

df['y'].value_counts().plot(kind = 'bar', 
                                 figsize = (12, 5), 
                                 title = 'Distribution', 
                                 cmap = 'ocean')

# It seems like there is an imbalance between number of "yes" and "no"s. We should handle this imbalancy in the preprocessing step for our model to be successful.

df['age'].hist(bins=50)

df.boxplot(column='age', by='y')

df['job'].value_counts().plot(kind='bar')

pd.crosstab(df['job'], df['y']).plot(kind='bar', stacked=True)

# First, select only the numeric columns of the DataFrame
numeric_df = df.select_dtypes(include=[np.number])

# Now, you can safely calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Then, plot the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f")

corr_matrix

sns.pairplot(df, vars=['age', 'balance'], hue='y')

### Preprocessing

# 1- 

### Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv('bank-additional.csv', sep = ';')  # Replace with your dataset path
corr_matrix = df.corr()

# Assuming df is your DataFrame
# Loop through each column in the DataFrame
df = df.drop(['contact', 'month', 'day_of_week', 'duration'], axis=1)

# Handle missing values (if any)
# df.fillna(method='ffill', inplace=True)

# Assume df is your DataFrame and already loaded.

# Encode categorical variables
encoder = OneHotEncoder(sparse=False)
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
df_encoded.columns = encoder.get_feature_names_out()  # Updated for newer versions of scikit-learn

# Drop original categorical columns and concatenate the encoded ones
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, df_encoded], axis=1)

# Feature selection (excluding 'duration' for a realistic model)
features = df.drop(['y'], axis=1)
target = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Ensure the target is encoded as 0 and 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate and train the model with class_weight set to 'balanced'
model = LogisticRegression(max_iter=6000, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


#### Results:

### Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv('bank-additional.csv', sep = ';')  # Replace with your dataset path


# Assuming df is your DataFrame
# Loop through each column in the DataFrame
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype('category')

df = df.drop(['contact', 'month', 'day_of_week', 'duration'], axis=1)

# Encode categorical variables
encoder = OneHotEncoder(sparse=False)
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
df_encoded.columns = encoder.get_feature_names_out()  # Updated for newer versions of scikit-learn

# Drop original categorical columns and concatenate the encoded ones
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, df_encoded], axis=1)

# Feature selection (excluding 'duration' for a realistic model)
features = df.drop(['y'], axis=1)
target = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Ensure the target is encoded as 0 and 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate and train the Random Forest model
# Note: 'class_weight' is set to 'balanced' to handle the class imbalance
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
rf_predictions = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))


### Neural Networks

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv('bank-additional.csv', sep = ';')  # Replace with your dataset path


# Assuming df is your DataFrame
# Loop through each column in the DataFrame
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype('category')

df = df.drop(['contact', 'month', 'day_of_week', 'duration'], axis=1)


# Encode categorical variables
encoder = OneHotEncoder(sparse=False)
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
df_encoded.columns = encoder.get_feature_names_out()  # Updated for newer versions of scikit-learn

# Drop original categorical columns and concatenate the encoded ones
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, df_encoded], axis=1)

# Feature selection (excluding 'duration' for a realistic model)
features = df.drop(['y'], axis=1)
target = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Ensure the target is encoded as 0 and 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate and train the neural network model
# Here we'll just start with one hidden layer with 100 neurons
nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
nn_model.fit(X_train, y_train)

# Make predictions and evaluate the model
nn_predictions = nn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, nn_predictions))
print(classification_report(y_test, nn_predictions))


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the hyperparameters and their respective ranges to consider
# For logistic regression, 'C' is the inverse of regularization strength;
# smaller values specify stronger regularization.
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Example values
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Solvers to consider
    # You can add more hyperparameters to tune
}

# Create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=6000, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,  # Number of folds in cross-validation
    scoring='f1',  # Consider other scoring metrics as well
    verbose=1,  # For logging output
    n_jobs=-1  # Use all CPU cores
)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# The best hyperparameters from GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)

# Use the best model found to make predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Instantiate and train the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

df.head()