import pandas as pd
import re
import numpy as np
import os
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import optuna
pd.set_option('display.max_columns', None)
random_seed = 18
#directory_path = r"C:\Users\gohar\OneDrive - Georgia Institute of Technology\Project 7406"
#os.chdir(directory_path)
train = pd.read_csv(r"data\train.csv")
train2 = pd.read_csv(r"data\ObesityDataSet.csv")
test = pd.read_csv(r"data\test.csv")
print('shape: ',train.shape)
print(train.dtypes)
print(train.describe())

categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'MTRANS','CALC']
numerical_columns = [col for col in train.columns if col not in categorical_columns and col != 'id' and col != 'NObeyesdad']
numerical_columns.append('BMI')

subset = train[categorical_columns]
subset['NObeyesdad'] = train['NObeyesdad']
train['BMI'] = train['Weight'] / (train['Height'] ** 2)

# count of each value in NObeyesdad
plt.figure(figsize=(8, 6))
sns.countplot(x='NObeyesdad', data=train)
plt.title('Count of Each Value in NObeyesdad',fontsize=12, fontweight='bold')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom = .25, top = .93)
plt.show()

# count of each value in categorical columns
for column in categorical_columns:
    print()
    print(pd.crosstab(subset[column], columns='count'))

# boxplot for each numerical column grouped by NObeyesdad
for column in numerical_columns:
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='NObeyesdad', y=column, data=train)
    plt.title(f'Boxplot of {column} Stratified by NObeyesdad',fontsize=12, fontweight='bold')
    plt.show()

# KDE plots for each numerical column segregated by NObeyesdad
for column in numerical_columns:
    plt.figure()
    for category in train['NObeyesdad'].unique():
        subset = train[train['NObeyesdad'] == category]
        sns.kdeplot(data=subset, x=column, fill=True, label=category)
    plt.title(f'KDE Plot of {column} Stratified by NObeyesdad',fontsize=12, fontweight='bold')
    plt.legend(title='NObeyesdad', loc='upper right')
    plt.show()
train.drop(columns = 'BMI',inplace = True)

subset = train[categorical_columns]
subset['NObeyesdad'] = train['NObeyesdad']
# bar plot of count of each value in categorical columns by NObeyesdad
order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II','Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
melted = pd.melt(subset, id_vars=['NObeyesdad'], value_vars=categorical_columns)
count = melted.groupby(['variable', 'value', 'NObeyesdad']).size().reset_index(name='count')
for column in categorical_columns:
    plt.figure(figsize=(10, 10))
    sns.barplot(x='NObeyesdad', y='count', hue='value', data=count[count['variable'] == column],order=order)
    plt.title(f'Count of {column} Stratified by NObeyesdad', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title=column, loc='upper right')
    plt.show()

subset['MTRANS'] = subset['MTRANS'].replace('Public_Transportation', 'Public')
plt.figure()
sns.countplot(x='MTRANS', hue='Gender', data=subset)
plt.title('Count of MTRANS by Gender', fontsize=12, fontweight='bold')
plt.legend(title='Gender')
plt.show()

# convert categorical to binary representation.
train = pd.get_dummies(train, columns=categorical_columns)
train2 = pd.get_dummies(train2, columns=categorical_columns)
test = pd.get_dummies(test,columns=categorical_columns)
print(train.columns)


# correlation plot of all data
numerical_columns.remove('BMI')
correlation = train.drop(columns = ['id', 'NObeyesdad']).corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation, cmap='coolwarm',linewidths=.5)
plt.title('Correlation Plot',fontsize=24, fontweight='bold')
plt.subplots_adjust(left = .2,bottom = .25,right = 1.05, top = .93)
plt.show()

# correlation plot of just numerical
correlation = train[numerical_columns].corr()
plt.figure()
sns.heatmap(correlation, cmap='coolwarm',annot=True,fmt=".2f")
plt.title('Correlation Plot (Numeric Columns)',fontsize=10, fontweight='bold')
plt.show()



train.drop(columns = 'id', inplace = True)
index_CALC_Frequently = train.columns.get_loc('CALC_Frequently')
train.insert(index_CALC_Frequently, 'CALC_Always', 0)
encoder = LabelEncoder()



# combine original dataset and synthetic datset
#train = pd.concat([train, train2], axis=0, ignore_index=True) # comment this line to use just the competition/synthetic dataset for modeling
print(train.dtypes)
print(train.head())

train['NObeyesdad'] = encoder.fit_transform(train['NObeyesdad'])

#train = train.apply(encoder.fit_transform)
test = train2.copy()                                               #uncomment this line to test on original nonsynthetic dataset
test['NObeyesdad'] = encoder.transform(test['NObeyesdad'])         #uncomment this line to test on original nonsynthetic dataset

'''
#scale data
#####################
columns_to_scale = ['Age', 'Weight']
scaler = MinMaxScaler()
train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])
####################
'''
Ytrain = train['NObeyesdad']
Xtrain = train.drop(columns = 'NObeyesdad')
Xtest = test.drop(columns = 'NObeyesdad')                         #uncomment this line to test on original nonsynthetic dataset
Ytest = test['NObeyesdad']                                        #uncomment this line to test on original nonsynthetic dataset




# Logistic Regression
LR = LogisticRegression()
LR_scores = cross_val_score(LR, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Logistic Regression")
print(f"Average Accuracy: {LR_scores.mean():.2f}")
print()


# Random Forest
RF = RandomForestClassifier()
RF_scores = cross_val_score(RF, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Random Forest")
print(f"Average Accuracy: {RF_scores.mean():.2f}")
print()


# SVM
SVM = SVC()
SVM_scores = cross_val_score(SVM, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Support Vector Machine")
print(f"Average Accuracy: {SVM_scores.mean():.2f}")
print()

# Naïve Bayes (Multinomial)
nb_model_multinomial = MultinomialNB()
NB_multinomial_scores = cross_val_score(nb_model_multinomial, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Naïve Bayes (Multinomial)")
print(f"Average Accuracy: {NB_multinomial_scores.mean():.2f}")
print()

# Naïve Bayes (Gaussian)
nb_model_gaussian = GaussianNB()
NB_gaussian_scores = cross_val_score(nb_model_gaussian, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Naïve Bayes (Gaussian)")
print(f"Average Accuracy: {NB_gaussian_scores.mean():.2f}")
print()

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=18)
DT_scores = cross_val_score(dt_model, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("Decision Tree")
print(f"Average Accuracy: {DT_scores.mean():.2f}")
print()


# KNN
optimal_knn_model = KNeighborsClassifier(n_neighbors=4)
KNN_scores = cross_val_score(optimal_knn_model, Xtrain, Ytrain, cv=5, scoring='accuracy')
print(f"K-Nearest Neighbors (Optimal k={4})")
print(f"Average Accuracy: {KNN_scores.mean():.2f}")
print()

# XGBoost
xgb_model = XGBClassifier(random_state=18)
XGB_scores = cross_val_score(xgb_model, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("XGBoost Classifier")
print(f"Average Accuracy: {XGB_scores.mean():.2f}")
print()


'''
########################## Testing models for submission ######################

# XGBoost
params = {'booster': 'dart', 'learning_rate': 0.08723360512826742, 'max_depth': 10, 'subsample': 0.7999669669900065, 'colsample_bytree': 0.5413275309396097, 'min_child_weight': 3}
xgb_model = XGBClassifier(random_state=18,**params)
xgb_model.fit(Xtrain, Ytrain)
Ypred = xgb_model.predict(test.drop(columns = 'id'))
predicted = encoder.inverse_transform(Ypred)

predictions_df = pd.DataFrame({
    'id': test['id'],
    'NObeyesdad': predicted
})
print(predictions_df)
#predictions_df.to_csv('Predictions.csv', index=False)


# Random Forest
RF = RandomForestClassifier()
RF.fit(Xtrain, Ytrain)
Ypred = RF.predict(test.drop(columns = 'id'))
predicted = encoder.inverse_transform(Ypred)
predictions_df = pd.DataFrame({
    'id': test['id'],
    'NObeyesdad': predicted
})
print(predictions_df)
#predictions_df.to_csv('Predictions.csv', index=False)
'''



############ non cross validation modeling with extra accuracy information ###############
# Logistic Regression
LR = LogisticRegression()
LR.fit(Xtrain, Ytrain)
predict_LR = LR.predict(Xtest)
# Access coefficients for Logistic Regression
feature_names = Xtrain.columns
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': LR.coef_[0]
})
print("Coefficients Logistic Regression:")
print(coefficients_df)

# Random Forest
RF = RandomForestClassifier()
RF.fit(Xtrain, Ytrain)
predict_RF = RF.predict(Xtest)
# Access feature importances for Random Forest
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': RF.feature_importances_
})
print("Feature Importances Random Forest:")
print(importances_df)

# SVM
SVM = SVC()
SVM.fit(Xtrain, Ytrain)
predict_SVM = SVM.predict(Xtest)

# Naïve Bayes (Multinomial)
nb_model = MultinomialNB()
nb_model.fit(Xtrain, Ytrain)
predict_NBM = nb_model.predict(Xtest)

# Naïve Bayes (Gaussian)
nb_model = GaussianNB()
nb_model.fit(Xtrain, Ytrain)
predict_NBG = nb_model.predict(Xtest)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(Xtrain, Ytrain)
predict_DT = dt_model.predict(Xtest)
# Access feature importances for Decision Tree
importances_dt_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt_model.feature_importances_
})
print("Feature Importances Decision Tree:")
print(importances_dt_df)

# XGBoost
xgb_model = XGBClassifier(random_state=18)
xgb_model.fit(Xtrain, Ytrain)
predict_XGB = xgb_model.predict(Xtest)
# Access feature importances for XGBoost
importances_xgb_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
})
print("Feature Importances XGBoost:")
print(importances_xgb_df)


# KNN
optimal_knn_model = KNeighborsClassifier(n_neighbors=4)
optimal_knn_model.fit(Xtrain, Ytrain)
predict_KNN = optimal_knn_model.predict(Xtest)
'''
k_values = np.arange(1, 21) 
cv_scores = []
# perform cross-validation for each k
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, Xtrain, Ytrain, cv=5, scoring='accuracy')  
    cv_scores.append(scores.mean())
optimal_k = k_values[np.argmax(cv_scores)]
print(f"The optimal value for k is: {optimal_k}")
'''



# evaluate models
def evaluate_model(Ytrue, Ypred, model_name):
    accuracy = accuracy_score(Ytrue, Ypred)
    confusion_mat = confusion_matrix(Ytrue, Ypred)
    classification_rep = classification_report(Ytrue, Ypred)
    print(f"{model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Classification Report:")
    print(classification_rep)
    print("\n")

# evaluate models
evaluate_model(Ytest, predict_LR, "Logistic Regression")
evaluate_model(Ytest, predict_RF, "Random Forest")
evaluate_model(Ytest, predict_SVM, "Support Vector Machine")
evaluate_model(Ytest, predict_NBM, "Naïve Bayes (Multinomial)")
evaluate_model(Ytest, predict_NBG, "Naïve Bayes (Gaussian)")
evaluate_model(Ytest, predict_DT, "Decision Tree")
evaluate_model(Ytest, predict_KNN, f"K-Nearest Neighbors (Optimal k={4})")
evaluate_model(Ytest, predict_XGB, "XGBoost Classifier")


'''
########## hyperparameter tuning for XGBoost ###########    # references https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }


X_train, X_test, y_train, y_test = train_test_split(train.drop(columns = 'NObeyesdad'), train['NObeyesdad'], test_size = 0.2, random_state = 0)

def objective(space):
    clf = XGBClassifier(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']),eval_metric="auc",early_stopping_rounds=10)

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation,
             verbose=False)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()


best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)
'''




'''
def objective(trial):
    params = {
        'objective': 'multi:softmax',
        'num_class': len(y_train.unique()),  
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1.0 - accuracy  # Optuna minimizes the objective function


# Create study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Access best hyperparameters found during optimization
best_params = study.best_params
print("Best Hyperparameters:", best_params)

best_xgb_model = XGBClassifier(**best_params)
XGB_scores = cross_val_score(best_xgb_model, Xtrain, Ytrain, cv=5, scoring='accuracy')
print("XGBoost Classifier")
print(f"Average Accuracy: {XGB_scores.mean():.2f}")
'''