#!/usr/bin/env python
# coding: utf-8


#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler




df = pd.read_csv("Cleaned_data.csv")
label_encoder = LabelEncoder()

df['label'] = label_encoder.fit_transform(df['label'])

df['label'] = df['label'].apply(lambda x: 0 if x == label_encoder.transform(['white'])[0] else 1)
X=df.drop("label",axis=1)
y=(df["label"]>0).astype('int')
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                   random_state=42,
                                   test_size=0.2,
                                   shuffle=True)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)




from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline




xgb_classifier = XGBClassifier()
xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
xgb_grid = GridSearchCV(xgb_classifier, xgb_params, cv=5)
xgb_grid.fit(X_train, y_train)
xgb_pred = xgb_grid.predict(X_test)




# Evaluate XGBoost
print("XGBoost:")
print(f'Accuracy: {accuracy_score(y_test, xgb_pred)}')
print(f'Classification Report:\n{classification_report(y_test, xgb_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, xgb_pred)}')



#pip install joblib



from xgboost import XGBClassifier
import joblib

joblib.dump(xgb_grid, 'xgboost_model.pkl')