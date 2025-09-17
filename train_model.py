
# train_model.py
# Re-trains a Decision Tree on the Fraud.csv and saves fraudmodel.pkl.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'Fraud.csv'
MODEL_PATH = BASE / 'fraudmodel.pkl'

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    # simple label encoding for 'type','nameOrig','nameDest'
    le_type = LabelEncoder(); le_type.fit(df['type'].astype(str))
    le_orig = LabelEncoder(); le_orig.fit(df['nameOrig'].astype(str))
    le_dest = LabelEncoder(); le_dest.fit(df['nameDest'].astype(str))
    df['type_le'] = le_type.transform(df['type'].astype(str))
    df['Orig_le'] = le_orig.transform(df['nameOrig'].astype(str))
    df['Dest_le'] = le_dest.transform(df['nameDest'].astype(str))
    X = df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','Orig_le','Dest_le','type_le']]
    y = df['isFraud']
    return X, y, (le_type, le_orig, le_dest)

def main():
    df = load_data()
    X, y, encs = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print('Train score:', model.score(X_train, y_train))
    print('Test score:', model.score(X_test, y_test))
    # Save model
    with open(MODEL_PATH,'wb') as f:
        pickle.dump(model, f)
    print('Saved model to', MODEL_PATH)

if __name__ == '__main__':
    main()
