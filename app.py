
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Paths (adjust if you place files elsewhere)
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'fraudmodel.pkl'
DATA_PATH = BASE / 'Fraud.csv'  # optional - included for encoder fitting

st.set_page_config(page_title='Transaction Fraud Detection', layout='wide')

@st.cache(allow_output_mutation=True)
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH,'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error('Model file fraudmodel.pkl not found in app folder.')
        return None

@st.cache(allow_output_mutation=True)
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return df
    return None

@st.cache(allow_output_mutation=True)
def build_encoders(df):
    # Fit LabelEncoders on columns seen during training: 'type','nameOrig','nameDest'
    encoders = {}
    for col in ['type','nameOrig','nameDest']:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders

# 

def encode_value(col, val, encoders):
    le = encoders.get(col)
    if le is None:
        return 0
    val = str(val)
    if val in le.classes_:
        return int(np.where(le.classes_==val)[0][0])
    else:
        # unseen -> map to new index at end (safe fallback)
        return int(len(le.classes_))

def preprocess_row(row, encoders, feature_order):
    # row is dict-like or Series with raw columns
    # Build features in the order model expects: 
    # ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','Orig_le','Dest_le','type_le']
    feat = {}
    feat['step'] = float(row.get('step',0))
    feat['amount'] = float(row.get('amount',0))
    feat['oldbalanceOrg'] = float(row.get('oldbalanceOrg',0))
    feat['newbalanceOrig'] = float(row.get('newbalanceOrig',0))
    feat['oldbalanceDest'] = float(row.get('oldbalanceDest',0))
    feat['newbalanceDest'] = float(row.get('newbalanceDest',0))
    feat['Orig_le'] = encode_value('nameOrig', row.get('nameOrig',''), encoders)
    feat['Dest_le'] = encode_value('nameDest', row.get('nameDest',''), encoders)
    feat['type_le'] = encode_value('type', row.get('type',''), encoders)
    return pd.DataFrame([feat])[feature_order]

def predict_df(df_input, model, encoders, feature_order):
    rows = []
    for _, r in df_input.iterrows():
        pr = preprocess_row(r, encoders, feature_order)
        rows.append(pr.iloc[0].to_list())
    X = pd.DataFrame(rows, columns=feature_order)
    # predict_proba if available
    if hasattr(model,'predict_proba'):
        probs = model.predict_proba(X)[:,1]
        preds = (probs >= 0.5).astype(int)
        return pd.DataFrame({'prediction':preds, 'fraud_prob':probs})
    else:
        preds = model.predict(X)
        return pd.DataFrame({'prediction':preds})

def main():
    st.title('🚨 Transaction Fraud Detection — Real-time UI')
    st.markdown('Upload transactions (CSV/XLSX) or enter a single transaction to get real-time fraud predictions.')

    model = load_model()
    data = load_data()
    if data is not None:
        encoders = build_encoders(data)
    else:
        encoders = None

    feature_order = None
    if model is not None and hasattr(model,'feature_names_in_'):
        feature_order = list(model.feature_names_in_)
    else:
        # fallback to known order
        feature_order = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','Orig_le','Dest_le','type_le']

    col1, col2 = st.columns([2,1])

    with col1:
        st.header('Batch Prediction (file upload)')
        uploaded = st.file_uploader('Upload CSV or Excel containing transactions (columns: step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest)', type=['csv','xlsx','xls'])
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
                st.write('Preview of uploaded file:', df_up.head())
                if model is not None and encoders is not None:
                    results = predict_df(df_up, model, encoders, feature_order)
                    out = pd.concat([df_up.reset_index(drop=True), results], axis=1)
                    st.write('Predictions (first 10):', out.head(10))
                    st.markdown('Download results:')
                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV with predictions', data=csv, file_name='predictions.csv', mime='text/csv')
                else:
                    st.warning('Model or encoders not available.')
            except Exception as e:    # The model defines expected feature names order in attribute feature_names_in_

                st.error(f'Failed to read or process file: {e}')

    with col2:
        st.header('Single Transaction Prediction')
        with st.form('single_form'):
            step = st.number_input('Step (time unit)', min_value=0, value=1)
            ttype = st.selectbox('Transaction type', options=list(data['type'].unique()) if data is not None else ['PAYMENT','TRANSFER','CASH_OUT','DEBIT','WITHDRAW'])
            amount = st.number_input('Amount', min_value=0.0, value=100.0, format="%.2f")
            nameOrig = st.text_input('Origin account id (nameOrig)', value='C123456789')
            oldbalanceOrg = st.number_input('Old balance origin', min_value=0.0, value=0.0, format="%.2f")
            newbalanceOrig = st.number_input('New balance origin', min_value=0.0, value=0.0, format="%.2f")
            nameDest = st.text_input('Destination account id (nameDest)', value='M987654321')
            oldbalanceDest = st.number_input('Old balance dest', min_value=0.0, value=0.0, format="%.2f")
            newbalanceDest = st.number_input('New balance dest', min_value=0.0, value=0.0, format="%.2f")
            submitted = st.form_submit_button('Predict')

        if submitted:
            raw = {'step':step,'type':ttype,'amount':amount,'nameOrig':nameOrig,
                   'oldbalanceOrg':oldbalanceOrg,'newbalanceOrig':newbalanceOrig,
                   'nameDest':nameDest,'oldbalanceDest':oldbalanceDest,'newbalanceDest':newbalanceDest}
            if model is not None and encoders is not None:
                X = preprocess_row(raw, encoders, feature_order)
                if hasattr(model,'predict_proba'):
                    prob = model.predict_proba(X)[:,1][0]
                    pred = int(prob >= 0.5)
                    st.metric('Fraud Probability', f'{prob:.3f}', delta=None)
                    st.success('⚠️ Fraud' if pred==1 else '✅ Not Fraud')
                else:
                    pred = model.predict(X)[0]
                    st.write('Prediction:', pred)
            else:
                st.warning('Model or encoders not available to make prediction.')

    st.markdown('---')
    st.subheader('About & Flow')
    st.markdown("""
    **How it works (end-to-end)**
    1. A dataset (Fraud.csv) was used to train a Decision Tree classifier. The model expects numeric features:
       - step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
       - plus encoded categorical features: Orig_le, Dest_le, type_le
    2. The app fits label encoders on the included dataset so that new inputs are encoded the same way as training.
    3. For each incoming transaction (single or batch), the app:
       - encodes categorical fields (type, nameOrig, nameDest)
       - constructs the features in the model's expected order
       - calls model.predict_proba or model.predict to generate a fraud score/prediction
    4. The app shows results in real-time and allows downloading predicted batch results.
    """)

if __name__ == '__main__':
    main()
