import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# --- Page Configuration ---
st.set_page_config(page_title="Viral Song Classifier", layout="wide")

st.title("Viral Song Classification Model Evaluation")
st.markdown("Upload your test dataset, select a trained machine learning model, and click predict to view evaluation metrics.")
st.divider()

# Load saved preprocessors with caching for better performance
@st.cache_resource
def load_preprocessors():
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
    return scaler, le

try:
    scaler, le = load_preprocessors()
except Exception as e:
    st.error("Error loading preprocessors. Ensure scaler.pkl and label_encoder.pkl are in the 'model' directory.")
    st.stop()

# --- 1. Dataset Upload ---
st.subheader("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    
    # Hide the raw data behind an expander to keep the UI clean
    with st.expander("View Uploaded Data Preview"):
        st.dataframe(test_data.head())
        
    if 'viral' not in test_data.columns:
        st.error("The uploaded dataset must contain the target column 'viral'.")
    else:
        # Preprocess the data
        columns_to_drop = ['track_id', 'track_name', 'artist']
        for col in columns_to_drop:
            if col in test_data.columns:
                test_data = test_data.drop(col, axis=1)
                
        X_test = test_data.drop('viral', axis=1)
        y_test = test_data['viral']
        
        if 'genre' in X_test.columns:
            X_test['genre'] = le.transform(X_test['genre'].astype(str))
            
        X_test_scaled = scaler.transform(X_test)
        
        st.divider()
        
        # --- 2. Model Selection & Prediction ---
        st.subheader("2. Select Model & Evaluate")
        model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
        
        # Place the dropdown and button side-by-side using columns
        col_select, col_button = st.columns([3, 1])
        
        with col_select:
            selected_model_name = st.selectbox("Choose a Classification Model:", model_names, label_visibility="collapsed")
            
        with col_button:
            # The execution halts here until the user clicks the button
            predict_clicked = st.button("Predict Performance", type="primary", use_container_width=True)
            
        if predict_clicked:
            file_name = selected_model_name.replace(" ", "_").lower() + ".pkl"
            
            try:
                model = joblib.load(f'model/{file_name}')
                
                # Generate Predictions
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                st.divider()
                st.subheader(f" Evaluation Results: {selected_model_name}")
                
                # Display metrics in a clean 6-column dashboard row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                m2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
                m3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
                m4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                m5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
                m6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
                
                st.write("") # Add some vertical spacing
                
                # Display Classification Report and Confusion Matrix side-by-side
                report_col, cm_col = st.columns(2)
                
                with report_col:
                    st.markdown("**Classification Report**")
                    # st.code formats the text as an easy-to-read monospaced block
                    st.code(classification_report(y_test, y_pred), language="text")
                    
                with cm_col:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    st.pyplot(fig)
                    
            except FileNotFoundError:
                st.error(f"Model file model/{file_name} not found. Please train and save the model first.")