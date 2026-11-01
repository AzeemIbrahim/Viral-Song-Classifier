# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# 2. Load Data
df = pd.read_csv('viral_songs.csv')

# 3. Preprocessing
# Drop non-predictive identifiers
df = df.drop(['track_id', 'track_name', 'artist'], axis=1)

# Encode categorical feature 'genre'
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])

# Separate features and target
X = df.drop('viral', axis=1)
y = df['viral']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test dataset for Streamlit upload (with original genre names for clarity)
test_df = X_test.copy()
test_df['genre'] = le.inverse_transform(test_df['genre'])
test_df['viral'] = y_test.values
test_df.to_csv('test_data.csv', index=False)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and encoder for deployment
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

# 6. Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Metrics[cite: 3]
    res = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    results.append(res)
    
    # Save the trained model
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')

# 7. Results Comparison
results_df = pd.DataFrame(results)
print(results_df)