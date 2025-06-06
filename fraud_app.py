import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Agentic Fraud Pattern Detector")

# --- 1. Synthetic Data Generation (Copied from previous script) ---
@st.cache_data # Cache data generation for faster re-runs
def generate_synthetic_fraud_data(num_samples=2000):
    np.random.seed(42)
    random.seed(42)

    data = []
    # Define common fraud types and their typical characteristics
    fraud_types = ['Legitimate', 'Account Takeover', 'Card Testing', 'Friendly Fraud', 'Phishing-Related']
    
    # Feature definitions (simplified for demonstration)
    features = [
        'transaction_amount',
        'num_failed_attempts',
        'new_device_used',
        'location_mismatch',
        'time_since_last_txn_minutes',
        'chargeback_initiated_days_after_txn',
        'unusual_login_pattern',
        'multiple_small_txns_rapid',
        'suspicious_email_domain'
    ]

    # Generate data for each type
    for _ in range(int(num_samples * 0.70)): # 70% legitimate transactions
        data.append({
            'transaction_amount': np.random.normal(50, 20),
            'num_failed_attempts': 0,
            'new_device_used': 0,
            'location_mismatch': 0,
            'time_since_last_txn_minutes': np.random.randint(60, 1440),
            'chargeback_initiated_days_after_txn': 0,
            'unusual_login_pattern': 0,
            'multiple_small_txns_rapid': 0,
            'suspicious_email_domain': 0,
            'fraud_type': 'Legitimate'
        })

    for _ in range(int(num_samples * 0.10)): # 10% Account Takeover
        data.append({
            'transaction_amount': np.random.normal(500, 150),
            'num_failed_attempts': random.choice([0, 1, 2]),
            'new_device_used': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'location_mismatch': random.choices([0, 1], weights=[0.2, 0.8])[0],
            'time_since_last_txn_minutes': np.random.randint(1000, 10000),
            'chargeback_initiated_days_after_txn': 0,
            'unusual_login_pattern': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'multiple_small_txns_rapid': 0,
            'suspicious_email_domain': 0,
            'fraud_type': 'Account Takeover'
        })

    for _ in range(int(num_samples * 0.08)): # 8% Card Testing
        data.append({
            'transaction_amount': np.random.normal(5, 2),
            'num_failed_attempts': random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.2, 0.2])[0],
            'new_device_used': random.choices([0, 1], weights=[0.5, 0.5])[0],
            'location_mismatch': 0,
            'time_since_last_txn_minutes': np.random.randint(1, 10),
            'chargeback_initiated_days_after_txn': 0,
            'unusual_login_pattern': 0,
            'multiple_small_txns_rapid': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'suspicious_email_domain': 0,
            'fraud_type': 'Card Testing'
        })

    for _ in range(int(num_samples * 0.07)): # 7% Friendly Fraud
        data.append({
            'transaction_amount': np.random.normal(100, 50),
            'num_failed_attempts': 0,
            'new_device_used': 0,
            'location_mismatch': 0,
            'time_since_last_txn_minutes': np.random.randint(60, 1440),
            'chargeback_initiated_days_after_txn': np.random.randint(5, 60),
            'unusual_login_pattern': 0,
            'multiple_small_txns_rapid': 0,
            'suspicious_email_domain': 0,
            'fraud_type': 'Friendly Fraud'
        })
        
    for _ in range(int(num_samples * 0.05)): # 5% Phishing-Related
        data.append({
            'transaction_amount': np.random.normal(300, 100),
            'num_failed_attempts': random.choice([0, 1]),
            'new_device_used': random.choices([0, 1], weights=[0.3, 0.7])[0],
            'location_mismatch': random.choices([0, 1], weights=[0.3, 0.7])[0],
            'time_since_last_txn_minutes': np.random.randint(60, 500),
            'chargeback_initiated_days_after_txn': 0,
            'unusual_login_pattern': random.choices([0, 1], weights=[0.2, 0.8])[0],
            'multiple_small_txns_rapid': 0,
            'suspicious_email_domain': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'fraud_type': 'Phishing-Related'
        })

    df = pd.DataFrame(data)
    # Ensure numerical features are positive
    for col in ['transaction_amount', 'num_failed_attempts', 'time_since_last_txn_minutes', 'chargeback_initiated_days_after_txn']:
        df[col] = df[col].apply(lambda x: max(0, x))
    return df

# --- 2. LLM Simulation Function (Copied from previous script) ---
def simulate_llm_explanation(transaction_details, shap_values, feature_names, predicted_type):
    explanation_text = f"**Predicted Fraud Type:** {predicted_type}\n\n"
    explanation_text += "**Why this transaction was flagged as potential fraud of this type:**\n"

    # Get top contributing features from SHAP values
    sorted_shap_indices = np.argsort(np.abs(shap_values))[::-1]
    top_features_shap = [(feature_names[i], shap_values[i]) for i in sorted_shap_indices if np.abs(shap_values[i]) > 0.01][:5]

    if not top_features_shap:
        explanation_text += "    No significant features identified by SHAP for this prediction.\n"
    else:
        for feature, shap_val in top_features_shap:
            contribution = "increased" if shap_val > 0 else "decreased"
            explanation_text += f"    - The `'{feature}'` (value: {transaction_details[feature]:.2f}) significantly **{contribution}** the likelihood of this being `'{predicted_type}'` fraud. (SHAP: {shap_val:.3f})\n"

    explanation_text += "\n**Potential Fraud Pattern Insights:**\n"
    if predicted_type == 'Account Takeover':
        explanation_text += "    This pattern suggests an unauthorized individual gained access to the user's account. The use of a new device/location and an unusual login pattern are strong indicators.\n"
    elif predicted_type == 'Card Testing':
        explanation_text += "    This pattern is typical of fraudsters validating stolen card numbers by making many small, rapid transactions. High failed attempt rates are common.\n"
    elif predicted_text == 'Friendly Fraud':
        explanation_text += "    This indicates a legitimate transaction that was later disputed by the cardholder. This can sometimes be a misunderstanding or an intentional false claim.\n"
    elif predicted_type == 'Phishing-Related':
        explanation_text += "    This pattern often involves credentials or information obtained through phishing attacks, leading to transactions from new devices/locations or suspicious email domains.\n"
    else:
        explanation_text += "    This is a legitimate transaction with no apparent fraud pattern.\n"

    explanation_text += "\n**Suggested Immediate Actions:**\n"
    if predicted_type == 'Account Takeover':
        explanation_text += "    1. Immediately block the transaction and freeze the account.\n"
        explanation_text += "    2. Contact the legitimate account holder via a pre-registered, verified channel (e.g., phone number on file, not email).\n"
        explanation_text += "    3. Initiate a password reset and require multi-factor authentication for account recovery.\n"
    elif predicted_type == 'Card Testing':
        explanation_text += "    1. Block the transaction and the associated card/IP if multiple rapid attempts are observed.\n"
        explanation_text += "    2. Implement stricter velocity rules for small transactions, especially from new IPs.\n"
    elif predicted_type == 'Friendly Fraud':
        explanation_text += "    1. Review transaction history and user interaction logs for evidence of service delivery.\n"
        explanation_text += "    2. Attempt to contact the customer to resolve the dispute before accepting the chargeback.\n"
    elif predicted_type == 'Phishing-Related':
        explanation_text += "    1. Block the transaction and flag the account for review.\n"
        explanation_text += "    2. Alert the user about potential phishing attempts and advise on security best practices.\n"
    else:
        explanation_text += "    No immediate action required. Monitor for future suspicious activity.\n"
    return explanation_text


# --- Streamlit App ---
st.title("🛡️ Agentic Fraud Pattern Detector")
st.markdown("Leveraging Machine Learning and Explainable AI (SHAP) with LLM insights to identify and explain fraud patterns.")

# --- Model Training and Overview ---
st.header("Overall Fraud Landscape Analysis")
st.write("First, let's train our model on synthetic data and see the general distribution of detected fraud patterns.")

# Cache the model training for faster re-runs
@st.cache_resource
def train_model(df_train, y_train_data):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(df_train, y_train_data)
    return model

# Generate data and train model
df = generate_synthetic_fraud_data()
X = df.drop('fraud_type', axis=1)
y = df['fraud_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = train_model(X_train, y_train)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance (Test Set)")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
    st.write("*(Note: Performance on synthetic data is illustrative.)*")

with col2:
    st.subheader("Predicted Fraud Type Distribution")
    predicted_fraud_types = pd.Series(y_pred)
    fraud_type_counts = predicted_fraud_types[predicted_fraud_types != 'Legitimate'].value_counts()
    
    if not fraud_type_counts.empty:
        total_fraud_predictions = fraud_type_counts.sum()
        fraud_type_percentages = (fraud_type_counts / total_fraud_predictions) * 100
        
        # Create a DataFrame for Plotly Pie Chart
        plot_df = pd.DataFrame({'Fraud Type': fraud_type_percentages.index, 'Percentage': fraud_type_percentages.values})
        
        fig = px.pie(plot_df, values='Percentage', names='Fraud Type', title='Distribution of Predicted Fraud Types',
                     hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraudulent transactions predicted in the test set.")


st.markdown("---")

# --- Real-time Transaction Analysis ---
st.header("Analyze a New Transaction")
st.write("Enter details for a new transaction to get an instant fraud type prediction and an AI-driven explanation.")

# Input fields for new transaction
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)
    num_failed_attempts = st.slider("Number of Failed Attempts", 0, 10, 0)
    new_device_used = st.checkbox("New Device Used?", False)

with input_col2:
    location_mismatch = st.checkbox("Location Mismatch?", False)
    time_since_last_txn_minutes = st.number_input("Time Since Last Transaction (minutes)", min_value=1, value=120)
    chargeback_initiated_days_after_txn = st.number_input("Chargeback Initiated Days After Txn", min_value=0, value=0)

with input_col3:
    unusual_login_pattern = st.checkbox("Unusual Login Pattern?", False)
    multiple_small_txns_rapid = st.checkbox("Multiple Small Rapid Transactions?", False)
    suspicious_email_domain = st.checkbox("Suspicious Email Domain?", False)

# Create a DataFrame for the new transaction
new_transaction_data = pd.DataFrame([{
    'transaction_amount': transaction_amount,
    'num_failed_attempts': num_failed_attempts,
    'new_device_used': 1 if new_device_used else 0,
    'location_mismatch': 1 if location_mismatch else 0,
    'time_since_last_txn_minutes': time_since_last_txn_minutes,
    'chargeback_initiated_days_after_txn': chargeback_initiated_days_after_txn,
    'unusual_login_pattern': 1 if unusual_login_pattern else 0,
    'multiple_small_txns_rapid': 1 if multiple_small_txns_rapid else 0,
    'suspicious_email_domain': 1 if suspicious_email_domain else 0
}])

if st.button("Analyze Transaction"):
    st.subheader("Analysis Results:")
    
    # Predict fraud type
    predicted_type_single = model.predict(new_transaction_data)[0]
    st.success(f"**Predicted Fraud Type:** {predicted_type_single}")

    if predicted_type_single != 'Legitimate':
        st.write("Our system has flagged this transaction as potentially fraudulent. Here's why:")

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values_instance = explainer.shap_values(new_transaction_data)
        
        # Get SHAP values for the predicted class
        predicted_class_idx = np.where(model.classes_ == predicted_type_single)[0][0]
        shap_values_for_predicted_class = shap_values_instance[predicted_class_idx]

        st.subheader("Feature Contributions (SHAP Values)")
        st.write("The chart below shows how each feature contributed to the prediction of this specific fraud type.")
        
        # Create SHAP force plot (requires JS, so using a workaround for Streamlit)
        # For a simple display, we'll plot a bar chart of SHAP values
        shap_df = pd.DataFrame({
            'Feature': X.columns,
            'SHAP Value': shap_values_for_predicted_class
        }).sort_values(by='SHAP Value', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x > 0 else 'blue' for x in shap_df['SHAP Value']]
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        ax.set_title(f"SHAP Values for Predicted '{predicted_type_single}' Fraud")
        plt.tight_layout()
        st.pyplot(fig)

        # Simulated LLM Explanation
        st.subheader("Agentic AI Explanation & Actions")
        llm_explanation = simulate_llm_explanation(new_transaction_data.iloc[0], 
                                                     shap_values_for_predicted_class, 
                                                     X.columns, 
                                                     predicted_type_single)
        st.markdown(llm_explanation)
    else:
        st.info("This transaction is predicted as **Legitimate**. No further action required.")

