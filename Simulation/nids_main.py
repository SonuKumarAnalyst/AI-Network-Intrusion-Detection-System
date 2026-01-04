import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

# Title
st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic in real-time.

It classifies traffic into:
- **Benign:** Safe traffic  
- **Malicious:** Cyber attacks (DDoS, Port Scan, etc.)
""")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000

    data = {
        'Destination_Port': np.random.randint(1, 65535, n_samples),
        'Flow_Duration': np.random.randint(100, 100000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Packet_Length_Mean': np.random.uniform(10, 1500, n_samples),
        'Active_Mean': np.random.uniform(0, 1000, n_samples),
        'Label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Add attack patterns
    attack_idx = df['Label'] == 1
    df.loc[attack_idx, 'Total_Fwd_Packets'] += np.random.randint(
        50, 200, size=attack_idx.sum()
    )
    df.loc[attack_idx, 'Flow_Duration'] = np.random.randint(
        1, 1000, size=attack_idx.sum()
    )

    return df


df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Control Panel")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

# --- PREPROCESSING ---
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - split_size) / 100, random_state=42
)

# --- MODEL TRAINING ---
st.divider()
col_train, col_metrics = st.columns([1, 2])

with col_train:
    st.subheader("1. Model Training")

    if st.button("Train Model Now"):
        with st.spinner("Training Random Forest Model..."):
            model = RandomForestClassifier(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            st.session_state["model"] = model
            st.success("Training Complete!")

    if "model" in st.session_state:
        st.info("Model is ready for testing")

# --- METRICS ---
with col_metrics:
    st.subheader("2. Performance Metrics")

    if "model" in st.session_state:
        model = st.session_state["model"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc*100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Detected Threats", int(np.sum(y_pred)))

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please train the model first")

# --- LIVE ATTACK SIMULATOR ---
st.divider()
st.subheader("3. Live Traffic Simulator")

c1, c2, c3, c4 = st.columns(4)
p_dur = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
p_pkts = c2.number_input("Total Packets", 0, 500, 100)
p_len = c3.number_input("Packet Length Mean", 0, 1500, 500)
p_active = c4.number_input("Active Mean Time", 0, 1000, 50)

if st.button("Analyze Packet"):
    if "model" in st.session_state:
        model = st.session_state["model"]

        input_data = np.array([[80, p_dur, p_pkts, p_len, p_active]])
        pred = model.predict(input_data)

        if pred[0] == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED")
        else:
            st.success("✅ BENIGN TRAFFIC (Safe)")
    else:
        st.error("Please train the model first")
