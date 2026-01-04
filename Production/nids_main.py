import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")

st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze real-world
network traffic from the **CIC-IDS2017 dataset**.

It classifies traffic into:
- **Benign:** Normal traffic  
- **Malicious:** Cyber attacks (DDoS, Port Scan, etc.)
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Control Panel")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    ]
)

split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

# ---------------- DATA LOADING (FIXED) ----------------
@st.cache_data
def load_data(file_name):
    df = pd.read_csv(file_name)

    # FIX: Remove hidden spaces from column names
    df.columns = df.columns.str.strip()

    df = df.dropna()

    df = df.rename(columns={
        'Destination Port': 'Destination_Port',
        'Flow Duration': 'Flow_Duration',
        'Total Fwd Packets': 'Total_Fwd_Packets',
        'Packet Length Mean': 'Packet_Length_Mean',
        'Active Mean': 'Active_Mean',
        'Label': 'Label'
    })

    # Convert labels to binary
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Select only required features
    df = df[
        [
            'Destination_Port',
            'Flow_Duration',
            'Total_Fwd_Packets',
            'Packet_Length_Mean',
            'Active_Mean',
            'Label'
        ]
    ]

    return df


df = load_data(dataset_name)

# ---------------- PREPROCESSING ----------------
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(100 - split_size) / 100,
    random_state=42
)

# ---------------- MODEL TRAINING ----------------
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

# ---------------- METRICS ----------------
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

# ---------------- LIVE ATTACK SIMULATOR ----------------
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
