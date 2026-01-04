# 🚨 AI Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An AI-powered Network Intrusion Detection System (NIDS) that uses Machine Learning (Random Forest) to detect malicious network traffic from the CIC-IDS2017 dataset.

---

## 📸 Project Screenshots

Dashboard Overview  
![Dashboard](images/dashboard.png)

Model Performance & Confusion Matrix  
![Metrics](images/metrics.png)

Live Traffic Simulator  
![Simulator](images/simulator.png)

---

## 🧠 System Architecture

![Architecture](images/architecture.png)

The system works as follows:
1. Load real network traffic data from CIC-IDS2017  
2. Preprocess and select relevant features  
3. Train a Random Forest classifier  
4. Evaluate the model using accuracy and confusion matrix  
5. Predict live traffic as Benign or Malicious  

---

## ✨ Features

- Real-world CIC-IDS2017 dataset integration  
- Machine Learning–based intrusion detection  
- Interactive Streamlit dashboard  
- Accuracy and confusion matrix visualization  
- Live traffic simulation  
- Supports multiple CIC datasets  

---

## 🧰 Tech Stack

Language: Python  
Machine Learning Model: Random Forest  
Frontend: Streamlit  
Libraries: Pandas, NumPy, Scikit-learn  
Visualization: Matplotlib, Seaborn  

---

## 📂 Dataset

- **Dataset Name:** CIC-IDS2017  
- **Provided By:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick  
- **Official Dataset URL:**  
  https://www.unb.ca/cic/datasets/ids-2017.html  

**Note:**  
The dataset files are not included in this repository due to their large size.  
Please download the required CSV files from the official source and place them in the project root directory before running the application.


---

## ▶️ How to Run the Project

1. Clone the repository  
   git clone https://github.com/SonuKumarAnalyst/AI-Network-Intrusion-Detection-System.git

2. Go to the project directory  
   cd AI-Network-Intrusion-Detection-System

3. Install dependencies  
   pip install -r requirements.txt

4. Run the application  
   streamlit run nids_main.py

---

## 📁 Project Structure

AI-Network-Intrusion-Detection-System  
│── nids_main.py  
│── requirements.txt  
│── README.md  
│── .gitignore  
│── images  
│   ├── dashboard.png  
│   ├── metrics.png  
│   ├── simulator.png  
│   └── architecture.png  

---

## 🎓 Academic Use

This project is suitable for:
- BCA / MCA final year projects  
- Internship submissions  
- Machine Learning demonstrations  
- Cybersecurity coursework  

---

## 👤 Author

Sonu Kumar  
Aspiring Data Analyst & Machine Learning Enthusiast  

GitHub: https://github.com/SonuKumarAnalyst  

---

## 📜 License

This project is licensed under the MIT License.
