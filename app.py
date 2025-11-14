
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import json
import os
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.markdown("""
    <style>
        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #a2d2ff, #b9fbc0, #ffe6a7, #ffd6e0);
            background-size: 400% 400%;
            animation: gradientMove 18s ease infinite;
            color: #222;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #001f3f, #004080, #0074D9);
            background-size: 400% 400%;
            animation: gradientMove 15s ease infinite;
            color: white !important;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #0074D9, #00BCD4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #00BCD4, #0074D9);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="💳 Card Fraud Dashboard", layout="wide")

USER_DB = "users.json"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "register_mode" not in st.session_state:
    st.session_state.register_mode = False
if "username" not in st.session_state:
    st.session_state.username = None

if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

def hash_password(password: str) -> str:
    """Return SHA256 hex digest of password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_users():
    with open(USER_DB, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_users(users: dict):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)

def register_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        return False, "⚠️ Username and password cannot be empty."
    users = load_users()
    if username in users:
        return False, "⚠️ Username already exists."
    users[username] = {
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    save_users(users) 
    return True, "✅ Registration successful. You may now log in."

def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    if username in users and users[username].get("password_hash") == hash_password(password):
        return True
    return False

def login_page():
    st.markdown("<h1 style='text-align:center;'>🔐 Card Fraud Dashboard</h1>", unsafe_allow_html=True)
    st.write("---")

    if st.session_state.register_mode:
        st.subheader("📝 Register New Account")
        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            new_user = st.text_input("👤 Create Username", key="reg_user")
            new_name = st.text_input("🧾 Full name (optional)", key="reg_name")
        with r_col2:
            new_pass = st.text_input("🔑 Create Password", type="password", key="reg_pass")
            new_pass2 = st.text_input("🔐 Confirm Password", type="password", key="reg_pass2")

        if st.button("Register"):
            if new_pass != new_pass2:
                st.error("❌ Passwords do not match.")
            else:
                ok, msg = register_user(new_user, new_pass)
                if ok:
                    st.success(msg)
                    st.session_state.register_mode = False
                    st.session_state.prefill_user = new_user
                    st.rerun()
                else:
                    st.error(msg)

        if st.button("← Back to Login"):
            st.session_state.register_mode = False
            st.rerun()

    else:
        st.subheader("🔑 Login")
        login_col1, login_col2 = st.columns([2, 1])
        with login_col1:
            user = st.text_input("👤 Username", value=st.session_state.get("prefill_user", ""), key="login_user")
            pw = st.text_input("🔒 Password", type="password", key="login_pass")
        with login_col2:
            if st.button("Login"):
                if authenticate_user(user, pw):
                    st.session_state.authenticated = True
                    st.session_state.username = user

                    log_file = "login_log.json"
                    logs = []
                    if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                            try:
                                logs = json.load(f)
                            except Exception:
                                logs = []
                    logs.append({
                        "username": user,
                        "action": "login",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_used": None
                    })
                    with open(log_file, "w") as f:
                        json.dump(logs, f, indent=2)

                    st.success("✅ Login successful. Loading dashboard...")
                    st.rerun()

                else:
                    st.error("❌ Invalid username or password.")

            if st.button("🆕 Register"):
                st.session_state.register_mode = True
                st.rerun()

if not st.session_state.authenticated:
    login_page()
    st.stop()

st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.username}**")
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

st.markdown("""
    <style>
        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #a2d2ff, #b9fbc0, #ffe6a7, #ffd6e0);
            background-size: 400% 400%;
            animation: gradientMove 18s ease infinite;
            color: #222;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #001f3f, #004080, #0074D9);
            background-size: 400% 400%;
            animation: gradientMove 15s ease infinite;
            color: white !important;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #0074D9, #00BCD4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #00BCD4, #0074D9);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Card Fraud Data Analysis Dashboard")
st.markdown("""
### 🎯 Project Goal
Analyze and model credit/debit card transactions to visualize:
- Fraud vs Non-Fraud distribution  
- Transaction amount variation  
- Correlation between features  
- Fraud trends over time  
- **AI prediction models for fraud detection**
---
""")

st.sidebar.header("📂 Navigation")
menu = st.sidebar.radio(
    "Select a section:",
    ["📘 Overview", "💰 Fraud Analysis", "📈 Trends Over Time", "🤖 AI Fraud Prediction"]
)

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        log_file = "login_log.json"
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []
        logs.append({
            "username": st.session_state.username,
            "action": "file_upload",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_used": uploaded_file.name
        })
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    except MemoryError:
        st.warning("⚠️ File too big — loading only the first 100,000 rows for preview.")
        df = pd.read_csv(uploaded_file, nrows=100_000)

    detected_fraud = [c for c in df.columns if 'fraud' in c.lower() or 'class' in c.lower() or 'target' in c.lower()]
    detected_amount = [c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower()]
    detected_time = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]

    st.sidebar.markdown("### 🔧 Column Selection")
    fraud_col = st.sidebar.selectbox("Fraud Label Column:", ["None"] + list(df.columns),
                                     index=(list(df.columns).index(detected_fraud[0]) + 1 if detected_fraud else 0))
    amount_col = st.sidebar.selectbox("Transaction Amount Column:", ["None"] + list(df.columns),
                                      index=(list(df.columns).index(detected_amount[0]) + 1 if detected_amount else 0))
    time_col = st.sidebar.selectbox("Date/Time Column:", ["None"] + list(df.columns),
                                    index=(list(df.columns).index(detected_time[0]) + 1 if detected_time else 0))

    if menu == "📘 Overview":
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head())
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        st.markdown("### 🧩 Missing Values")
        st.dataframe(df.isnull().sum())
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe())

    elif menu == "💰 Fraud Analysis":
        fraud_fig, amount_fig, box_fig, heatmap_fig = None, None, None, None
        if fraud_col != "None":
            st.subheader(f"🔍 Fraud Label Column: `{fraud_col}`")
            fraud_counts = df[fraud_col].value_counts()
            fraud_fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette="coolwarm", ax=ax)
            ax.set_title("Fraud vs Non-Fraud Transactions")
            st.pyplot(fraud_fig)
            pie_fig = px.pie(values=fraud_counts.values, names=fraud_counts.index, title="Fraud Distribution",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(pie_fig, use_container_width=True)
            fraud_percent = (fraud_counts.min() / fraud_counts.sum()) * 100
            st.markdown(f"⚠️ **Fraud Percentage:** `{fraud_percent:.3f}%`")
        else:
            st.warning("⚠️ Please select a fraud label column in the sidebar.")

        if amount_col != "None":
            st.subheader(f"💰 Transaction Amount Analysis (`{amount_col}`)")
            amount_fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[amount_col], bins=50, kde=True, ax=ax)
            ax.set_title("Transaction Amount Distribution")
            st.pyplot(amount_fig)
            if fraud_col != "None":
                box_fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[fraud_col], y=df[amount_col], palette="Set2", ax=ax)
                ax.set_title("Transaction Amount by Fraud Status")
                st.pyplot(box_fig)
        else:
            st.warning("⚠️ Please select an amount column.")

        st.subheader("🔗 Correlation Heatmap")
        heatmap_fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Between Numerical Features")
        st.pyplot(heatmap_fig)

        st.markdown("---")
        st.subheader("📄 Download Analysis Report")
        if st.button("📥 Generate & Download Report"):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [Paragraph("💳 Card Fraud Analysis Report", styles["Title"]),
                     Spacer(1, 12),
                     Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
                     Spacer(1, 12)]
            def add_plot(fig, name):
                path = f"{name}.png"
                fig.savefig(path, bbox_inches="tight")
                story.append(Image(path, width=400, height=250)); story.append(Spacer(1, 15))
            if fraud_fig: add_plot(fraud_fig, "fraud_plot")
            if amount_fig: add_plot(amount_fig, "amount_plot")
            if box_fig: add_plot(box_fig, "box_plot")
            if heatmap_fig: add_plot(heatmap_fig, "heatmap_plot")
            story.append(Paragraph("✅ End of Report", styles["Normal"]))
            doc.build(story)
            st.download_button("⬇️ Download PDF Report", buffer.getvalue(),
                               file_name="fraud_analysis_report.pdf", mime="application/pdf")

    elif menu == "📈 Trends Over Time":
        if time_col != "None":
            st.subheader(f"🕒 Time Column: `{time_col}`")
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df_time = df.dropna(subset=[time_col])
            if fraud_col != "None":
                time_trend = df_time.groupby(df_time[time_col].dt.date)[fraud_col].sum().reset_index()
                time_trend = time_trend.rename(columns={time_trend.columns[0]: time_col, time_trend.columns[1]: fraud_col})
                line_fig = px.line(time_trend, x=time_col, y=fraud_col,
                                   title="Fraudulent Transactions Over Time",
                                   markers=True, color_discrete_sequence=["#d00000"])
                st.plotly_chart(line_fig, use_container_width=True)
            else:
                st.warning("⚠️ Please select a fraud column for trend analysis.")
        else:
            st.info("ℹ️ Please select a time/date column in the sidebar.")

    elif menu == "🤖 AI Fraud Prediction":
        if fraud_col == "None":
            st.warning("⚠️ Please select a fraud label column to train AI models.")
        else:
            st.subheader("🧠 Machine Learning Fraud Detection Models")
            df_model = df.select_dtypes(include=[np.number]).dropna()
            if fraud_col not in df_model.columns:
                st.error("⚠️ Selected fraud column is not numeric — cannot train models with current selection.")
            else:
                X = df_model.drop(columns=[fraud_col])
                y = df_model[fraud_col]
                if X.shape[0] < 10:
                    st.warning("⚠️ Not enough numeric rows to train models. Need more data after dropping NaNs.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                    }

                    results = {}
                    for name, model in models.items():
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        acc = accuracy_score(y_test, y_pred)
                        results[name] = acc

                    best_model = max(results, key=results.get)
                    st.success(f"🏆 Best Model: **{best_model}** with Accuracy: `{results[best_model]*100:.2f}%`")

                    st.markdown("### 📊 Model Comparison")
                    comp_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)
                    st.bar_chart(comp_df)

                    if best_model == "Random Forest":
                        feature_imp = pd.Series(models[best_model].feature_importances_, index=X.columns)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        feature_imp.nlargest(10).plot(kind='barh', ax=ax)
                        ax.set_title("Top 10 Important Features (Random Forest)")
                        st.pyplot(fig)
else:
    st.info("👆 Upload a CSV file from the sidebar to begin.")

