
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="🛡️ Admin Dashboard", layout="wide")

st.title("🛡️ Admin Control Panel - Card Fraud System")

st.markdown("""
This dashboard shows:
- 👤 User login activity  
- 📂 Files uploaded for analysis  
- ⏰ Login timestamps  
- 📄 Option to download full log
""")

LOG_FILE = "login_log.json"
USER_DB = "users.json"

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

def admin_login():
    st.markdown("### 🔐 Admin Login")
    admin_user = st.text_input("👤 Username")
    admin_pass = st.text_input("🔑 Password", type="password")
    if st.button("Login as Admin"):
        if admin_user == "admin" and admin_pass == "12345":  # change as needed
            st.session_state.admin_logged_in = True
            st.success("✅ Welcome, Admin!")
        else:
            st.error("❌ Invalid admin credentials.")

if not st.session_state.admin_logged_in:
    admin_login()
    st.stop()

st.sidebar.success("✅ Logged in as Admin")
if st.sidebar.button("🚪 Logout"):
    st.session_state.admin_logged_in = False
    st.rerun()

st.markdown("---")

if not os.path.exists(LOG_FILE):
    st.warning("⚠️ No login data found yet.")
else:
    with open(LOG_FILE, "r") as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []

    if logs:
        df_logs = pd.DataFrame(logs)
        df_logs = df_logs.sort_values(by="timestamp", ascending=False)

        st.subheader("📜 Login and File Upload History")
        st.dataframe(df_logs, use_container_width=True)

        users = sorted(df_logs["username"].dropna().unique().tolist())
        selected_user = st.selectbox("Filter by user:", ["All"] + users)
        if selected_user != "All":
            df_filtered = df_logs[df_logs["username"] == selected_user]
        else:
            df_filtered = df_logs

        actions = sorted(df_logs["action"].dropna().unique().tolist())
        selected_action = st.selectbox("Filter by action:", ["All"] + actions)
        if selected_action != "All":
            df_filtered = df_filtered[df_filtered["action"] == selected_action]

        st.markdown("### 🔍 Filtered Results")
        st.dataframe(df_filtered, use_container_width=True)

        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Log as CSV",
            csv_data,
            file_name=f"login_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ Log file is empty — no user activity recorded yet.")

if os.path.exists(USER_DB):
    st.markdown("---")
    st.subheader("👥 Registered Users")
    with open(USER_DB, "r") as f:
        try:
            users = json.load(f)
        except Exception:
            users = {}

    if isinstance(users, dict) and users:
        df_users = pd.DataFrame({
            "Username": list(users.keys()),
            "Password (hashed or plain)": list(users.values())
        })
        st.dataframe(df_users, use_container_width=True)
    else:
        st.info("ℹ️ No registered users found.")
