import os
import pymysql
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from io import BytesIO
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()  # baca file .env (lokal)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="📊 Its Kelly!", layout="wide")

# ---------------- DATABASE CONNECTION ----------------
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "Exraid2009"),
        database=os.getenv("DB_NAME", "user_db"),
        cursorclass=pymysql.cursors.DictCursor
    )

# ---------------- DB FUNCTIONS ----------------
def save_file_to_db(file_name, file_data):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO uploaded_files (file_name, file_data) VALUES (%s, %s)",
                (file_name, file_data),
            )
        conn.commit()

def get_uploaded_files():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, file_name FROM uploaded_files")
            return cur.fetchall()

def get_file_content(file_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT file_name, file_data FROM uploaded_files WHERE id=%s", (file_id,))
            return cur.fetchone()

# ---------------- SUMMARY FUNCTION ----------------
def generate_data_summary(df: pd.DataFrame):
    summary_report = {}
    summary_report["rows"] = df.shape[0]
    summary_report["columns"] = df.shape[1]
    summary_report["duplicates"] = df.duplicated().sum()
    summary_report["missing_total"] = int(df.isna().sum().sum())
    summary_report["missing_percent"] = round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    summary_report["dtypes"] = df.dtypes.value_counts().to_dict()
    return summary_report

# ---------------- CHUNK & AI ANALYSIS ----------------
def analyze_large_file(text, query, chunk_size=10000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI data auditor. Summarize clearly."},
                {"role": "user", "content": f"Chunk {i+1}/{len(chunks)}:\n{chunk}\n\nTask: {query}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        partial_summaries.append(response.choices[0].message.content)

    combined_summary = "\n\n".join(partial_summaries)
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI data auditor. Provide a final combined summary."},
            {"role": "user", "content": f"Here are all chunk summaries:\n{combined_summary}\n\nNow provide a FINAL overall analysis that combines everything."}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return final_response.choices[0].message.content

# ---------------- TABS ----------------
def show_file_upload_tab():
    st.header("📤 Upload Files")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            save_file_to_db(file.name, file.getvalue())
            st.success(f"✅ {file.name} saved to database")

def show_data_management_tab():
    st.header("🗂 Data Management - Source Table")
    uploaded_files = get_uploaded_files()
    if not uploaded_files:
        st.info("No files uploaded yet.")
        return

    file_options = {f"{f['file_name']} (ID: {f['id']})": f["id"] for f in uploaded_files}
    selected_file = st.selectbox("📂 Select a file:", list(file_options.keys()))

    if selected_file:
        file_id = file_options[selected_file]
        file_row = get_file_content(file_id)

        st.write(f"📌 **File Name:** {file_row['file_name']}")
        st.write(f"🆔 **Database ID:** {file_id}")

        if file_row and file_row["file_data"]:
            file_bytes = BytesIO(file_row["file_data"])
            file_name = file_row["file_name"]

            if file_name.endswith(".pdf"):
                pdf = PdfReader(file_bytes)
                text_list = [page.extract_text() for page in pdf.pages if page.extract_text()]
                st.text_area("📄 PDF Preview", "\n".join(text_list[:5]), height=200)

            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_bytes, dtype=str).fillna("")
                st.dataframe(df)

            elif file_name.endswith(".csv"):
                df = pd.read_csv(file_bytes, dtype=str).fillna("")
                st.dataframe(df)

            else:
                st.warning("⚠️ Unsupported file format")

def show_audit_report_tab():
    st.header("🤖 Audit Report - AI Assistant")
    uploaded_files = get_uploaded_files()
    if not uploaded_files:
        st.info("No files available for AI analysis.")
        return

    file_options = {f"{f['file_name']} (ID: {f['id']})": f["id"] for f in uploaded_files}
    selected_files = st.multiselect("Select up to 2 files for AI analysis:", list(file_options.keys()))

    if len(selected_files) == 0:
        return
    if len(selected_files) > 2:
        st.warning("⚠️ Please select maximum 2 files only.")
        return

    cols = st.columns(len(selected_files))
    dfs = []
    texts = []

    for idx, selected_file in enumerate(selected_files):
        file_id = file_options[selected_file]
        file_row = get_file_content(file_id)

        if file_row:
            file_bytes = BytesIO(file_row["file_data"])
            file_name = file_row["file_name"]
            df = None
            extracted_text = ""

            if file_name.endswith(".pdf"):
                pdf = PdfReader(file_bytes)
                extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                cols[idx].subheader(f"📄 {file_name} Preview")
                cols[idx].text_area("Extracted Data", extracted_text[:1500], height=200)

            elif file_name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_bytes).replace({np.nan: None})
                extracted_text = df.to_csv(index=False)

            elif file_name.endswith(".csv"):
                df = pd.read_csv(file_bytes).replace({np.nan: None})
                extracted_text = df.to_csv(index=False)

            if df is not None:
                dfs.append(df)
                summary = generate_data_summary(df)

                cols[idx].subheader(f"📊 {file_name} Summary")
                cols[idx].metric("Rows", f"{summary['rows']:,}")
                cols[idx].metric("Columns", f"{summary['columns']:,}")
                cols[idx].metric("Missing", f"{summary['missing_total']:,}", f"{summary['missing_percent']}%")
                cols[idx].metric("Duplicates", f"{summary['duplicates']:,}")

                with cols[idx].expander("🔍 Data Types"):
                    cols[idx].json(summary["dtypes"])

                with cols[idx].expander("📌 Column Overview"):
                    cols[idx].write(pd.DataFrame({
                        "Column": df.columns,
                        "Type": df.dtypes.astype(str),
                        "Missing": df.isna().sum().values,
                        "Unique": [df[c].nunique() for c in df.columns]
                    }))
            texts.append(extracted_text)

    query = st.text_input("💬 Ask AI to analyze these file(s):")
    if query:
        with st.spinner("AI is analyzing..."):
            if len(texts) == 1:
                answer = analyze_large_file(texts[0], query)
            else:
                combined_text = f"=== FILE 1 ===\n{texts[0]}\n\n=== FILE 2 ===\n{texts[1]}"
                answer = analyze_large_file(combined_text, query)

            st.success("✅ AI Response")
            st.info(answer)

# ---------------- MAIN APP ----------------
tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🗂 Data Management", "🤖 Audit Report"])

with tab1:
    show_file_upload_tab()

with tab2:
    show_data_management_tab()

with tab3:
    show_audit_report_tab()
