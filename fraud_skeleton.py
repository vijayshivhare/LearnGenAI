import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from time import sleep

st.set_page_config(page_title="Fraud Pattern UI", layout="wide")
st.title("🔍 Fraud Pattern Explorer")


openai.api_key = ""

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

def show_llm_progress_steps():
    steps = [
        "🔍 Understanding query with Fraud Query Understanding Tool",
        "🧠 Generating intermediate code using Fraud Code Generation Tool",
        "🛡️ Generating fraud rule using Fraud Rule Generation Tool"
    ]
    st.markdown("<div style='display: flex; gap: 15px; margin-top: 1em;'>" +
                "".join([f"<div style='background-color:#e6f7ff; padding:10px; border-radius:5px;'>{step}</div>" for step in steps]) +
                "</div>", unsafe_allow_html=True)

def explain_fraud(events):
    if not openai.api_key:
        return "OpenAI API key not set."
    try:
        prompt = f"The following events occurred in order: {events}. Explain what kind of fraud pattern this might be and why. Also suggest a rule that could help flag similar fraud attempts in future. Format the rule in pseudo-SQL like: IF ... THEN ..."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a fraud analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        explanation = response['choices'][0]['message']['content']
        show_llm_progress_steps()
        return explanation
    except Exception as e:
        show_llm_progress_steps()
        return f"LLM Error: {e}"

def chat_with_llm(question):
    if not openai.api_key:
        return "OpenAI API key not set."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful fraud analyst assistant."},
                {"role": "user", "content": question}
            ]
        )
        answer = response['choices'][0]['message']['content']
        show_llm_progress_steps()
        return answer
    except Exception as e:
        show_llm_progress_steps()
        return f"Chat Error: {e}"

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    st.sidebar.header("Filters")

    customer_ids = df['Customer ID'].dropna().unique()
    selected_customer = st.sidebar.selectbox("Select Customer ID", options=["All"] + sorted(customer_ids.astype(str)))

    fraud_types = df["Fraud Type"].dropna().unique()
    selected_fraud = st.sidebar.selectbox("Select Fraud Type", options=["All"] + sorted(fraud_types))

    min_date = df['Time'].min().date() if 'Time' in df.columns else None
    max_date = df['Time'].max().date() if 'Time' in df.columns else None

    start_date = st.sidebar.date_input("Start Date", min_value=min_date, value=min_date)
    end_date = st.sidebar.date_input("End Date", min_value=min_date, value=max_date)

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    filtered_df = df.copy()
    if selected_customer != "All":
        filtered_df = filtered_df[filtered_df["Customer ID"].astype(str) == selected_customer]
    if selected_fraud != "All":
        filtered_df = filtered_df[filtered_df["Fraud Type"] == selected_fraud]
    if 'Time' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Time'] >= start_datetime) & (filtered_df['Time'] <= end_datetime)]

    st.subheader("Summary")
    st.metric(label="Total Fraud Cases", value=len(filtered_df))

    if st.checkbox("🧠 Generate Summary of Common Patterns"):
        top_patterns = filtered_df['Story'].value_counts().head(3).index.tolist()
        summaries = []
        for p in top_patterns:
            try:
                events = eval(p)
                summaries.append((p, explain_fraud(events)))
            except:
                summaries.append((p, "Invalid format"))
        for i, (pattern, explanation) in enumerate(summaries):
            st.markdown(f"**Pattern {i+1}:** {pattern}")
            st.write(explanation)

    with st.expander("📈 Visualize Fraud Patterns", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if 'Fraud Type' in filtered_df.columns:
                st.markdown("**Fraud Count by Type**")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=filtered_df, x='Fraud Type', ax=ax1)
                ax1.set_title("Fraud Cases by Type")
                ax1.tick_params(axis='x', rotation=45)
                st.pyplot(fig1)

        with col2:
            if 'Time' in filtered_df.columns:
                st.markdown("**Fraud Over Time**")
                time_series = filtered_df.set_index('Time').resample('D').size()
                fig2, ax2 = plt.subplots()
                time_series.plot(ax=ax2)
                ax2.set_title("Fraud Cases Over Time")
                ax2.set_ylabel("Cases")
                st.pyplot(fig2)

        st.markdown("---")
        st.markdown("**Box Plot - Txn Amount by Fraud Type**")
        fig3 = px.box(filtered_df, x='Fraud Type', y='Txn Amount', title="Transaction Amount Distribution by Fraud Type")
        st.plotly_chart(fig3)

        st.markdown("**Device Usage by Fraud Type**")
        fig4 = px.histogram(filtered_df, x='Device', color='Fraud Type', barmode='group', title="Device Usage by Fraud Type")
        st.plotly_chart(fig4)

        st.markdown("**Top IP Addresses Involved in Fraud**")
        top_ips = filtered_df['IP Address'].value_counts().head(10).reset_index()
        top_ips.columns = ['IP Address', 'Count']
        fig5 = px.bar(top_ips, x='IP Address', y='Count', title="Top 10 IPs in Fraudulent Transactions")
        st.plotly_chart(fig5)

    st.subheader("Fraudulent Transactions")
    for index, row in filtered_df.iterrows():
        with st.expander(f"🧾 Customer ID: {row['Customer ID']} | {row['Fraud Type']} | ${row['Txn Amount']}"):
            st.write(f"**Device:** {row.get('Device', 'N/A')}")
            st.write(f"**IP Address:** {row.get('IP Address', 'N/A')}")
            st.write(f"**Time:** {row.get('Time', 'N/A')}")
            if isinstance(row.get('Story'), str):
                try:
                    events = eval(row['Story'])
                    if isinstance(events, list):
                        st.markdown("**Event Timeline:**")
                        for event in events:
                            st.markdown(f"- {event}")
                        with st.spinner("Analyzing pattern..."):
                            explanation = explain_fraud(events)
                            st.markdown("**LLM Explanation and Suggested Rule:**")
                            st.write(explanation)
                except:
                    st.warning("Story format is not valid list")

    st.subheader("💬 Ask a question to the Fraud Assistant")
    st.markdown("**Suggested Questions:**")
    for q in [
        "What are the most common fraud devices?",
        "Which IP addresses are used most often in fraud?",
        "What pattern is common among high-value frauds?"
    ]:
        if st.button(q):
            with st.spinner("Analyzing your question..."):
                response = chat_with_llm(q)
                st.markdown(f"**Question:** {q}")
                st.markdown("**Assistant Response:**")
                st.write(response)

    user_input = st.text_input("Type your question here")
    if user_input:
        with st.spinner("Analyzing your question..."):
            response = chat_with_llm(user_input)
            st.markdown("**Assistant Response:**")
            st.write(response)

else:
    st.info("Please upload a CSV file with columns: Customer ID, Txn Amount, Device, IP Address, Fraud Type, Time, and Story (list of events as string).")
