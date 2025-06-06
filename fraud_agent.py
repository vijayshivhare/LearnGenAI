import os, io, re
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from time import sleep
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import yaml
from pandasai import SmartDatalake
from pandasai.llm import OpenAI
from PIL import Image

# === Configuration (Combined from both original files) ===
load_dotenv() # Load environment variables

st.set_page_config(page_title="Combined Fraud Analysis App", layout="wide")
st.title("Welcome to the Combined Fraud Analysis Application!")
st.write("Upload your transaction data below, then select a tab to begin your analysis.")

# --- API Key Setup (Centralized) ---
# It's recommended to use st.secrets for API keys in a deployed app
openai_api_key = "" # Replace with your actual key
openai.api_key = openai_api_key # For langchain and openai library directly

# Initialize LangChain LLM for OpenAI models
llm_model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key) # Pass API key to LangChain LLM

# Initialize PandasAI LLM
pandasai_llm = OpenAI(api_token=openai_api_key)

# Load messages from the YAML file
try:
    # Ensure 'messages_v2.yaml' is in the same directory as this app.py
    with open('messages_v2.yaml', 'r') as file:
        MESSAGES = yaml.safe_load(file)
except FileNotFoundError:
    st.error("Error: messages_v2.yaml not found. Please ensure it's in the same directory as app.py.")
    st.stop()


# --- Centralized File Uploader ---
uploaded_file_central = st.file_uploader("Upload your Excel transaction data (xlsx, xls)", type=["xlsx", "xls"])

if uploaded_file_central:
    try:
        all_sheets_data = pd.read_excel(uploaded_file_central, sheet_name=None)
        st.session_state.all_sheets_data = all_sheets_data
        st.success("Data uploaded and ready for analysis!")

        st.subheader("Uploaded Sheets Overview:")
        for sheet_name, df in all_sheets_data.items():
            st.write(f"**Sheet: {sheet_name}** ({len(df)} rows, {len(df.columns)} columns)")
            st.dataframe(df.head(3))

    except Exception as e:
        st.error(f"Error reading Excel file: {e}. Please ensure it's a valid Excel format.")
        st.session_state.all_sheets_data = None
else:
    st.info("No data uploaded yet. Please upload an Excel file to enable analysis on the tabs below.")
    st.session_state.all_sheets_data = None


# --- Create Tabs ---
tab1, tab2 = st.tabs(["🔍 Fraud Pattern Explorer", "🧠 AI-Powered Data Analysis"])

with tab1:
    st.subheader("Fraud Pattern Explorer")

    if 'all_sheets_data' in st.session_state and st.session_state.all_sheets_data is not None:
        sheet_names = list(st.session_state.all_sheets_data.keys())
        if len(sheet_names) > 0:
            # Assuming the first sheet or a specific one is the fraud data
            current_fraud_df = st.session_state.all_sheets_data[sheet_names[0]]
            st.info(f"Analyzing data from sheet: '{sheet_names[0]}'")

            # Initialize FraudQueryAssistantAgent with the current DataFrame
            agent = FraudQueryAssistantAgent(df=current_fraud_df, llm_model=llm_model, messages=MESSAGES)

            st.subheader("Data Overview (First 5 Rows)")
            st.dataframe(current_fraud_df.head())

            st.markdown("---")
            st.subheader("Suggested Questions:")
            suggested_questions_fraud = [
                "What are the top 5 transaction types by amount?",
                "Show me the distribution of transaction amounts.",
                "Are there any outliers in transaction amounts?",
                "What is the average transaction amount for each transaction type?",
            ]
            for q in suggested_questions_fraud:
                if st.button(q, key=f"suggested_q_fraud_{q}"):
                    with st.spinner("Analyzing your question..."):
                        response_fraud = agent.chat_with_llm(q)
                        if not st.session_state.chat_history or st.session_state.chat_history[-1].get("question") != q:
                            st.session_state.chat_history.append({"question": q, "response": response_fraud, "is_plot": False})
                        st.write(response_fraud) # Display text response immediately

            user_input_fraud = st.text_input("Type your question here for Fraud Pattern Explorer", key="user_question_fraud_input")
            if user_input_fraud:
                with st.spinner("Analyzing your question..."):
                    response_fraud = agent.chat_with_llm(user_input_fraud)
                    if not st.session_state.chat_history or st.session_state.chat_history[-1].get("question") != user_input_fraud:
                        st.session_state.chat_history.append({"question": user_input_fraud, "response": response_fraud, "is_plot": False})
                    st.write(response_fraud) # Display text response immediately


            # Display chat history and plots
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("Assistant Responses History (Fraud)")
                for i, entry in enumerate(st.session_state.chat_history):
                    if "plot" in entry['question'].lower() and entry.get("is_plot") and i < len(st.session_state.plots):
                        st.markdown(f"**Question:** {entry['question']}")
                        st.pyplot(st.session_state.plots[i])
                    else:
                        st.markdown(f"**Question:** {entry['question']}")
                        st.markdown("**Assistant Response:**")
                        st.write(entry['response'])
                    st.markdown("---")
        else:
            st.warning("No sheets found in the uploaded Excel file.")
    else:
        st.warning("Please upload an Excel file on the 'Welcome' page to use the Fraud Pattern Explorer.")


with tab2:
    st.subheader("AI-Powered Data Analysis")

    if 'all_sheets_data' in st.session_state and st.session_state.all_sheets_data is not None:
        # pandasai SmartDatalake expects a list of DataFrames
        data_for_pandasai = list(st.session_state.all_sheets_data.values())

        # Configure SmartDatalake
        config = {
            "llm": pandasai_llm, # Use the pandasai_llm here
            "save_charts": False, # Attempt to disable chart saving
        }

        datalake = SmartDatalake(data_for_pandasai, config=config)

        question_ai = st.text_input("Ask a question about your transaction data (e.g., 'What is the average transaction amount?', 'Show me a histogram of Amount by Type'):", key="ai_question_input")
        if question_ai:
            with st.spinner("Thinking..."):
                try:
                    response_ai = datalake.chat(question_ai)

                    # Debugging: Print the response to see what pandasai returns
                    print(f"Response type (AI tab): {type(response_ai)}")
                    print(f"Response value (AI tab): {response_ai}")

                    if isinstance(response_ai, pd.DataFrame):
                        st.dataframe(response_ai)
                    elif hasattr(response_ai, 'figure'): # Matplotlib plot object
                        st.pyplot(response_ai)
                    elif hasattr(response_ai, 'show'): # Plotly plot object
                        st.plotly_chart(response_ai)
                    elif isinstance(response_ai, str) and response_ai.endswith(".png"):
                        # Fallback: if pandasai still saves to file, display it
                        try:
                            image = Image.open(response_ai)
                            st.image(image, caption="Generated Plot")
                        except FileNotFoundError:
                            st.error(f"Plot file not found at: {response_ai}. Please check the path.")
                    else:
                        st.write(response_ai) # For text answers
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please try rephrasing your question.")
    else:
        st.warning("No data available for AI analysis. Please upload an Excel file on the 'Welcome' page.")