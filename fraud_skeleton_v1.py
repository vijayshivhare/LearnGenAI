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

# === Configuration ===
load_dotenv()

# Load messages from the YAML file
with open('messages_v1.yaml', 'r') as file:
    MESSAGES = yaml.safe_load(file)

# Initialize LangChain LLM for OpenAI models
llm_model = ChatOpenAI(model_name="gpt-4o") # Using gpt-4o as specified in v1.1

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

def explain_fraud(df: pd.DataFrame):
    if not openai.api_key:
        return "OpenAI API key not set."
    try:
        #prompt = f"The following events occurred in order: {events}. Explain what kind of fraud pattern this might be and why. Also suggest a rule that could help flag similar fraud attempts in future. Format the rule in pseudo-SQL like: IF ... THEN ..."
        #response = openai.ChatCompletion.create(
        #    model="gpt-4",
        #    messages=[
        #        {"role": "system", "content": "You are a fraud analyst assistant."},
        #        {"role": "user", "content": prompt}
        #    ],
        #    temperature=0.3
        #)
        #explanation = response['choices'][0]['message']['content']
        #show_llm_progress_steps()
        explanation = FraudDataInsightAgent(df)
        return explanation
    except Exception as e:
        show_llm_progress_steps()
        return f"LLM Error: {e}"

def chat_with_llm(user_q):
    if not openai.api_key:
        return "OpenAI API key not set."
    try:
        #response = openai.ChatCompletion.create(
        #    model="gpt-4",
        #    messages=[
        #        {"role": "system", "content": "You are a helpful fraud analyst assistant."},
        #        {"role": "user", "content": question}
        #    ]
        #)
        #answer = response['choices'][0]['message']['content']
        code, should_plot_flag, code_thinking = FraudCodeGenerationAgent(user_q, df)
        result_obj = FraudExecutionAgent(code, df, should_plot_flag)
        raw_thinking, reasoning_txt, pattern_description = FraudReasoningAgent(user_q, result_obj)

        rule_suggestion = ""
        if pattern_description:
            rule_suggestion = FraudRuleGenerationAgent(pattern_description)
        # Build assistant response
        is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
        plot_idx = None
        if is_plot:
            fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
            st.session_state.plots.append(fig)
            plot_idx = len(st.session_state.plots) - 1
            header = "Here is the visualization you requested:"
        elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
            header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
        else:
            header = f"Result: {result_obj}"

        # Show only reasoning thinking in Model Thinking (collapsed by default)
        thinking_html = ""
        if raw_thinking:
            thinking_html = (
                    '<details class="thinking">'
                    '<summary>🧠 Reasoning</summary>'
                    f'<pre>{raw_thinking}</pre>'
                    '</details>'
                )

        # Show model explanation and pattern directly
        explanation_html = reasoning_txt
        if pattern_description:
            explanation_html += f"\n\n**Pattern Detected:** {pattern_description}"


        # Code accordion with proper HTML <pre><code> syntax highlighting
        code_html = (
                '<details class="code">'
                '<summary>View code</summary>'
                '<pre><code class="language-python">'
                f'{code}'
                '</code></pre>'
                '</details>'
            )
        
        #show_llm_progress_steps()
        print(f"{thinking_html}{explanation_html}\n\n{code_html}")
        return f"{thinking_html}{explanation_html}\n\n{code_html}"
    except Exception as e:
        show_llm_progress_steps()
        return f"Chat Error: {e}"
    
# === FraudCodeGeneration TOOLS ============================================

# ------------------ FraudQueryUnderstandingTool ---------------------------
def FraudQueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    messages = [
        SystemMessage(content=MESSAGES['fraud_query_understanding_tool']['system_message']),
        HumanMessage(content=query)
    ]

    # Assuming llm_model is defined elsewhere
    response = llm_model.invoke(messages, config={"max_tokens": 5, "temperature": 0.1})

    # Extract the response content and convert to boolean
    intent_response = response.content.strip().lower()
    return intent_response == "true"

# === FraudCodeGeneration TOOLS ============================================

# ------------------ FraudPlotCodeGeneratorTool ---------------------------
def FraudPlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns."""
    prompt_template = MESSAGES['fraud_plot_code_generator']['prompt']
    return prompt_template.format(cols=', '.join(cols), query=query)

# ------------------ FraudCodeWritingTool ---------------------------------
def FraudCodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    prompt_template = MESSAGES['fraud_code_writing']['prompt']
    return prompt_template.format(cols=', '.join(cols), query=query)

# === FraudCodeGenerationAgent ==============================================
def FraudCodeGenerationAgent(query: str, df: pd.DataFrame):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    # Assuming FraudQueryUnderstandingTool is defined elsewhere
    should_plot = FraudQueryUnderstandingTool(query)
    tool_name = "Fraud Plot Code Generator Tool" if should_plot else "Fraud Code Writing Tool"

    # Log the action of the FraudCodeGenerationAgent
    #log_agent_action(
    #    agent_name="Fraud Code Generation Agent", # Example agent name from image
    #    tool_used=tool_name,
    #    thought="To generate the appropriate code based on the user's query, I need to determine if a plot is requested and then use the correct code generation tool."
    #)
    prompt_content = FraudPlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else FraudCodeWritingTool(df.columns.tolist(), query)

    messages = [
        SystemMessage(content=MESSAGES['fraud_code_generation_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]

    # Assuming llm_model is defined elsewhere
    response = llm_model.invoke(messages, config={"max_tokens": 1024, "temperature": 0.2})

    full_response = response.content
    # Assuming extract_first_code_block is defined elsewhere
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# === FraudExecutionAgent ====================================================
def FraudExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    # Log the action of the FraudExecutionAgent
    #log_agent_action(
    #    agent_name="Fraud Execution Agent", 
    #    tool_used="Python Interpreter/Execution Environment",
    #    thought="To execute the generated Python code and obtain the result, I will run it in a safe environment."
    #)

    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"

# === FraudReasoningCurator TOOL =========================================
def FraudReasoningCurator(query: str, result: Any) -> Tuple[str, bool, bool]:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]

    if is_plot:
        prompt = MESSAGES['fraud_reasoning_curator']['plot_description'].format(query=query, desc=desc)
    else:
        prompt = MESSAGES['fraud_reasoning_curator']['data_description'].format(query=query, desc=desc)
    return prompt, is_error, is_plot

# === FraudReasoningAgent (streaming) =========================================
def FraudReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    # Log the action of the FraudReasoningAgent
    #log_agent_action(
    #    agent_name="Fraud Reasoning Agent", 
    #    tool_used="Fraud Reasoning Curator",
    #    thought="To interpret the results and identify potential fraud patterns."
    #)    
    
    prompt_content, is_error, is_plot = FraudReasoningCurator(query, result)
    
    messages = [
        SystemMessage(content=MESSAGES['fraud_reasoning_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]

    # Streaming LLM call using LangChain
    # Assuming llm_model and st (streamlit) are defined elsewhere
    response_generator = llm_model.stream(messages, config={"max_tokens": 1024, "temperature": 0.2})

    # Stream and display thinking
    # Assuming st.empty() and st.markdown are from Streamlit
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response_generator:
        token = chunk.content
        full_response += token

        # Simple state machine to extract <think>...</think> as it streams
        if "<think>" in token:
            in_think = True
            token = token.split("<think>", 1)[1]
        if "</think>" in token:
            token = token.split("</think>", 1)[0]
            in_think = False
        if in_think or ("<think>" in full_response and not "</think>" in full_response):
            thinking_content += token
            thinking_placeholder.markdown(
                f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                unsafe_allow_html=True
            )

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

    # Extract pattern description
    pattern_match = re.search(r"Pattern Detected: (.*)", cleaned, re.IGNORECASE)
    pattern_description = pattern_match.group(1).strip() if pattern_match else ""
    reasoning_without_pattern = re.sub(r"Pattern Detected:.*", "", cleaned, flags=re.DOTALL).strip()

    return thinking_content, reasoning_without_pattern, pattern_description



# === FraudRuleGenerationAgent =========================================
def FraudRuleGenerationAgent(pattern_description: str) -> str:
    """
    Uses the LLM to generate a simple 'IF [condition] THEN [action]' rule
    based on the identified fraud pattern.
    """

    # Log the action of the FraudRuleGenerationAgent
    #log_agent_action(
    #    agent_name="Fraud Rule Generation Agent", 
    #    tool_used="Fraud Rule Generation Tool",
    #    thought="To generate a concise fraud detection rule based on the identified pattern."
    #)
    if not pattern_description:
        return MESSAGES['fraud_rule_generation_agent']['no_pattern_message']

    prompt_content = MESSAGES['fraud_rule_generation_agent']['prompt'].format(pattern_description=pattern_description)
    messages = [
        SystemMessage(content=MESSAGES['fraud_rule_generation_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]

    try:
        response = llm_model.invoke(messages, config={"max_tokens": 100, "temperature": 0.1})
        return response.content.strip()
    except Exception as exc:
        return MESSAGES['fraud_rule_generation_agent']['error_message'].format(exc=exc)


# === FraudDataFrameSummary TOOL (pandas only) =========================================
def FraudDataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = MESSAGES['fraud_dataframe_summary_tool']['prompt'].format(
        num_rows=len(df),
        num_cols=len(df.columns),
        data_types=df.dtypes.to_dict(),
        sample_data=df.head(10).to_csv(index=False)
    )
    return prompt

# === FraudDataInsightAgent (upload-time only) ===============================
def FraudDataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset."""
     # Log the action of the FraudDataInsightAgent
    #log_agent_action(
    #    agent_name="Fraud Data Insight Agent", 
    #    tool_used="Fraud Data Frame Summary Tool",
    #    thought="To provide an initial summary of the dataset and suggest potential fraud patterns and follow-up questions."
    #)

    prompt_content = FraudDataFrameSummaryTool(df)
    messages = [
        SystemMessage(content=MESSAGES['fraud_data_insight_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]
    try:
        response = llm_model.invoke(messages, config={"max_tokens": 512, "temperature": 0.2})
        return response.content
    except Exception as exc:
        return MESSAGES['fraud_data_insight_agent']['error_message'].format(exc=exc)

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    st.sidebar.header("Filters")

    customer_ids = df['Customer ID'].dropna().unique()
    sorted_customer_ids = sorted(customer_ids.astype(str))
    selected_customer = st.sidebar.selectbox("Select Customer ID", options=["All"] + sorted_customer_ids)

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

    st.subheader("📊 Fraud Overview")

    # Display Key Metrics
    col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)

    with col_metrics_1:
        st.metric(label="Total Transactions", value=len(df)) # Total loaded transactions
    with col_metrics_2:
        st.metric(label="Total Fraud Cases", value=len(filtered_df)) # Fraud cases after filters
    with col_metrics_3:
        # Example: Number of unique customers involved in fraud
        if 'Customer ID' in filtered_df.columns:
            st.metric(label="Unique Fraud Customers", value=filtered_df['Customer ID'].nunique())
        else:
            st.metric(label="Unique Fraud Customers", value="N/A")
    with col_metrics_4:
        # Example: Total fraudulent amount
        if 'Txn Amount' in filtered_df.columns:
            st.metric(label="Total Fraud Amount", value=f"${filtered_df['Txn Amount'].sum():,.2f}")
        else:
            st.metric(label="Total Fraud Amount", value="N/A")


    st.markdown("---") # Separator

    st.subheader("🧠 Common Fraud Patterns Summary")
    with st.spinner("Analyzing common patterns..."):
        # The general summary (explain_fraud uses the full df, not filtered_df)
        st.write(explain_fraud(df))

    st.markdown("---") # Another separator before visualizations or other sections

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

    st.markdown("---") # Another separator before visualizations or other sections
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

    st.markdown("---") # Another separator before visualizations or other sections
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
                            chat_with_llm("Please provide fraud reason for Customer ID: "+row['Customer ID'])
                            #st.markdown("**LLM Explanation and Suggested Rule:**")
                            #st.write(explanation)
                except:
                    st.warning("Story format is not valid list")

else:
    st.info("Please upload a CSV file with columns: Customer ID, Txn Amount, Device, IP Address, Fraud Type, Time, and Story (list of events as string).")
