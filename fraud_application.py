import os, io, re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns # Added for countplot in original fraud_skeleton
import plotly.express as px # Added for plotly in original fraud_skeleton
from typing import List, Dict, Any, Tuple

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import yaml

# === Configuration ===
load_dotenv()

# Load messages from the YAML file
with open('messages.yaml', 'r') as file:
    MESSAGES = yaml.safe_load(file)

# Initialize LangChain LLM for OpenAI models
llm_model = ChatOpenAI(model_name="gpt-4o") # Using gpt-4o as specified in v1.1

# === New Logging Function ===
def log_agent_action(agent_name: str, tool_used: str, thought: str = None):
    """
    Logs agent actions and tool usage to the Streamlit UI,
    mimicking the format in the provided image.
    """
    st.markdown(f'<div style="background-color:#E0F2F7; padding: 10px; border-radius: 5px; margin-bottom: 10px;">'
                f'<b>Agent Name:</b> {agent_name}<br>'
                f'<b>Tool used:</b> {tool_used}', unsafe_allow_html=True)
    #if thought:
    #    st.markdown(f'<b>Thought:</b> {thought}', unsafe_allow_html=True)

# ------------------ FraudQueryUnderstandingTool ---------------------------
def FraudQueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    messages = [
        SystemMessage(content=MESSAGES['fraud_query_understanding_tool']['system_message']),
        HumanMessage(content=query)
    ]
    response = llm_model.invoke(messages, config={"max_tokens": 5, "temperature": 0.1})
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
    should_plot = FraudQueryUnderstandingTool(query)
    tool_name = "Fraud Plot Code Generator Tool" if should_plot else "Fraud Code Writing Tool"

    log_agent_action(
        agent_name="Fraud Code Generation Agent",
        tool_used=tool_name,
        thought="To generate the appropriate code based on the user's query, I need to determine if a plot is requested and then use the correct code generation tool."
    )
    prompt_content = FraudPlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else FraudCodeWritingTool(df.columns.tolist(), query)

    messages = [
        SystemMessage(content=MESSAGES['fraud_code_generation_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]
    response = llm_model.invoke(messages, config={"max_tokens": 1024, "temperature": 0.2})
    full_response = response.content
    code = extract_first_code_block(full_response)
    return code, should_plot, "" # code_thinking is not used currently from original v1.1

# === FraudExecutionAgent ====================================================
def FraudExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    log_agent_action(
        agent_name="Fraud Execution Agent",
        tool_used="Python Interpreter/Execution Environment",
        thought="To execute the generated Python code and obtain the result, I will run it in a safe environment."
    )

    env = {"pd": pd, "df": df, "plt": plt, "sns": sns, "px": px, "io": io} # Added sns and px for plotting
    if should_plot:
        plt.rcParams["figure.dpi"] = 100
        # For plotly, the code should return the figure object directly if it's px.
        # For matplotlib, plt.show() is not needed, as we capture the figure.
    try:
        # Using a dictionary for exec's locals to capture results
        exec_locals = {"pd": pd, "df": df, "plt": plt, "sns": sns, "px": px, "io": io, "result": None}
        exec(code, {}, exec_locals)
        # Check if result is set by the executed code
        if exec_locals.get("result") is not None:
            return exec_locals["result"]
        # If the code produced a matplotlib plot, it will be the current figure
        elif should_plot and plt.get_fignums():
            return plt.gcf()
        else:
            return None # Or a more informative message if no explicit result or plot
    except Exception as exc:
        return f"Error executing code: {exc}"

# === FraudReasoningCurator TOOL =========================================
def FraudReasoningCurator(query: str, result: Any) -> Tuple[str, bool, bool]:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes, px.Figure)) # Added px.Figure
    
    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        elif isinstance(result, px.Figure):
            title = result.layout.title.text if result.layout.title else ""
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
    log_agent_action(
        agent_name="Fraud Reasoning Agent",
        tool_used="Fraud Reasoning Curator",
        thought="To interpret the results and identify potential fraud patterns."
    )
    prompt_content, is_error, is_plot = FraudReasoningCurator(query, result)

    messages = [
        SystemMessage(content=MESSAGES['fraud_reasoning_agent']['system_message']),
        HumanMessage(content=prompt_content)
    ]

    response_generator = llm_model.stream(messages, config={"max_tokens": 1024, "temperature": 0.2})

    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response_generator:
        token = chunk.content
        full_response += token

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

    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
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
    log_agent_action(
        agent_name="Fraud Rule Generation Agent",
        tool_used="Fraud Rule Generation Tool",
        thought="To generate a concise fraud detection rule based on the identified pattern."
    )
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
    log_agent_action(
        agent_name="Fraud Data Insight Agent",
        tool_used="Fraud Data Frame Summary Tool",
        thought="To provide an initial summary of the dataset and suggest potential fraud patterns and follow-up questions."
    )

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

def load_and_process_file(uploaded_file):
    """
    Reads the uploaded file (CSV or XLSX), processes it, and returns a single primary DataFrame.
    """
    primary_df = None
    status_message = ""
    is_success = False

    if uploaded_file.name.endswith('.csv'):
        try:
            primary_df = pd.read_csv(uploaded_file)
            status_message = "CSV file loaded successfully."
            is_success = True
        except Exception as e:
            status_message = f"Error loading CSV file: {e}"
            st.error(status_message)

    elif uploaded_file.name.endswith('.xlsx'):
        try:
            all_sheets_df = pd.read_excel(uploaded_file, sheet_name=None)
            df_customers = all_sheets_df.get('Customers')
            df_accounts = all_sheets_df.get('Accounts')
            df_events = all_sheets_df.get('Events')
            df_transactions = all_sheets_df.get('Transactions')

            if all(df is not None for df in [df_customers, df_accounts, df_events, df_transactions]):
                df_merged_customer_accounts = pd.merge(df_customers, df_accounts, on='customer_id', how='inner')
                df_merged_customer_accounts_events = pd.merge(df_merged_customer_accounts, df_events, on='customer_id', how='inner')
                final_df = pd.merge(df_merged_customer_accounts_events, df_transactions, on='customer_id', how='inner')

                primary_df = final_df
                status_message = "Successfully created a single combined DataFrame from XLSX sheets."
                is_success = True
            else:
                status_message = "One or more expected sheets (Customers, Accounts, Events, Transactions) were not found in the uploaded XLSX file. Please ensure all necessary sheets are present."
                st.error(status_message)
        except Exception as e:
            status_message = f"An error occurred during XLSX file processing or merging: {e}"
            st.error(status_message)
    else:
        status_message = "Unsupported file type. Please upload a CSV or XLSX file."
        st.error(status_message)

    return primary_df, status_message, is_success

# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide")
    st.title("🔍 Fraud Pattern Explorer") # Added from fraud_skeleton.py
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []

    left, right = st.columns([3,7])

    with left:
        st.header("Fraud Data Analysis Agent")
        file = st.file_uploader("Choose CSV/XLSX", type=["csv","xlsx"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                primary_df, status_msg, success_flag = load_and_process_file(file)
                # Ensure column names match for compatibility with existing UI elements in the right pane
                if primary_df is not None:
                    # Rename columns to match the expected names in fraud_skeleton.py's filtering and display
                    # This is a critical step for integrating the two parts
                    column_mapping = {
                        'customer_id': 'Customer ID',
                        'transaction_amount': 'Txn Amount',
                        'device_type': 'Device',
                        'ip_address': 'IP Address',
                        'fraud_type': 'Fraud Type',
                        'event_timestamp': 'Time', # Assuming event_timestamp or similar from events table
                        'event_sequence': 'Story' # Assuming a sequence of events forms the 'Story'
                    }
                    primary_df.rename(columns=column_mapping, inplace=True)

                    # Ensure 'Time' column is datetime
                    if 'Time' in primary_df.columns:
                        primary_df['Time'] = pd.to_datetime(primary_df['Time'], errors='coerce')

                    st.session_state.df = primary_df # Store the single DataFrame
                    st.session_state.current_file = file.name
                    st.session_state.messages = []
                    st.session_state.feedback_log = [] # Reset feedback for new file
                    with st.spinner("Generating dataset insights …"):
                        st.session_state.insights = FraudDataInsightAgent(st.session_state.df)
                else:
                    st.error(status_msg)
            if "df" in st.session_state and st.session_state.df is not None:
                st.dataframe(st.session_state.df.head())
                st.markdown("### Dataset Insights")
                st.markdown(st.session_state.insights)
            else:
                st.info("Upload a CSV to begin analyzing data for fraud pattern.")
        else:
            st.info("Upload a CSV/XLSX file to begin analyzing data for fraud pattern.")


    with right:
        st.header("Chat with your data for fraud insights.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Streamlit UI elements from fraud_skeleton.py start here (filters, summary, visualizations)
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df # Use the processed DataFrame

            st.sidebar.header("Filters")
            # Ensure columns exist before using them for filters
            customer_ids = df['Customer ID'].dropna().unique() if 'Customer ID' in df.columns else []
            selected_customer = st.sidebar.selectbox("Select Customer ID", options=["All"] + sorted(customer_ids.astype(str)))

            fraud_types = df["Fraud Type"].dropna().unique() if 'Fraud Type' in df.columns else []
            selected_fraud = st.sidebar.selectbox("Select Fraud Type", options=["All"] + sorted(fraud_types))

            min_date = df['Time'].min().date() if 'Time' in df.columns and not df['Time'].empty else None
            max_date = df['Time'].max().date() if 'Time' in df.columns and not df['Time'].empty else None

            start_date = st.sidebar.date_input("Start Date", min_value=min_date, value=min_date)
            end_date = st.sidebar.date_input("End Date", min_value=min_date, value=max_date)

            start_datetime = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
            end_datetime = pd.to_datetime(end_date) if end_date else pd.Timestamp.max

            filtered_df = df.copy()
            if selected_customer != "All":
                filtered_df = filtered_df[filtered_df["Customer ID"].astype(str) == selected_customer]
            if selected_fraud != "All":
                filtered_df = filtered_df[filtered_df["Fraud Type"] == selected_fraud]
            if 'Time' in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df['Time'] >= start_datetime) & (filtered_df['Time'] <= end_datetime)]

            st.subheader("Summary")
            st.metric(label="Total Fraud Cases", value=len(filtered_df))

            # "Generate Summary of Common Patterns" now uses the agent for LLM explanation
            if st.checkbox("🧠 Generate Summary of Common Patterns"):
                if 'Story' in filtered_df.columns and not filtered_df['Story'].empty:
                    top_patterns = filtered_df['Story'].value_counts().head(3).index.tolist()
                    summaries = []
                    for p in top_patterns:
                        try:
                            # Assuming 'Story' contains string representations of lists, e.g., "['event1', 'event2']"
                            events = eval(p)
                            if isinstance(events, list):
                                # Use FraudReasoningAgent to explain the pattern
                                with st.spinner(f"Analyzing pattern: {p[:50]}..."):
                                    _, explanation, pattern_desc = FraudReasoningAgent(f"Explain the fraud pattern: {events}", "N/A")
                                    # Use FraudRuleGenerationAgent to get a rule
                                    rule = FraudRuleGenerationAgent(pattern_desc) if pattern_desc else "No rule generated."
                                    summaries.append((p, f"Explanation: {explanation}\n\nSuggested Rule: {rule}"))
                            else:
                                summaries.append((p, "Story format is not a valid list"))
                        except Exception as e:
                            summaries.append((p, f"Error processing story: {e}"))
                    for i, (pattern, explanation) in enumerate(summaries):
                        st.markdown(f"**Pattern {i+1}:** `{pattern}`")
                        st.write(explanation)
                else:
                    st.info("No 'Story' column found or no data to summarize common patterns.")


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
                if 'Txn Amount' in filtered_df.columns and 'Fraud Type' in filtered_df.columns:
                    st.markdown("**Box Plot - Txn Amount by Fraud Type**")
                    fig3 = px.box(filtered_df, x='Fraud Type', y='Txn Amount', title="Transaction Amount Distribution by Fraud Type")
                    st.plotly_chart(fig3)

                if 'Device' in filtered_df.columns and 'Fraud Type' in filtered_df.columns:
                    st.markdown("**Device Usage by Fraud Type**")
                    fig4 = px.histogram(filtered_df, x='Device', color='Fraud Type', barmode='group', title="Device Usage by Fraud Type")
                    st.plotly_chart(fig4)

                if 'IP Address' in filtered_df.columns:
                    st.markdown("**Top IP Addresses Involved in Fraud**")
                    top_ips = filtered_df['IP Address'].value_counts().head(10).reset_index()
                    top_ips.columns = ['IP Address', 'Count']
                    fig5 = px.bar(top_ips, x='IP Address', y='Count', title="Top 10 IPs in Fraudulent Transactions")
                    st.plotly_chart(fig5)

            st.subheader("Fraudulent Transactions")
            for index, row in filtered_df.iterrows():
                with st.expander(f"🧾 Customer ID: {row.get('Customer ID', 'N/A')} | {row.get('Fraud Type', 'N/A')} | ${row.get('Txn Amount', 'N/A')}"):
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
                                    # Use FraudReasoningAgent and FraudRuleGenerationAgent
                                    _, explanation, pattern_desc = FraudReasoningAgent(f"Explain the fraud pattern: {events}", "N/A")
                                    rule = FraudRuleGenerationAgent(pattern_desc) if pattern_desc else "No rule generated."
                                    st.markdown("**LLM Explanation and Suggested Rule:**")
                                    st.write(f"{explanation}\n\nSuggested Rule: {rule}")
                        except:
                            st.warning("Story format is not a valid list. Please ensure 'Story' column contains string representations of lists, e.g., \"['event1', 'event2']\"")

            # Chat interface for LLM interaction
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"], unsafe_allow_html=True)
                        if msg.get("plot_index") is not None:
                            idx = msg["plot_index"]
                            if 0 <= idx < len(st.session_state.plots):
                                plot_obj = st.session_state.plots[idx]
                                if isinstance(plot_obj, (plt.Figure, plt.Axes)):
                                    st.pyplot(plot_obj, use_container_width=False)
                                elif isinstance(plot_obj, px.Figure):
                                    st.plotly_chart(plot_obj)
                        if msg.get("rule_suggestion"):
                            st.markdown(f"**Suggested Rule:** `{msg['rule_suggestion']}`")
                            col1, col2 = st.columns([0.1, 0.9])
                            with col1:
                                if st.button("👍 Useful", key=f"useful_{msg['message_id']}"):
                                    st.session_state.feedback_log.append({
                                        "query": msg.get("user_query", ""),
                                        "rule": msg['rule_suggestion'],
                                        "feedback": "Useful",
                                        "timestamp": pd.Timestamp.now()
                                    })
                                    st.toast("Feedback recorded: Useful!")
                            with col2:
                                if st.button("👎 Not Useful", key=f"not_useful_{msg['message_id']}"):
                                    st.session_state.feedback_log.append({
                                        "query": msg.get("user_query", ""),
                                        "rule": msg['rule_suggestion'],
                                        "feedback": "Not Useful",
                                        "timestamp": pd.Timestamp.now()
                                    })
                                    st.toast("Feedback recorded: Not Useful!")

            st.subheader("💬 Ask a question to the Fraud Assistant")
            st.markdown("**Suggested Questions:**")
            for q in [
                "What are the most common fraud devices?",
                "Which IP addresses are used most often in fraud?",
                "What pattern is common among high-value frauds?"
            ]:
                # Use a unique key for each button if they are generated dynamically
                if st.button(q, key=f"suggested_q_{q.replace(' ', '_').replace('?', '')}"):
                    user_q = q
                    st.session_state.messages.append({"role": "user", "content": user_q})
                    with st.spinner("Working …"):
                        code, should_plot_flag, code_thinking = FraudCodeGenerationAgent(user_q, st.session_state.df)
                        result_obj = FraudExecutionAgent(code, st.session_state.df, should_plot_flag)
                        raw_thinking, reasoning_txt, pattern_description = FraudReasoningAgent(user_q, result_obj)

                        rule_suggestion = ""
                        if pattern_description:
                            rule_suggestion = FraudRuleGenerationAgent(pattern_description)

                    is_plot = isinstance(result_obj, (plt.Figure, plt.Axes, px.Figure)) # Added px.Figure
                    plot_idx = None
                    if is_plot:
                        if isinstance(result_obj, (plt.Figure, plt.Axes)):
                            fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                            st.session_state.plots.append(fig)
                        elif isinstance(result_obj, px.Figure):
                            st.session_state.plots.append(result_obj) # Plotly figure can be stored directly
                        plot_idx = len(st.session_state.plots) - 1

                    thinking_html = ""
                    if raw_thinking:
                        thinking_html = (
                            '<details class="thinking">'
                            '<summary>🧠 Reasoning</summary>'
                            f'<pre>{raw_thinking}</pre>'
                            '</details>'
                        )
                    explanation_html = reasoning_txt
                    if pattern_description:
                        explanation_html += f"\n\n**Pattern Detected:** {pattern_description}"

                    code_html = (
                        '<details class="code">'
                        '<summary>View code</summary>'
                        '<pre><code class="language-python">'
                        f'{code}'
                        '</code></pre>'
                        '</details>'
                    )
                    assistant_msg_content = f"{thinking_html}{explanation_html}\n\n{code_html}"
                    message_id = len(st.session_state.messages)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_msg_content,
                        "plot_index": plot_idx,
                        "rule_suggestion": rule_suggestion if rule_suggestion else None,
                        "user_query": user_q,
                        "message_id": message_id
                    })
                    st.rerun() # Rerun to update chat display

            user_input = st.text_input("Type your question here", key="user_input_text_box")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner("Working …"):
                    code, should_plot_flag, code_thinking = FraudCodeGenerationAgent(user_input, st.session_state.df)
                    result_obj = FraudExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt, pattern_description = FraudReasoningAgent(user_input, result_obj)

                    rule_suggestion = ""
                    if pattern_description:
                        rule_suggestion = FraudRuleGenerationAgent(pattern_description)

                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes, px.Figure)) # Added px.Figure
                plot_idx = None
                if is_plot:
                    if isinstance(result_obj, (plt.Figure, plt.Axes)):
                        fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                        st.session_state.plots.append(fig)
                    elif isinstance(result_obj, px.Figure):
                        st.session_state.plots.append(result_obj)
                    plot_idx = len(st.session_state.plots) - 1

                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )
                explanation_html = reasoning_txt
                if pattern_description:
                    explanation_html += f"\n\n**Pattern Detected:** {pattern_description}"

                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                assistant_msg_content = f"{thinking_html}{explanation_html}\n\n{code_html}"
                message_id = len(st.session_state.messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg_content,
                    "plot_index": plot_idx,
                    "rule_suggestion": rule_suggestion if rule_suggestion else None,
                    "user_query": user_input,
                    "message_id": message_id
                })
                st.rerun()

        # Display Feedback Log (Optional, for demo purposes)
        if st.session_state.feedback_log:
            st.sidebar.markdown("### Feedback Log")
            for entry in st.session_state.feedback_log:
                st.sidebar.write(f"- **Query:** {entry['query']}")
                st.sidebar.write(f"  **Rule:** `{entry['rule']}`")
                st.sidebar.write(f"  **Feedback:** {entry['feedback']} @ {entry['timestamp'].strftime('%H:%M:%S')}")
                st.sidebar.markdown("---")
        else:
            if "df" not in st.session_state or st.session_state.df is None:
                st.info("Upload a CSV/XLSX file to begin analyzing data for fraud pattern and chat with the assistant.")


if __name__ == "__main__":
    main()