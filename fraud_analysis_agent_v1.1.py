import os, io, re
import pandas as pd
import streamlit as st
# from openai import OpenAI # No longer directly needed for LLM calls after LangChain integration
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Import LangChain components
from langchain_openai import ChatOpenAI
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import yaml

# === Configuration ===
# The NVIDIA_API_KEY is used by ChatNVIDIA
# You will also need to set LangSmith specific environment variables
#os.environ["NVIDIA_API_KEY"] = ""
load_dotenv()

# Load messages from the YAML file
with open('messages.yaml', 'r') as file:
    MESSAGES = yaml.safe_load(file)

# Initialize LangChain LLM for NVIDIA models
# This will pick up NVIDIA_API_KEY automatically
#llm_model = ChatNVIDIA(model="meta/llama2-70b")
llm_model = ChatOpenAI(model_name="gpt-4o")

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
    #st.markdown(f'<b>Action:</b> {tool_used} Action Input: ["search_query":"<query_details>"]<br>' # Placeholder for tool input
    #            f'<button onclick="alert(\'Observation would be shown here!\')">Show observation</button>' # Simple JS alert for demo
    #            f'</div>', unsafe_allow_html=True)

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
    log_agent_action(
        agent_name="Fraud Code Generation Agent", # Example agent name from image
        tool_used=tool_name,
        thought="To generate the appropriate code based on the user's query, I need to determine if a plot is requested and then use the correct code generation tool."
    )
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
    log_agent_action(
        agent_name="Fraud Execution Agent", 
        tool_used="Python Interpreter/Execution Environment",
        thought="To execute the generated Python code and obtain the result, I will run it in a safe environment."
    )

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
     # Log the action of the FraudDataInsightAgent
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

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        tuple: (primary_df, status_message, is_success)
               primary_df: The main DataFrame to be used for display/analysis.
               status_message: A string indicating the status (success, warning, error).
               is_success: Boolean indicating if the operation was successful.
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
            # Read all sheets from the XLSX file into a dictionary of DataFrames
            all_sheets_df = pd.read_excel(uploaded_file, sheet_name=None)
            status_message = f"XLSX file loaded with sheets: {', '.join(all_sheets_df.keys())}"

            # Extract individual DataFrames for merging
            df_customers = all_sheets_df.get('Customers')
            df_accounts = all_sheets_df.get('Accounts')
            df_events = all_sheets_df.get('Events')
            df_transactions = all_sheets_df.get('Transactions')

            # Check if all expected sheets are present before attempting to merge
            if all(df is not None for df in [df_customers, df_accounts, df_events, df_transactions]):
                # --- Merging Logic ---
                df_merged_customer_accounts = pd.merge(
                    df_customers,
                    df_accounts,
                    on='customer_id',
                    how='inner'
                )

                df_merged_customer_accounts_events = pd.merge(
                    df_merged_customer_accounts,
                    df_events,
                    on='customer_id',
                    how='inner'
                )

                final_df = pd.merge(
                    df_merged_customer_accounts_events,
                    df_transactions,
                    on='customer_id',
                    how='inner'
                )               

                primary_df = final_df # This is the single DataFrame to be returned
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
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []

    left, right = st.columns([3,7])

    with left:
        st.header("Fraud Data Analysis Agent")
        #st.markdown("<medium>Powered by <a href='[https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1)'>NVIDIA Llama-3.1-Nemotron-Ultra-253b-v1</a></medium>", unsafe_allow_html=True)
        file = st.file_uploader("Choose CSV/XLSX", type=["csv","xlsx"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                primary_df, status_msg, success_flag = load_and_process_file(file)
                primary_df.to_csv("output.csv", index=False)
                st.session_state.df = primary_df # Store the single DataFrame
                st.session_state.current_file = file.name
                st.session_state.messages = []
                st.session_state.feedback_log = [] # Reset feedback for new file
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = FraudDataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights")
            st.markdown(st.session_state.insights)
        else:
            st.info("Upload a CSV to begin analyzing data for fraud pattern.")

    with right:
        st.header("Chat with your data for fraud insights.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)
                    if msg.get("rule_suggestion"):
                        st.markdown(f"**Suggested Rule:** `{msg['rule_suggestion']}`")
                        # Simulated feedback UI
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
                            if st.button("👎 Not Useful", key=f"not_useful_{msg['message_id']}"):
                                st.session_state.feedback_log.append({
                                    "query": msg.get("user_query", ""),
                                    "rule": msg['rule_suggestion'],
                                    "feedback": "Not Useful",
                                    "timestamp": pd.Timestamp.now()
                                })
                                st.toast("Feedback recorded: Not Useful!")


        if file:  # only allow chat after upload
            user_q = st.chat_input("Query about fraud patterns…")
            if user_q:
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    # Placeholder for Data Summarization/Feature Engineering
                    # For the hackathon, if your LLM queries hit token limits,
                    # you'd implement logic here to summarize `st.session_state.df`
                    # or extract key features for the LLM to analyze.
                    # Example:
                    # summarized_df = st.session_state.df.groupby('user_id').agg(
                    #     total_transactions=('transaction_id', 'count'),
                    #     avg_amount=('amount', 'mean'),
                    #     # ... other aggregated features
                    # ).reset_index()
                    # Then pass summarized_df to FraudCodeGenerationAgent, or just
                    # use the query to guide the LLM to perform these aggregations.

                    code, should_plot_flag, code_thinking = FraudCodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = FraudExecutionAgent(code, st.session_state.df, should_plot_flag)
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
                # Combine thinking, explanation, and code accordion
                assistant_msg_content = f"{thinking_html}{explanation_html}\n\n{code_html}"

                # Generate a unique ID for the message to tie feedback buttons to it
                message_id = len(st.session_state.messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg_content,
                    "plot_index": plot_idx,
                    "rule_suggestion": rule_suggestion if rule_suggestion else None, # Include rule suggestion
                    "user_query": user_q, # Store user query for feedback log
                    "message_id": message_id # Unique ID for feedback
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

if __name__ == "__main__":
    main()