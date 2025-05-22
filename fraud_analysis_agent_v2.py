import os, io, re
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# === Configuration ===
# It's good practice to load API key from environment variables
api_key = os.environ.get("NVIDIA_API_KEY") 

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# ------------------ FraudQueryUnderstandingTool ---------------------------
def FraudQueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    messages = [
        {"role": "system", "content": "detailed thinking off. You are an assistant that determines if a query is requesting a data visualization. Respond with only 'true' if the query is asking for a plot, chart, graph, or any visual representation of data. Otherwise, respond with 'false'."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=5  # We only need a short response
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

# === FraudCodeGeneration TOOLS ============================================

# ------------------ FraudPlotCodeGeneratorTool ---------------------------
def FraudPlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+matplotlib code for a plot based on the query and columns, with a fraud focus."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    3. Create only ONE relevant plot, focusing on potential fraud patterns. Set `figsize=(8,5)`, add a clear title and labels.
    4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    """

# ------------------ FraudCodeWritingTool ---------------------------------
def FraudCodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting), with a fraud focus."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only, specifically looking for fraud-related insights.
    2. Assign the final result to `result`.
    3. Wrap the snippet in a single ```python code fence (no extra prose).
    """

# === FraudCodeGenerationAgent ==============================================

def FraudCodeGenerationAgent(query: str, df: pd.DataFrame):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""
    should_plot = FraudQueryUnderstandingTool(query)
    prompt = FraudPlotCodeGeneratorTool(df.columns.tolist(), query) if should_plot else FraudCodeWritingTool(df.columns.tolist(), query)

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert specializing in **fraud detection**. Write clean, efficient code that identifies and highlights potential fraudulent patterns. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# === FraudExecutionAgent ====================================================

def FraudExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    env = {"pd": pd, "df": df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # Set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {"__builtins__": None}, env) # Limit builtins for security
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"

# === FraudReasoningCurator TOOL =========================================
def FraudReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result, focusing on fraud implications."""
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
        desc = str(result)[:500] # Increased for more detail if needed

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Analyze this chart from a **fraud detection perspective**. Explain in 3-5 concise sentences what the chart shows and **what potential fraud patterns or anomalies it might indicate**.
        '''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Analyze this result from a **fraud detection perspective**. Explain in 3-5 concise sentences what this tells about the data and **its implications for identifying fraud**.
        '''
    return prompt

# === FraudReasoningAgent (streaming) =========================================
def FraudReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation, with a fraud focus."""
    prompt = FraudReasoningCurator(query, result)
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    # Streaming LLM call
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst specializing in **fraud detection**."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )

    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
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
    return thinking_content, cleaned

# === FraudDataFrameSummary TOOL (pandas only) =========================================
def FraudDataFrameSummaryTool(df: pd.DataFrame) -> str:
    """
    Generate a summary prompt string for the LLM based on the DataFrame,
    with a strong focus on identifying potential fraud insights and patterns.
    """
    
    # --- Basic DataFrame Information ---
    num_rows = len(df)
    num_cols = len(df.columns)
    column_names = ', '.join(df.columns)
    data_types = df.dtypes.apply(lambda x: str(x)).to_dict() # Ensure string representation for dict keys
    missing_values = df.isnull().sum().to_dict()

    # --- Fraud-Specific Aggregations and Insights ---
    fraud_insights = {}

    # Identify common columns for fraud analysis, making them case-insensitive
    cols_lower = [col.lower() for col in df.columns]
    
    # Transaction Amount Analysis
    transaction_amount_col = next((col for col in df.columns if col.lower() in ['transaction_amount', 'amount', 'value']), None)
    if transaction_amount_col and pd.api.types.is_numeric_dtype(df[transaction_amount_col]):
        fraud_insights['transaction_amount_stats'] = {
            'mean': df[transaction_amount_col].mean(),
            'median': df[transaction_amount_col].median(),
            'max': df[transaction_amount_col].max(),
            'min': df[transaction_amount_col].min(),
            'std': df[transaction_amount_col].std(),
            'Q1': df[transaction_amount_col].quantile(0.25),
            'Q3': df[transaction_amount_col].quantile(0.75)
        }
        # Highlight potential outliers for fraud (e.g., very high/low transactions)
        # Using IQR for more robust outlier detection
        Q1 = df[transaction_amount_col].quantile(0.25)
        Q3 = df[transaction_amount_col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_upper_bound = Q3 + 1.5 * IQR
        outlier_lower_bound = Q1 - 1.5 * IQR

        high_value_transactions = df[df[transaction_amount_col] > outlier_upper_bound]
        low_value_transactions = df[df[transaction_amount_col] < outlier_lower_bound]
        
        if not high_value_transactions.empty:
            fraud_insights['high_value_transaction_count'] = len(high_value_transactions)
            fraud_insights['high_value_transaction_sample_max'] = high_value_transactions[transaction_amount_col].max()
            fraud_insights['high_value_transaction_sample_min'] = high_value_transactions[transaction_amount_col].min()

        if not low_value_transactions.empty:
            fraud_insights['low_value_transaction_count'] = len(low_value_transactions)
            fraud_insights['low_value_transaction_sample_max'] = low_value_transactions[transaction_amount_col].max()
            fraud_insights['low_value_transaction_sample_min'] = low_value_transactions[transaction_amount_col].min()

    # User/Account Activity Analysis
    user_id_col = next((col for col in df.columns if col.lower() in ['user_id', 'customer_id', 'account_id']), None)
    if user_id_col:
        user_transaction_counts = df[user_id_col].value_counts()
        # Users with unusually high number of transactions
        frequent_users_threshold = user_transaction_counts.quantile(0.99) # Top 1% by transaction count
        frequent_users = user_transaction_counts[user_transaction_counts >= frequent_users_threshold]
        if not frequent_users.empty:
            fraud_insights['frequent_users_count'] = len(frequent_users)
            fraud_insights['frequent_users_max_transactions'] = frequent_users.max()
            fraud_insights['frequent_users_min_transactions'] = frequent_users.min()
        
        # Check for single-use accounts or very low transaction counts (might indicate test accounts for fraud)
        sparse_users = user_transaction_counts[user_transaction_counts <= 2] # Users with 1 or 2 transactions
        if not sparse_users.empty:
            fraud_insights['sparse_users_count'] = len(sparse_users)

    # Time-based Analysis
    timestamp_col = next((col for col in df.columns if col.lower() in ['timestamp', 'transaction_time', 'datetime']), None)
    if timestamp_col:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df.dropna(subset=[timestamp_col], inplace=True) # Drop rows where conversion failed
            if not df.empty:
                df_sorted = df.sort_values(by=timestamp_col)
                time_diffs = df_sorted[timestamp_col].diff().dt.total_seconds().dropna()
                if not time_diffs.empty:
                    fraud_insights['inter_transaction_time_stats'] = {
                        'mean_seconds': time_diffs.mean(),
                        'min_seconds': time_diffs.min(),
                        'max_seconds': time_diffs.max(),
                        'median_seconds': time_diffs.median()
                    }
                    # Highlight very short time differences (rapid fire transactions)
                    short_time_transactions_count = time_diffs[time_diffs < 10].count() # Transactions within 10 seconds
                    if short_time_transactions_count > 0:
                        fraud_insights['very_short_inter_transaction_count'] = short_time_transactions_count
                    
                    # Transactions at unusual hours (e.g., late night/early morning for user's timezone if applicable)
                    # This requires timezone info or a heuristic. For simplicity, we can look at global hours.
                    df['hour_of_day'] = df[timestamp_col].dt.hour
                    unusual_hours_transactions = df[(df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 5)].shape[0] # 0-5 AM
                    if unusual_hours_transactions > 0:
                        fraud_insights['transactions_in_unusual_hours_count'] = unusual_hours_transactions

        except Exception as e:
            # Handle cases where timestamp conversion fails or column is not appropriate
            print(f"Warning: Could not process timestamp column '{timestamp_col}' for fraud insights. Error: {e}")

    # Location-based Analysis (if applicable)
    location_cols = [col for col in df.columns if col.lower() in ['country', 'state', 'ip_address', 'city']]
    if location_cols:
        for loc_col in location_cols:
            if df[loc_col].nunique() > 1: # Only if there's more than one unique value
                top_locations = df[loc_col].value_counts().head(5).to_dict()
                fraud_insights[f'top_5_{loc_col}_counts'] = top_locations
                # Could add logic for single IP with many users, or unusual geo combinations

    # Device/Browser/OS Analysis (if applicable)
    device_cols = [col for col in df.columns if col.lower() in ['device_type', 'browser', 'os']]
    if device_cols:
        for dev_col in device_cols:
            top_devices = df[dev_col].value_counts().head(3).to_dict()
            if not top_devices.empty:
                fraud_insights[f'top_3_{dev_col}_counts'] = top_devices
            # Could add logic for single device with many accounts/transactions

    # --- Construct the Prompt ---
    prompt = f"""
    You are an AI specialized in **fraud detection and anomaly analysis**.
    
    Given a dataset related to transactions or user activities, with {num_rows} rows and {num_cols} columns:
    
    ---
    ### **Dataset Overview:**
    - **Columns:** {column_names}
    - **Data types:** {data_types}
    - **Missing values per column:** {missing_values}

    ---
    ### **Key Statistical Aggregations and Potential Fraud Indicators:**
    """
    
    if fraud_insights:
        for key, value in fraud_insights.items():
            prompt += f"- **{key.replace('_', ' ').title()}:** "
            if isinstance(value, dict):
                sub_items = []
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        sub_items.append(f"{sub_key.replace('_', ' ').title()}: {sub_value:.2f}")
                    else:
                        sub_items.append(f"{sub_key.replace('_', ' ').title()}: {sub_value}")
                prompt += ", ".join(sub_items) + "\n"
            else:
                prompt += f"{value}\n"
    else:
        prompt += "- *No specific fraud-related aggregations could be generated based on available columns. Please ensure relevant columns like 'transaction_amount', 'user_id', 'timestamp', 'ip_address', or 'device_type' are present and appropriately named.*"

    prompt += f"""
    ---
    ### **Fraud Analysis Directives:**
    Based on the provided information and considering common fraud patterns, please provide:
    1.  **A concise description of the dataset from a fraud detection perspective**, highlighting any immediate red flags, unusual distributions, or areas of concern that warrant further investigation for fraud.
    2.  **3-5 high-impact data analysis questions** specifically geared towards identifying and understanding potential fraudulent activities, schemes, or anomalies within this data. These questions should be actionable and lead to specific analytical steps.
    3.  **Recommendations for additional features or data points** that would significantly enhance fraud detection capabilities for this dataset, explaining why each is important.

    Keep your response focused on actionable fraud insights and use clear, direct language.
    """
    return prompt

# === FraudDataInsightAgent (upload-time only) ===============================

def FraudDataInsightAgent(df: pd.DataFrame) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset, with a strong fraud focus."""
    prompt = FraudDataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": "detailed thinking off. You are a highly specialized fraud data analyst providing brief, focused insights and actionable questions for fraud detection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700 # Increased max tokens to allow for more detailed fraud insights
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error generating dataset insights: {exc}"

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

# === Main Streamlit App ===============================================

def main():
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []

    left, right = st.columns([3,7])

    with left:
        st.header("Fraud Data Analysis Agent")
        st.markdown("<medium>Powered by <a href='[https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1)'>NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</a></medium>", unsafe_allow_html=True)
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                # Clear chat history and plots on new file upload
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                st.session_state.plots = [] # Clear plots for new dataset
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = FraudDataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights (Fraud Focus)")
            st.markdown(st.session_state.insights)
            st.markdown("---")
            st.markdown("### Detailed Data Profile")
            profile = ProfileReport(st.session_state.df, explorative=True, title="Data Profiling Report (Fraud Context)")
            st_profile_report(profile)
        else:
            st.info("Upload a CSV to begin chatting with your data and uncover fraud patterns.")

    with right:
        st.header("Chat with your data about fraud patterns")
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

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about potential fraud patterns…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Analyzing for fraud…"):
                    code, should_plot_flag, code_thinking = FraudCodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = FraudExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = FraudReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant response
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization of potential fraud patterns:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Analysis Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Analysis Result"
                else:
                    header = f"Analysis Result: {result_obj}"

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning (Fraud Analysis)</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Code accordion with proper HTML <pre><code> syntax highlighting
                code_html = (
                    '<details class="code">'
                    '<summary>View generated code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                # Combine thinking, explanation, and code accordion
                assistant_msg = f"{header}\n\n{thinking_html}{explanation_html}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

if __name__ == "__main__":
    main()