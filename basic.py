import pandas as pd
from pandasai.llm import OpenAI # Or whatever LLM you're using (e.g., GoogleGenerativeAI)
from pandasai import SmartDataframe # <-- CORRECT IMPORT FOR PANDASAI 2.x

# --- Your LLM setup (example for OpenAI) ---
# IMPORTANT: Replace "YOUR_OPENAI_API_KEY" with your actual key or an environment variable
# If using an environment variable, it would be: os.environ.get("OPENAI_API_KEY")
llm = OpenAI(api_token="")

# --- Your DataFrame ---
df = pd.DataFrame({
    'country': ['United States', 'United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'Canada', 'Australia', 'Japan', 'China'],
    'gdp': [19294482071571, 2823075598683, 2699870257270, 3677437012809, 1943867515121, 1393335222070, 1700208151477, 1399709292896, 4972427027581, 13368060851493],
    'happiness_index': [7.0, 6.7, 6.5, 7.0, 6.0, 6.3, 7.3, 7.2, 5.9, 5.0]
})

# --- Use SmartDataframe to chat with your DataFrame ---
# Pass the LLM instance via the config dictionary
sdf = SmartDataframe(df, config={"llm": llm})

# --- Make your chat query ---
response = sdf.chat("What is the sum of the gdp of the top 5 countries?")
print(response)