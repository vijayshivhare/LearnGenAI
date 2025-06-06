# In your pages/2_Data_Visualization.py or a new analytics page
import streamlit as st
import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import OpenAI
from PIL import Image # Import the Image module from PIL (Pillow)

# Your OpenAI API key (ensure it's loaded securely, e.g., from .env or st.secrets)
# from dotenv import load_dotenv
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# Or, if deployed on Streamlit Cloud:
openai_api_key = "" # Recommended for deployed apps
llm = OpenAI(api_token=openai_api_key)

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"]) # Changed accepted types

st.title("🧠 AI-Powered Data Analysis")
st.write("Ask questions about your transaction data in natural language!")


# Process uploaded file and store in session state
if uploaded_file:
    # Read the Excel and store in session state
    try:
        # Load each sheet as a DataFrame
        sheet_dict = pd.read_excel(uploaded_file, sheet_name=None)

    except Exception as e:
        st.error(f"Error reading Excel file: {e}. Please ensure it's a valid Excel format.")

    # SmartDatalake can accept multiple DataFrames with names
    datalake = SmartDatalake(list(sheet_dict.values()), config={"llm": llm,"save_charts": False})

    if datalake is not None:
        question = st.text_input("Ask a question about the transaction data (e.g., 'What is the average transaction amount?', 'Show me a histogram of Amount by Type'):")
        if question:
            with st.spinner("Thinking..."):
                try:
                    # This is where pandas-ai does its magic
                    # It will generate and execute pandas code based on your question
                    # The output can be a value, a DataFrame, or even a Matplotlib/Plotly figure
                    response = datalake.chat(question)
                    print("respose is: "+response)
                    if isinstance(response, pd.DataFrame):
                        st.dataframe(response)
                    elif isinstance(response, str) and response.endswith(".png"):
                        # If the response is a string ending with .png, it's a file path
                        try:
                            image = Image.open(response)
                            st.image(image, caption="Generated Plot")
                        except FileNotFoundError:
                            st.error(f"Plot file not found at: {response}. Please check the path.")
                    elif hasattr(response, 'figure'): # Check if it's a matplotlib plot object
                        st.pyplot(response)
                    elif hasattr(response, 'show'): # Check if it's a plotly plot object
                        st.plotly_chart(response)
                    else:
                        st.write(response) # For text answers
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please try rephrasing your question.")
else:
    st.warning("Transaction data not loaded for AI analysis.")