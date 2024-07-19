from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere

llm_map = {
    "Gemini (1.5-pro)": GoogleGenerativeAI(model='gemini-1.5-pro', temperature=0),
    "Cohere (command-r)": ChatCohere(model="command-r", temperature=0),
    "Groq (llama3-70b-8192)": ChatGroq(temperature=0, model_name="llama3-70b-8192")
}
