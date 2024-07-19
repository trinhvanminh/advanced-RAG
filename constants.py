from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere

llm_map = {
    "gemini": GoogleGenerativeAI(model='gemini-1.5-pro', temperature=0),
    "cohere": ChatCohere(model="command-r", temperature=0),
    "groq": ChatGroq(temperature=0, model_name="llama3-70b-8192")
}


llm_label_map = {
    "gemini": 'Gemini (1.5-pro)',
    "cohere": 'Cohere (command-r)',
    "groq": 'Groq (llama3-70b-8192)',
}
