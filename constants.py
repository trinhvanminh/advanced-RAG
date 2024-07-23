from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

llm_map = {
    "gemini": GoogleGenerativeAI(model='gemini-1.5-pro', temperature=0),
    "cohere": ChatCohere(model="command-r", temperature=0),
    "groq": ChatGroq(model_name="llama3-70b-8192", temperature=0),
    "chatgpt": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    "ollamma": ChatOllama(model="llama3", temperature=0),
}

llm_label_map = {
    "gemini": 'Gemini (gemini-1.5-pro)',
    "cohere": 'Cohere (command-r)',
    "groq": 'Groq (llama3-70b-8192)',
    "chatgpt": "OpenAI (gpt-3.5-turbo)",
    "ollamma": "Ollama (llama3)"
}
