from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_fireworks import ChatFireworks

llm_map = {
    "chatgpt": ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    "gemini": ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0),
    "cohere": ChatCohere(model="command-r-plus", temperature=0),
    "ollamma": ChatOllama(model="llama3", temperature=0),
    "groq": ChatGroq(model_name="llama3-70b-8192", temperature=0),
    "fireworks": ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=0)
}

llm_label_map = {
    # "chatgpt": "OpenAI (gpt-3.5-turbo)",
    "gemini": 'Gemini (gemini-1.5-pro)',
    "cohere": 'Cohere (command-r-plus)',
    "ollamma": "Ollama (llama3)",
    "groq": 'Groq (llama3-70b-8192)',
    "fireworks": "Fireworks (llama-v3-70b-instruct)"
}
