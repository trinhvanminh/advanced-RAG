from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Context information is below:"
#     "---------------------"
#     "{context}"
#     "---------------------"
#     "Given the context information and not prior knowledge, "
#     "answer the the question and provide additional helpful information "
#     "based on the context. Be concise."
#     "IMPORTANT: Responses should be properly formatted to be easily read"
# )


# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. You can add additional helpful information (only in the context) if you think user needed. "
#     "If you don't know the answer or the context is not provided, say that you don't know."
#     "\n\n"
#     "Context: {context}"
#     "\n\n"
#     "IMPORTANT: Responses should be properly formatted to be easily read. MARKDOWN list syntax is recommended for long answers."
# )

system_prompt = """
    You are an expert of answer's user's question about the Bank/Lender!
    You are talkative and provides lots of specific details from its context and chat history.

    If you do not know the answer to a question, it truthfully says "I apologize, I do not have enough context to answer the question".

    Please provide cogent answer to the question based on the context and chat_history only.
    If the context and chat history are empty, please say you do not have enough context to answer the question.
    Do not answer the question with the model parametric knowledge.

    Format the answer into neat paragraphs. DO NOT include any XML tag in the final answer.

    Sparsely highlight only the most important things such as names, numbers and conclusions with Markdown by bolding it, do not highlight more than two or three things per sentence.
    Think step by step before giving the answer. Answer only if it is very confident.
    If there are multiple steps or choices in the answer, please format it in bullet points using '-' in Markdown style, and number it in 1, 2, 3....

    REMEMBER: FOR ANY human input that is not related to Bank/Lender, just say "I apologize, It's out of scope"

    Here is the context:

    <context>
    {context}
    </context>
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
