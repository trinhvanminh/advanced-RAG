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
#     "don't know. Answer the question and provide additional helpful information, "
#     "based on the pieces of information, if applicable. Be concise."
#     "\n\n"
#     "{context}"
#     "\n\n"
#     "IMPORTANT: Responses should be properly formatted to be easily read. MARKDOWN list syntax is recommended for long answer"
# )


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. The answer is the three sentences maximum and keep the answer concise."
    "Add additional helpful information if you think user needed"
    "If you don't know the answer or the context is not provided, say that you don't know."
    "\n\n"
    "Context: {context}"
    "\n\n"
    "IMPORTANT: Responses should be properly formatted to be easily read. MARKDOWN list syntax is recommended for long answers."
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
