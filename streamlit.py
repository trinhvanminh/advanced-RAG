import uuid
import streamlit as st
from app import rag
from constants import llm_map
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

st.set_page_config(page_title="Mortgage Assistant")

st.title("Mortgage Assistant")


def delete_chat(session_id: str):
    session_history = rag.get_session_history(session_id=session_id)
    session_history.clear()


def _parse_llm_messages(messages: list[HumanMessage | AIMessage]):
    parsed_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            parsed_messages.append({
                "role": "user",
                "content": message.content
            })
        else:
            parsed_messages.append({
                "role": "assistant",
                "content": message.content
            })
    return parsed_messages


def load_messages(session_id: str):
    print("load_messages", session_id)
    session_history = rag.get_session_history(session_id=session_id)
    parsed_messages = _parse_llm_messages(session_history.messages)
    print("parsed_messages", parsed_messages)
    st.session_state.messages = parsed_messages


def select_chat(session_id: str):
    print("select_chat", session_id)
    st.session_state.selected_session_id = session_id
    load_messages(session_id)


def create_chat():
    session_id = str(uuid.uuid4())
    select_chat(session_id)
    st.session_state.temp_session_id = session_id


# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

if "selected_session_id" not in st.session_state:
    st.session_state.selected_session_id = ''

if "temp_session_id" not in st.session_state:
    st.session_state.temp_session_id = ''

if "model" not in st.session_state:
    st.session_state.model = 'Gemini (1.5-pro)'


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:

    model = st.selectbox(
        "Choose a LLM",
        ("Gemini (1.5-pro)", "Cohere (command-r)", "Groq (llama3-70b-8192)")
    )

    st.session_state.model = model

    st.button("New chat", on_click=create_chat)

    st.write("Previous chat")
    with st.container(height=640, border=False):
        # list all history
        history_collection = rag.collection('history')
        session_ids = history_collection.distinct("SessionId")

        # chat selection
        selected_session_id = st.session_state.selected_session_id
        if selected_session_id:
            if selected_session_id not in session_ids:
                session_ids.append(selected_session_id)
        # else:
        #     print("chat selection")
        #     select_chat(session_ids[0])

        # display chats
        sorted_session_ids = sorted(
            session_ids, key=lambda x: uuid.UUID(x).time, reverse=True)
        for session_id in sorted_session_ids:
            label_col, action_col = st.columns([6, 1])

            label_btn_type = "primary" if session_id == st.session_state.selected_session_id else 'secondary'

            with label_col:
                label_btn = label_col.button(
                    session_id[:8] + "..." + session_id[-8:],
                    key=session_id,
                    use_container_width=True,
                    on_click=select_chat,
                    kwargs={"session_id": session_id},
                    type=label_btn_type
                )

            with action_col:
                delete_btn = action_col.button(
                    "🗑️",
                    key=f'delete_btn{session_id}',
                    use_container_width=True,
                    on_click=delete_chat,
                    kwargs={"session_id": session_id}
                )


# Accept user input
if prompt := st.chat_input("Ask questions"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Loading..."):
        with st.chat_message("assistant"):
            rag.llm = llm_map.get(model)
            chain = rag.conversational_rag_chain
            response = chain.invoke(
                input={"input": prompt},
                config={"configurable": {
                    "session_id": st.session_state.selected_session_id}}
            )

            content = response.get('answer') or response.get('result')

            st.markdown(content)

    load_messages(st.session_state.selected_session_id)
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": content})
