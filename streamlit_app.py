import uuid
import streamlit as st
from app import rag
from constants import llm_map, llm_label_map
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

st.set_page_config(page_title="Mortgage Assistant")

st.title("Mortgage Assistant")


def delete_chat(session_id: str):
    session_history = rag.get_session_history(session_id=session_id)
    session_history.clear()

    st.session_state.messages = []
    st.session_state.conversations = list(
        filter(
            lambda x: x != session_id,
            st.session_state.conversations
        )
    )

    if session_id == st.session_state.selected_conversation:
        st.session_state.selected_conversation = ''


def _parse_llm_messages(messages: list[HumanMessage | AIMessage]):
    return [{"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} for msg in messages]


def load_messages(session_id: str):
    session_history = rag.get_session_history(session_id=session_id)
    st.session_state.messages = _parse_llm_messages(session_history.messages)


def select_chat(session_id: str):
    st.session_state.selected_conversation = session_id
    load_messages(session_id)


def create_chat():
    st.session_state.selected_conversation = ''
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_conversation" not in st.session_state:
    st.session_state.selected_conversation = ''
else:
    select_chat(st.session_state.selected_conversation)

if "conversations" not in st.session_state:
    history_collection = rag.collection('message_store')
    st.session_state.conversations = history_collection.distinct("SessionId")

if "model" not in st.session_state:
    st.session_state.model = 'cohere'


with st.sidebar:
    model = st.selectbox(
        "Choose a LLM",
        ("gemini", "cohere", "groq"),
        format_func=lambda option: llm_label_map[option],
        index=1
    )

    st.session_state.model = model
    st.button("New chat", on_click=create_chat)
    st.write("Previous chat")

    with st.container(height=640, border=False):
        # list all history
        session_ids = st.session_state.conversations

        # TODO: sort by timestamp
        # display chats
        #  sorted_session_ids = sorted(
        #     session_ids, key=lambda x: uuid.UUID(x).time, reverse=True)

        sorted_session_ids = session_ids
        for session_id in sorted_session_ids:
            label_col, action_col = st.columns([6, 1])

            label_btn_type = "primary" if session_id == st.session_state.selected_conversation else 'secondary'

            with label_col:
                label_col.button(
                    session_id[:8] + "..." + session_id[-8:],
                    key=f'label_btn.{session_id}',
                    use_container_width=True,
                    on_click=select_chat,
                    kwargs={"session_id": session_id},
                    type=label_btn_type
                )

            with action_col:
                action_col.button(
                    "🗑️",
                    key=f'delete_btn.{session_id}',
                    use_container_width=True,
                    on_click=delete_chat,
                    kwargs={"session_id": session_id}
                )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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

            rag.llm = llm_map.get(st.session_state.model)
            chain = rag.conversational_rag_chain

            session_id = st.session_state.selected_conversation or str(
                uuid.uuid4())

            response = chain.invoke(
                input={"input": prompt},
                config={
                    "configurable": {
                        "session_id": session_id,
                    }
                },
            )

            if session_id not in st.session_state.conversations:
                st.session_state.conversations.append(session_id)
                st.session_state.selected_conversation = session_id
                st.rerun()

            content = response.get('answer')

            st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
