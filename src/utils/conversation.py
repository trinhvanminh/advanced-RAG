import streamlit as st
from src.qna import QnA
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage


def delete_conversation(qa: QnA, session_id: str):
    session_history = qa.get_session_history(session_id=session_id)
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


def _load_messages(qa: QnA, session_id: str):
    session_history = qa.get_session_history(session_id=session_id)
    st.session_state.messages = _parse_llm_messages(session_history.messages)


def select_conversation(qa: QnA, session_id: str):
    st.session_state.selected_conversation = session_id
    _load_messages(qa, session_id)


def create_conversation():
    st.session_state.selected_conversation = ''
    st.session_state.messages = []
