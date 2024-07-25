import httpx
import pytz
import streamlit as st
from bson.objectid import ObjectId
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

import src.config as cfg
from src.qna import QnA


def delete_chat(qa: QnA, session_id: str):
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


def select_chat(qa: QnA, session_id: str):
    st.session_state.selected_conversation = session_id
    _load_messages(qa, session_id)


def create_chat():
    st.session_state.selected_conversation = ''
    st.session_state.messages = []


def main():
    st.set_page_config(page_title="Mortgage Assistant")
    st.title("Mortgage Assistant")

    qa = QnA(
        embeddings=cfg.embeddings,
        model=cfg.default_model,
        rerank=cfg.rerank
    )

    # init session states
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_conversation" not in st.session_state:
        st.session_state.selected_conversation = ''
    else:
        select_chat(qa, st.session_state.selected_conversation)

    if "conversations" not in st.session_state:
        history_collection = qa.get_collection('message_store')
        st.session_state.conversations = history_collection.distinct(
            "SessionId")

    if "model" not in st.session_state:
        st.session_state.model = 'cohere'

    # sidebar
    with st.sidebar:
        model = st.selectbox(
            "Choose a LLM",
            tuple(cfg.llm_label_map.keys()),
            format_func=lambda option: cfg.llm_label_map[option],
            index=1
        )

        st.session_state.model = model
        st.button("New chat", on_click=create_chat)
        st.write("Previous chat")

        with st.container(height=640, border=False):
            # list all conversations
            conversations = st.session_state.conversations
            conversations.sort(key=lambda x: str(x), reverse=True)

            for conversation in conversations:
                label_col, action_col = st.columns([6, 1])

                with label_col:
                    # label type for active/inactive conversation
                    label_btn_type = "primary" if conversation == st.session_state.selected_conversation else 'secondary'

                    if not isinstance(conversation, ObjectId):
                        # delete_chat(qa, conversation)
                        raise ValueError(
                            "Invalid `SessionId`, Expected: %s, Got: %s" % (ObjectId, type(conversation)))

                    # label tooltip
                    now_utc = conversation.generation_time
                    est_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                    created_at = now_utc.astimezone(
                        est_tz).strftime("%d/%m/%y %X")

                    # label as button
                    label_col.button(
                        str(conversation)[:8] + "..." + str(conversation)[-8:],
                        key=f'label_btn.{conversation}',
                        use_container_width=True,
                        on_click=select_chat,
                        kwargs={"qa": qa, "session_id": conversation},
                        type=label_btn_type,
                        help=created_at
                    )

                with action_col:
                    action_col.button(
                        "🗑️",
                        key=f'delete_btn.{conversation}',
                        use_container_width=True,
                        on_click=delete_chat,
                        kwargs={"qa": qa, "session_id": conversation}
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

                qa.model = cfg.llm_map.get(st.session_state.model)

                conversation = st.session_state.selected_conversation or ObjectId()

                try:
                    response = qa.ask_question(
                        query=prompt,
                        session_id=conversation
                    )

                    if conversation not in st.session_state.conversations:
                        st.session_state.conversations.append(conversation)
                        st.session_state.selected_conversation = conversation
                        st.rerun()

                    content = response.get('answer')

                    st.markdown(content)

                except httpx.ConnectError:
                    st.warning(
                        f"Check your {st.session_state.model} connection")
                    return

        st.session_state.messages.append(
            {"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
