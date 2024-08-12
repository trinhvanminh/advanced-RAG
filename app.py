from typing import Generator

import httpx
from openai import BadRequestError
import pytz
import streamlit as st
from bson.objectid import ObjectId

import src.config as cfg
from src.csv_retriever import CSVRetriever
from src.qna import QnA, QnAResponse
from src.rag import RAG
from src.utils.conversation import (create_conversation, delete_conversation,
                                    select_conversation)


def ai_response_wrapper(generator: Generator[QnAResponse, None, None]) -> Generator:
    for chunk in generator:

        # if 'context' in chunk:
        #     context = chunk['context']
        #     sources = []
        #     for doc in context:
        #         source = doc.metadata.get('source')
        #         if source not in sources:
        #             sources.append(source)
        #     print(sources)

        if 'answer' in chunk:
            yield chunk['answer']


def init_session_state(qa: QnA):
    # init session states
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_conversation" not in st.session_state:
        st.session_state.selected_conversation = ''
    else:
        select_conversation(qa, st.session_state.selected_conversation)

    if "conversations" not in st.session_state:
        history_collection = qa.get_collection('message_store')
        st.session_state.conversations = history_collection.distinct(
            "SessionId")

    if "model" not in st.session_state:
        st.session_state.model = 'azure-openai'


def render_sidebar(qa: QnA):
    with st.sidebar:
        model = st.selectbox(
            "Choose a LLM",
            tuple(cfg.llm_options.keys()),
            format_func=lambda option: cfg.llm_options[option]["label"],
            index=0
        )

        st.session_state.model = model
        st.button("New chat", on_click=create_conversation)
        st.write("Previous chat")

        # list all conversations
        with st.container(height=640, border=False):
            conversations = st.session_state.conversations
            conversations.sort(key=lambda x: str(x), reverse=True)

            for conversation in conversations:
                label_col, action_col = st.columns([6, 1])

                with label_col:
                    # label type for active/inactive conversation
                    if conversation == st.session_state.selected_conversation:
                        label_btn_type = "primary"
                    else:
                        label_btn_type = 'secondary'

                    if not isinstance(conversation, ObjectId):
                        # delete_conversation(qa, conversation)
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
                        on_click=select_conversation,
                        kwargs={"qa": qa, "session_id": conversation},
                        type=label_btn_type,
                        help=created_at
                    )

                with action_col:
                    action_col.button(
                        "🗑️",
                        key=f'delete_btn.{conversation}',
                        use_container_width=True,
                        on_click=delete_conversation,
                        kwargs={"qa": qa, "session_id": conversation}
                    )


def render_chat(qa: QnA):
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

                qa.model = cfg.llm_options[st.session_state.model].get("llm")

                conversation = st.session_state.selected_conversation or ObjectId()

                try:
                    response = qa.ask_question(
                        query=prompt,
                        session_id=conversation,
                        stream=True
                    )

                    content = st.write_stream(
                        ai_response_wrapper(response)
                    )

                    # content = st.markdown(response["answer"])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )

                    # rerun to rerender sidebar with new conversations as selected_conversation
                    if conversation not in st.session_state.conversations:
                        st.session_state.conversations.append(conversation)
                        st.session_state.selected_conversation = conversation
                        st.rerun()

                except httpx.ConnectError:
                    llm_option_label = (
                        cfg.llm_options[st.session_state.model]
                        .get("label")
                    )

                    st.warning(
                        f"Check your `{llm_option_label}` connection"
                    )
                except BadRequestError as e:
                    print(e.body['innererror']['content_filter_result'])
                    st.error(e.body['message'])
                except Exception as e:
                    st.error(e)


def main():
    st.set_page_config(page_title="Mortgage Assistant")
    st.title("Mortgage Assistant")

    default_model = cfg.llm_options['azure-openai'].get('llm')

    rag = RAG(model=default_model, rerank=cfg.rerank)

    csv_retriever = CSVRetriever(
        llm=default_model,
        directory_path='./data/preprocessed/csv/'
    )

    qa = QnA(
        model=default_model,
        retriever=rag.retriever,
        data_retriever=csv_retriever
    )

    init_session_state(qa)
    render_sidebar(qa)
    render_chat(qa)


if __name__ == "__main__":
    main()
