import os
from typing import Generator

import logging
import httpx
from openai import BadRequestError
import pytz
import streamlit as st
from bson.objectid import ObjectId
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import Client
from langchain.schema.runnable import RunnableConfig
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

from src.csv_retriever import CSVRetriever
from src.qna import QnA, QnAResponse
from src.rag import RAG
from src.utils.conversation import (create_conversation, delete_conversation,
                                    select_conversation)

import src.config as cfg
import src.constants as c

logger = logging.getLogger(__name__)


def set_up_langsmith_env():
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


def ai_response_wrapper(generator: Generator[QnAResponse, None, None]) -> Generator:
    for chunk in generator:
        if 'answer' in chunk:
            yield chunk['answer'].replace("$", "\$")


def _reset_feedback():
    st.session_state.feedback_update = None
    st.session_state.feedback = None


def _get_trace_link(client: Client, run_collector: RunCollectorCallbackHandler):
    # The run collector will store all the runs in order. We'll just take the root and then
    # reset the list for next interaction.
    run = run_collector.traced_runs[0]
    run_collector.traced_runs = []
    st.session_state.run_id = run.id
    wait_for_all_tracers()
    # Requires langsmith >= 0.0.19
    url = client.share_run(run.id)
    # Or if you just want to use this internally
    # without sharing
    # url = client.read_run(run.id).url
    st.session_state.trace_link = url


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

    if "run_id" not in st.session_state:
        st.session_state.run_id = None

    if "trace_link" not in st.session_state:
        st.session_state.trace_link = None


def render_sidebar(qa: QnA):
    with st.sidebar:

        filtered_llm_options = (key for key, value in cfg.llm_options.items(
        ) if not value.get("disabled", False))

        model = st.selectbox(
            "Choose a LLM",
            filtered_llm_options,
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


def render_chat(qa: QnA, client: Client, run_collector: RunCollectorCallbackHandler):
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        content: str = message.get("content", "").replace("$", "\$")
        role = message["role"]
        if role == 'user':
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(content)

    # Accept user input
    if prompt := st.chat_input("Ask questions"):
        if len(prompt) > c.MAX_CHAR_LIMIT:
            st.warning(
                f"⚠️ Your input is too long! Please limit your input to {c.MAX_CHAR_LIMIT} characters.")
            prompt = None  # Reset the prompt so it doesn't get processed further
            return

        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(prompt.replace("$", "\$"))

        _reset_feedback()

        # Display assistant response in chat message container
        with st.spinner("Loading..."):
            with st.chat_message("assistant", avatar="✨"):

                qa.model = cfg.llm_options[st.session_state.model].get(
                    "llm")

                conversation = st.session_state.selected_conversation or ObjectId()

                try:
                    response = qa.ask_question(
                        query=prompt,
                        config=RunnableConfig(
                            callbacks=[run_collector],
                            tags=["mortgage-broker-chat"],
                            configurable={"session_id": conversation}
                        ),
                        stream=True
                    )

                    content = st.write_stream(
                        ai_response_wrapper(response)
                    )

                    # content = st.markdown(response["answer"])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": content}
                    )

                    _get_trace_link(client, run_collector)

                    # rerun to rerender sidebar with new conversations as selected_conversation
                    if conversation not in st.session_state.conversations:
                        st.session_state.conversations.append(conversation)
                        st.session_state.selected_conversation = conversation
                        st.rerun()

                except httpx.ConnectError:
                    logger.error(e)
                    llm_option_label = (
                        cfg.llm_options[st.session_state.model]
                        .get("label")
                    )

                    st.warning(
                        f"Check your `{llm_option_label}` connection"
                    )
                except BadRequestError as e:
                    logger.error(e)
                    print("e.body", e.body)
                    print("e.body['message']", e.body['message'])
                    # print(e.body['innererror']['content_filter_result'])
                    if e.body.get('code', '') == 'string_above_max_length':
                        st.error(
                            'It seems the question is too complex for me to process. '
                            'Please try splitting it into multiple simpler questions.'
                        )
                    else:
                        st.error(
                            "400 Bad Request: The request could not be processed due to invalid input. "
                            "Please check the format and content of your request and try again."
                        )
                except Exception as e:
                    print('Exception', e)
                    logger.error(e)
                    st.error(
                        "Something went wrong, please try again. "
                        "If the problem persists, please contact the administrator."
                    )


def render_feedback(client: Client):
    has_chat_messages = len(st.session_state.get("messages", [])) > 0

    if not has_chat_messages:
        return

    # st.write(st.session_state.run_id)
    # st.write(st.session_state.trace_link)

    if st.session_state.get("run_id"):
        # feedback_option = (
        #     "faces" if st.toggle(label="`Thumbs` ⇄ `Faces`",
        #                          value=False) else "thumbs"
        # )
        feedback_option = "faces"

        # TODO: streamlit_feedback only return if key is different from previous key else return None
        # make a difference key each submit
        # TODO: fix duplicate render after submit
        feedback = streamlit_feedback(
            feedback_type=feedback_option,
            optional_text_label="[Optional] Please provide an explanation",
            key=f"feedback_{st.session_state.run_id}",
        )

        score_mappings = {
            "thumbs": {"👍": 1, "👎": 0},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings[feedback_option]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                # Formulate feedback type string incorporating the feedback option and score value
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                # Record the feedback with the formulated feedback type string and optional comment
                feedback_record = client.create_feedback(
                    st.session_state.run_id,
                    feedback_type_str,  # Updated feedback type
                    score=score,
                    comment=feedback.get("text"),
                    source_info={
                        "name": "streamlit"
                    },
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }

            else:
                st.warning("Invalid feedback score.")


def main():
    st.set_page_config(page_title="Mortgage Broker Assistant")
    st.title("Mortgage Broker Assistant")

    # Set LangSmith environment variables
    set_up_langsmith_env()

    default_model = cfg.llm_options['azure-openai'].get('llm')

    rag = RAG(model=default_model, rerank=cfg.rerank)

    csv_retriever = CSVRetriever(
        llm=default_model,
        # directory_path=c.AZURE_STORAGE_CONTAINER,
        directory_path='./data/preprocessed/csv/',
        # connection_string=c.AZURE_STORAGE_CONNECTION_STRING
    )

    qa = QnA(
        model=default_model,
        retriever=rag.retriever,
        data_retriever=csv_retriever
    )

    client = Client()
    run_collector = RunCollectorCallbackHandler()

    init_session_state(qa)
    render_sidebar(qa)
    render_chat(qa, client, run_collector)
    render_feedback(client)


if __name__ == "__main__":
    main()
