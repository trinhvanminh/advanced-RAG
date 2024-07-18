import streamlit as st
from app import chain

st.set_page_config(page_title="Mortgage")

st.title("Mortgage")

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

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
            response = chain.invoke({"input": prompt},
                                    config={"configurable": {"session_id": "zzz"}})
            st.markdown(response['answer'])
    st.session_state.messages.append(
        {"role": "assistant", "content": response['answer']})
