import streamlit as st
from app import chain

st.set_page_config(page_title="Mortgage Assistant")

st.title("Mortgage Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# TODO: dynamic LLM and list all previous chat
# Can delete chat
# Can create new chat
with st.sidebar:

    add_radio = st.selectbox(
        "Choose a LLM",
        ("Gemini (1.5-pro)", "Cohere (command-r)", "Groq (llama3-70b-8192)")
    )

    st.button("New chat")

    st.write("Previous chat")
    with st.container(height=640, border=False):
        st.markdown("Item 1")
        st.markdown("Item 2")
        st.markdown("Item 3"*1000)


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
            # TODO: dynamic session id
            response = chain.invoke({"input": prompt},
                                    config={"configurable": {"session_id": "zzz12"}})

            content = response.get('answer') or response.get('result')

            st.markdown(content)

    st.session_state.messages.append(
        {"role": "assistant", "content": content})
