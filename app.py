import streamlit as st

from src.rag import ask_question


st.set_page_config(
    page_title="College Policy Assistant",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# UI
# -----------------------------

st.title("🎓 College Policy Assistant")

st.write(
    "Ask questions about college policies, "
    "academic calendar, examinations etc."
)


# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------

question = st.chat_input(
    "Ask a question..."
)


if question:

    # Show user question
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching college policies..."):

            answer, documents = ask_question(question)

        st.markdown(answer)


        # -----------------------------
        # Sources
        # -----------------------------

        with st.expander("📚 View sources"):

            for i, doc in enumerate(documents):

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    None
                )

                if page is not None:
                    page = page + 1

                st.markdown(
                    f"**Source {i + 1}: "
                    f"{source} — Page {page}**"
                )

                st.write(
                    doc.page_content
                )


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })