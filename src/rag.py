from langchain_core.prompts import ChatPromptTemplate

from src.retriever import get_retriever
from src.llm import get_llm


retriever = get_retriever()
llm = get_llm()


prompt = ChatPromptTemplate.from_template(
    """
You are a college policy assistant.

Answer the user's question using ONLY the context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I could not find this information in the college policy documents."
4. Give a concise and direct answer.

Context:
{context}

Question:
{question}

Answer:
"""
)


def ask_question(question):

    # 1. Retrieve relevant chunks
    documents = retriever.invoke(question)

    # 2. Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # 3. Create prompt
    messages = prompt.format_messages(
        context=context,
        question=question
    )

    # 4. Send context + question to Llama
    response = llm.invoke(messages)

    return response.content, documents