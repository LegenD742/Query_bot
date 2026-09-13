from src.retriever import get_retriever


retriever = get_retriever()


question = "When is the winter vacation?"


documents = retriever.invoke(question)


for i, doc in enumerate(documents):

    print("\n" + "=" * 80)

    print(f"RESULT {i + 1}")

    print("=" * 80)

    print("SOURCE:", doc.metadata.get("source"))

    print("PAGE:", doc.metadata.get("page", 0) + 1)

    print("\nCONTENT:")

    print(doc.page_content)