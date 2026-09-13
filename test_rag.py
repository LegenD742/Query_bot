from src.rag import ask_question


question = "When is the winter vacation for the students?"


answer, documents = ask_question(question)


print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:")

for doc in documents:

    print(
        f"- {doc.metadata.get('source')} "
        f"(Page {doc.metadata.get('page', 0) + 1})"
    )