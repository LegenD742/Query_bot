from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/college_policy.pdf")

documents = loader.load()

for doc in documents:
    print("=" * 80)
    print("PAGE:", doc.metadata["page"] + 1)
    print(doc.page_content)

