from src.retriever import get_retriever


# --------------------------------------------------
# Evaluation dataset
# --------------------------------------------------

EVALUATION_DATA = [
    {
        "question": "When is the winter vacation for students?",
        "keywords": [
            "15 Dec 2026",
            "03 Jan 2027"
        ]
    },
    {
        "question": "When is the Test-2 examination?",
        "keywords": [
            "12-19 Oct 2026"
        ]
    },
    {
        "question": "When do classes end in the odd semester?",
        "keywords": [
            "28 Nov 2026"
        ]
    },
    {
        "question": "When are the end semester examinations?",
        "keywords": [
            "01 -14 Dec 2026"
        ]
    },
    {
        "question": "When does the next semester commence?",
        "keywords": [
            "04 Jan 2027"
        ]
    }
]



def is_relevant(document, keywords):

    text = document.page_content.lower()

    # We consider a chunk relevant if
    # at least one important keyword appears.
    for keyword in keywords:

        if keyword.lower() in text:
            return True

    return False


# --------------------------------------------------
# Recall@K
# --------------------------------------------------

def calculate_recall(retriever, k=4):

    successful_queries = 0

    for item in EVALUATION_DATA:

        documents = retriever.invoke(
            item["question"]
        )

        # Only consider top K
        documents = documents[:k]

        found = any(
            is_relevant(
                doc,
                item["keywords"]
            )
            for doc in documents
        )

        if found:
            successful_queries += 1

    recall = (
        successful_queries
        / len(EVALUATION_DATA)
    )

    return recall


# --------------------------------------------------
# MRR
# --------------------------------------------------

def calculate_mrr(retriever, k=4):

    reciprocal_ranks = []

    for item in EVALUATION_DATA:

        documents = retriever.invoke(
            item["question"]
        )

        documents = documents[:k]

        rank = None

        for i, document in enumerate(
            documents,
            start=1
        ):

            if is_relevant(
                document,
                item["keywords"]
            ):

                rank = i
                break

        if rank is not None:

            reciprocal_ranks.append(
                1 / rank
            )

        else:

            reciprocal_ranks.append(0)

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    return mrr


# --------------------------------------------------
# Detailed evaluation
# --------------------------------------------------

def evaluate():

    retriever = get_retriever()

    k = 4

    print("\n")
    print("=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for item in EVALUATION_DATA:

        question = item["question"]

        documents = retriever.invoke(question)

        documents = documents[:k]

        print("\n" + "-" * 60)
        print("QUESTION:")
        print(question)

        print("\nRETRIEVED:")

        for i, doc in enumerate(
            documents,
            start=1
        ):

            relevant = is_relevant(
                doc,
                item["keywords"]
            )

            print(
                f"\nRank {i}"
                f" | Relevant: {relevant}"
            )

            print(
                f"Source: "
                f"{doc.metadata.get('source')}"
            )

            print(
                f"Page: "
                f"{doc.metadata.get('page', 0) + 1}"
            )

            print(
                doc.page_content[:300]
            )

    # Calculate metrics
    recall = calculate_recall(
        retriever,
        k
    )

    mrr = calculate_mrr(
        retriever,
        k
    )

    print("\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"Recall@{k}: {recall:.2%}"
    )

    print(
        f"MRR@{k}:    {mrr:.3f}"
    )



if __name__ == "__main__":

    evaluate()