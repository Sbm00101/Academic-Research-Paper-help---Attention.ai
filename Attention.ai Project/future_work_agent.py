from transformers import pipeline
from neodummy import get_neo4j_connection

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
driver = get_neo4j_connection()

def generate_future_work(query):
    with driver.session() as session:
        retrieved_docs = session.run(
            """
            MATCH (d:Document)
            WHERE d.summary CONTAINS $query OR d.title CONTAINS $query
            RETURN d.title AS title, d.summary AS summary
            LIMIT 5
            """,
            query=query
        )

        docs = retrieved_docs.data()
        if not docs:
            return "No relevant documents found."

        context = " ".join([doc['summary'] for doc in docs])
        response = summarizer(context, max_length=150, min_length=50, do_sample=False)

        return response[0]['summary_text']
