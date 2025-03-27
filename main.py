from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import uuid
from datetime import datetime

load_dotenv()
#Initialize Vector Database
pc = Pinecone()

index_name = "semantic-cache"
index = pc.Index(index_name)

#creating index and then commenting once the index is created
# pc.create_index(
#     name=index_name,
#     dimension=384, 
#     metric="cosine", 
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

#Select embedding model

embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Initialize LLM (Groq)
llm = ChatGroq(temperature=0, model="llama3-70b-8192")

# Prompt user for question
question = input("Ask your question: ")

# 2. Embed the Question using BAAI/bge-small-en
question_embedding = embedding_model.encode(question, normalize_embeddings=True)

# 3. Perform Similarity Search in Pinecone
search_result = index.query(vector=question_embedding.tolist(), top_k=1, include_metadata=True)

# 4. If Similar Question Found → Return Cached Answer
threshold = 0.80
if search_result.matches and search_result.matches[0].score >= threshold:
    match = search_result.matches[0]
    print("\n Cache Hit")
    print("Matched Question:", match.metadata['question'])
    print("Similarity Score:", match.score)
    print("Answer:", match.metadata['answer'])
    print("Source: cache")

# 5. If Not Found → Generate Answer using LLM (Groq LLaMA3)
else:
    print("\n Cache Miss")
    # Generate answer via LLM
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer the following question:\n\nQuestion: {question}"
    )
    chain = prompt | llm
    response = chain.invoke({"question": question})
    answer = response.content

    # 6. Store New Q&A Pair in Pinecone (with timestamp & metadata)
    index.upsert([
        (
            str(uuid.uuid4()),
            question_embedding.tolist(),
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "LLM"
            }
        )
    ])

    # 7. Return Final Answer to User (indicating source: LLM or Cache)
    print("Answer:", answer)
    print("Source: LLM")