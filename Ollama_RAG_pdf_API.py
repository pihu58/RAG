import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from Openweather_API import get_weather

API_content_weather = get_weather()
#print("Weather API Content:", API_content_weather)
    
CHROMA_PATH = "chroma_db"
DOC_PATH = "/home/amantya/tech_work/RAG/Climatology-IMTC.pdf"

# --- Clear ChromaDB ---
# if os.path.exists(CHROMA_PATH):
#     print(f"Removing existing ChromaDB at {CHROMA_PATH}")
#     shutil.rmtree(CHROMA_PATH)
#     print("ChromaDB removed successfully.")

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

def retrieve_unique_chunks(question, retriever, target_k=7, initial_k=15, max_attempts=5):
    attempt = 0
    current_k = initial_k
    unique_chunks = []
    seen = set()

    while len(unique_chunks) < target_k and attempt < max_attempts:
        results = retriever.invoke(question)
        for chunk in results:
            content = chunk.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_chunks.append(chunk)
                if len(unique_chunks) >= target_k:
                    break
        current_k += 5  # Increment retrieval size
        retriever.search_kwargs["k"] = current_k
        attempt += 1

    return unique_chunks[:target_k]


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
chunks = splitter.split_documents(pages)

model_name="all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print(f"Initialized HuggingFace embeddings with model: {model_name}")

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

llm_model_name="qwen3:0.6b"
context_window=8912
llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window,
    )

retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 7}
    )



# --- Template with history included ---
template = """Answer the question based ONLY on the following context:
{context}

If asked about the weather, use the following API content to answer the question:

{API_content}

If you are unsure about the answer, say the most relevant information from the context that you can find.
Use the words from the context to answer the question.
if the question matches the exact context, answer from that relevant context.
If you are asked about features, list them in bullet points and also briefly explain each feature.
If the context does not provide enough information to answer the question, say "I don't know".

Question: {question}

"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

# --- MODIFIED CHAINS TO INCLUDE MEMORY ---

# Chain to get and print the context (doesn't need history, just the question)
get_context_chain = (
    RunnablePassthrough.assign(
        context=lambda x: retrieve_unique_chunks(x["question"], retriever, target_k=7)
    )
    | RunnableLambda(lambda x: x["context"])
)

# Initialize the RAG chain with history
rag_chain = (
    RunnablePassthrough.assign(
       context=lambda x: retrieve_unique_chunks(x["question"], retriever, target_k=7),
        API_content = RunnableLambda(lambda x: API_content_weather)
    )
    | prompt
    | llm
    | StrOutputParser()
)

flag = True
while flag == True:
    question = input("Enter the question: ")
    if question.lower() == "q":
        print("Exiting the program.")
        flag = False
        break

    print("\n--- Retrieved Context Documents ---")
    retrieved_docs = get_context_chain.invoke({"question": question})
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print("--------------------")
    print("--- End Retrieved Context ---\n")

    response = rag_chain.invoke({"question": question})
    print("\nResponse:")
    print(response)

    # After getting the response, save the current interaction to memory
    memory.save_context({"input": question}, {"output": response})