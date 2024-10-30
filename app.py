from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
import os

# Load and process document
file_name = "ChangeYourBrainEveryDay"
file_extension = "pdf"
file_path = "ChatbotDemo/data/"+file_name+"."+file_extension
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_KEY"])

# Connect or create a Chroma instance for persistent embeddings
chroma_db = Chroma(persist_directory="ChatbotDemo/data", collection_name=file_name, embedding_function=embeddings)

retriever = chroma_db.as_retriever()

# Get the collection from the Chroma database
collection = chroma_db.get()

# If the collection is empty, create a new one
if len(collection['ids']) == 0:
    print("Creating new embeddings...")
    chroma_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="ChatbotDemo/data",
        collection_name=file_name
    )

    # Save the Chroma database to disk
    chroma_db.persist()


# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# Set up memory and chain
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.environ["OPENAI_KEY"])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



question = st.text_input("Ask a question about the document")

if question:
    result = rag_chain.invoke({"input": question})
    st.text_area("Answer", value=result["answer"], height=200)
