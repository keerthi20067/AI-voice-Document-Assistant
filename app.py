import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import pipeline
import matplotlib.pyplot as plt
import networkx as nx
# Page settings
st.set_page_config(page_title="AI Document Assistant", layout="wide")

# Title
st.title("AI Document Assistant")
st.write("Upload a PDF and ask questions about it.")

# Sidebar
st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    # Save file
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Document uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector DB
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    hf_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=512
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Main question box
    st.subheader("Ask a Question")

    query = st.text_input("Type your question here")

    if st.button("Get Answer"):

        answer = qa_chain.run(query)

        st.subheader("Answer")
        st.write(answer)

        # Simple concept graph
        keywords = answer.split()[:5]

        G = nx.Graph()

        for i in range(len(keywords)-1):
            G.add_edge(keywords[i], keywords[i+1])

        fig = plt.figure(figsize=(5,5))
        nx.draw(G, with_labels=True, node_color="lightblue", node_size=2000)

        st.subheader("Concept Map")
        st.pyplot(fig)

else:
    st.info("Upload a PDF from the sidebar to start.")