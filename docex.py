# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())
# openai_api_key = os.environ["OPENAI_API_KEY"]
#
# from langchain_openai import ChatOpenAI
#
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
#
# # from langchain_community.document_loaders import PyMuPDFLoader
# # text = PyMuPDFLoader("C:/Users/hp/OneDrive/Desktop/MLA-C01/Amazon Kinesis Data Streams Troubleshooting Guide.pdf").load()
#
# from langchain.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredFileLoader
# # For DOCX files, you might use a dedicated loader if available:
# from langchain.document_loaders import UnstructuredWordDocumentLoader
#
# def load_document(file_path):
#     ext = os.path.splitext(file_path)[1].lower()
#     if ext == ".pdf":
#         loader = PyMuPDFLoader(file_path)
#     elif ext == ".txt":
#         loader = TextLoader(file_path)
#     elif ext in [".docx", ".doc"]:
#         loader = UnstructuredWordDocumentLoader(file_path)
#     else:
#         # Fallback to a generic loader (make sure to install its dependencies)
#         loader = UnstructuredFileLoader(file_path)
#     return loader.load()
#
#
# files = []
#
# text = []
# for file_path in files:
#     try:
#         docs = load_document(file_path)
#         text.extend(docs)
#     except Exception as e:
#         print(f"Failed to load {file_path}: {e}")
#
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# vector_db = FAISS.from_documents(text, OpenAIEmbeddings())
# retriever = vector_db.as_retriever()
#
# from langchain_core.prompts import ChatPromptTemplate
#
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}")
#     ]
# )
#
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
#
# qa_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, qa_chain)
# # output = rag_chain.invoke({"input": "What are the keywords of this article?"})
# # print(output["answer"])
#
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# contextualized_prompt = (
#     "Given a chat history and the latest user question and context "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )
# contextualized_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualized_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ]
# )
#
# from langchain.chains import create_history_aware_retriever
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt_template)
# history_qa_chain = create_stuff_documents_chain(llm, qa_prompt)
# history_rag_chain = create_retrieval_chain(aware_retriever, history_qa_chain)
#
# from langchain_core.messages import HumanMessage, AIMessage
# chat_history = []
#
#
# query = str(input("Human : "))
# while True :
#     if query == "exit":
#         break
#     else :
#         response = history_rag_chain.invoke({"input": query, "chat_history": chat_history})
#         print(response["answer"])
#         chat_history.extend(
#             [
#                 HumanMessage(content=query),
#                 AIMessage(content=response["answer"]),
#             ]
#         )

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load environment variables (make sure you have a .env with your OPENAI_API_KEY)
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Initialize the LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Import necessary document loaders
from langchain.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    return loader.load()

def load_uploaded_document(uploaded_file):
    # Save uploaded file temporarily so the loader can work with a file path.
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return load_document(tmp_file_path)

st.title("RAG Chat Interface with Multiple File Uploads")

# File uploader widget
uploaded_files = st.file_uploader("Upload files (PDF, TXT, DOCX, etc.)", type=["pdf", "txt", "doc", "docx"], accept_multiple_files=True)

# Load documents from the uploaded files
all_docs = []
if uploaded_files:
    st.info("Loading documents...")
    for uploaded_file in uploaded_files:
        try:
            docs = load_uploaded_document(uploaded_file)
            all_docs.extend(docs)
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load {uploaded_file.name}: {e}")

if all_docs:
    # Build the vector store
    from langchain.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = FAISS.from_documents(all_docs, embeddings)
    retriever = vector_db.as_retriever()

    # Define the prompt template for QA
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
            ("human", "{input}")
        ]
    )

    # Create the retrieval chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Create a history-aware chat setup
    contextualized_prompt = (
        "Given a chat history and the latest user question and context "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualized_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", contextualized_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    from langchain.chains import create_history_aware_retriever
    aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt_template)
    history_qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    history_rag_chain = create_retrieval_chain(aware_retriever, history_qa_chain)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with Your Documents")



    # Chat input box
    user_input = st.chat_input("Enter your question (type 'exit' to reset):", key="user_input")

    if user_input:
        if user_input.strip().lower() == "exit":
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        else:
            # Invoke the history-aware RAG chain
            response = history_rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
            # Append the conversation to session state
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "ai", "content": response["answer"]})
            # Display chat history in order
            if st.session_state.chat_history:
                st.markdown("### Conversation")
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                    # if isinstance(msg, HumanMessage):
                    #     st.markdown(f"**User:** {msg.content}")
                    # elif isinstance(msg, AIMessage):
                    #     st.markdown(f"**Assistant:** {msg.content}")

            else:
                st.info("Please upload at least one document to start the chat.")


