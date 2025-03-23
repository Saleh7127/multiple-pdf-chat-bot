import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage, AIMessage


from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template


def get_conversation_chain(vectorstore):

    llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
    
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("context"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    conversation_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return conversation_chain


def get_vector_store(text_chunks):
    # Initialize the HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    # Precompute embeddings for the text chunks
    text_embeddings = embeddings.embed_documents(text_chunks)

    # Pair each text with its corresponding embedding
    text_embedding_pairs = zip(text_chunks, text_embeddings)

    # Create a FAISS vector store from the text-embedding pairs
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs, embedding=embeddings
    )

    return vector_store


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_extracted_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def handle_user_input(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    response = st.session_state.conversation.invoke({
        "input": user_question,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.extend([AIMessage(content=msg) for msg in response.get("chat_history", [])])

    # Display the conversation
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF chatbot", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_qn = st.text_input("Ask a question about your documents:")

    if user_qn:
        handle_user_input(user_qn)

    # st.write(user_template.replace("{{MSG}}", "Hello Helper"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello, How can I assist"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the pdf text
                raw_text = get_extracted_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()