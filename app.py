import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuración de la página
st.set_page_config(page_title="Soporte Bancario RAG", layout="wide")

def process_documents(uploaded_file):
    """
    Procesa el archivo subido (PDF o CSV), genera chunks y crea la base vectorial.
    """
    try:
        # Guardar archivo temporalmente para lectura
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Selección de loader según extensión
        if uploaded_file.name.endswith('.pdf'):
            loader = PyMuPDFLoader(tmp_path)
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(tmp_path)
        else:
            return None
        
        docs = loader.load()
        
        # Eliminar archivo temporal
        os.remove(tmp_path)

        # Splitting (Configuración basada en el notebook: 500 chunk, 50 overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)

        # Embeddings (Modelo BAAI/bge-large-en-v1.5 usado en el notebook)
        # Nota: Se usa device='cpu' para compatibilidad general
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )

        # Creación de Vector Store (FAISS)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    except Exception as e:
        st.error(f"Error al procesar el documento: {e}")
        return None

def get_rag_chain(retriever, api_key):
    """
    Configura la cadena RAG con el modelo Llama 3 via Groq.
    """
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

    template = """You are a helpful virtual assistant for a bank. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, strictly say you don't know. 
    Keep your answer concise and professional.

    Context: {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    st.title("🏦 Asistente Bancario AI (RAG)")

    # --- Sidebar: Configuración ---
    with st.sidebar:
        st.header("Configuración")
        api_key = st.text_input("Groq API Key", type="password")
        
        st.subheader("Base de Conocimiento")
        uploaded_file = st.file_uploader("Cargar manual o datos (PDF/CSV)", type=["pdf", "csv"])
        
        if st.button("Procesar Documentos") and uploaded_file and api_key:
            with st.spinner("Indexando documentos..."):
                retriever = process_documents(uploaded_file)
                if retriever:
                    st.session_state["retriever"] = retriever
                    st.success("Documentos indexados correctamente.")
                
    # --- Verificación de estado ---
    if not api_key:
        st.info("Por favor, introduce tu API Key de Groq para continuar.")
        return

    if "retriever" not in st.session_state:
        st.info("Por favor, carga y procesa un documento para iniciar el chat.")
        return

    # --- Interfaz de Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input de usuario
    if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
        # Guardar y mostrar mensaje usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            try:
                chain = get_rag_chain(st.session_state["retriever"], api_key)
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error al generar respuesta: {e}")

if __name__ == "__main__":
    main()