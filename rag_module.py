# -*- coding: utf-8 -*-
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Troca OpenAI por Ollama
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vector_store = None
        
        # Configurando o modelo Local 
        # 'mistral'
        print("?? Inicializando modelo local (Ollama)...")
        self.llm = ChatOllama(model="mistral", temperature=0)

    def ingest_data(self):
        """Carrega o PDF, faz o chunking e indexa no banco vetorial."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.pdf_path}")

        print(f"Carregando {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        print("Gerando Embeddings locais (isso usa CPU)...")
        # Embeddings locais gratuitos
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        print("Indexacao concluida!")

    def ask(self, query):
        if not self.vector_store:
            raise ValueError("Voce precisa executar ingest_data() primeiro.")

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        system_prompt = (
            "Voce e um assistente tecnico. "
            "Use o contexto abaixo para responder. Se nao souber, diga que nao sabe."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print("Pensando...")
        response = rag_chain.invoke({"input": query})
        return response["answer"]