# -*- coding: utf-8 -*-
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

# IMPORTS CORRIGIDOS PARA VERSÕES MAIS NOVAS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega variaveis de ambiente (.env)
load_dotenv()

class RAGPipeline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vector_store = None
        
        # Inicializa o modelo (LLM)
        # temperature=0 para respostas mais diretas e menos criativas
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def ingest_data(self):
        """Carrega o PDF, faz o chunking e indexa no banco vetorial."""
        if not os.path.exists(self.pdf_path):
            # AQUI ESTAVA O ERRO: removi o acento de "nao"
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.pdf_path}")

        print(f"Carregando {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        # Estrategia de Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Tamanho do pedaco
            chunk_overlap=200   # Sobreposicao para manter contexto
        )
        splits = text_splitter.split_documents(docs)

        print("Gerando Embeddings e salvando no ChromaDB...")
        self.vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )
        print("Indexacao concluida!")

    def ask(self, query):
        """Realiza a busca semantica e gera a resposta."""
        if not self.vector_store:
            raise ValueError("Voce precisa executar ingest_data() primeiro.")

        # Cria o recuperador (Retriever)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3} # Retorna os 3 trechos mais relevantes
        )

        # Template do Prompt do Sistema (em ingles ou portugues sem acentos preferencialmente)
        system_prompt = (
            "Voce e um assistente util para tarefas de perguntas e respostas. "
            "Use os seguintes pedacos de contexto recuperado para responder a pergunta. "
            "Se voce nao souber a resposta, diga apenas que nao sabe. "
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Cria a cadeia de processamento (Chain)
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        return response["answer"]