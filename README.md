# RAG PDF Assistant com LangChain

> Um sistema de Perguntas e Respostas (QA) local e privado baseado em documentos PDF, utilizando a arquitetura RAG (Retrieval-Augmented Generation).

Este projeto demonstra a implementação de uma pipeline de IA Generativa capaz de ingerir documentos técnicos, vetorizá-los e permitir consultas em linguagem natural com total privacidade, já que todo o processamento ocorre localmente.

## Tecnologias Utilizadas

* **Python 3.10+**
* **LangChain**: Framework para orquestração de LLMs.
* **Ollama (Mistral)**: LLM local para geração de respostas.
* **nomic-embed-text**: Modelo de embeddings local via Ollama.
* **ChromaDB**: Banco de dados vetorial (Vector Store) para busca semântica.
* **PyPDF**: Carregamento e parsing de arquivos PDF.
* **Streamlit**: Interface web para interação com o chat.
* **Docker**: Containerização para garantir que o projeto rode em qualquer ambiente.

## Arquitetura do Projeto

O sistema segue o fluxo padrão de RAG:

1.  **Ingestão**: Carregamento do PDF via `PyPDFLoader`.
2.  **Chunking**: Divisão do texto usando `RecursiveCharacterTextSplitter` para manter o contexto semântico entre quebras.
3.  **Embedding Local**: Conversão de texto em vetores usando `OllamaEmbeddings`.
4.  **Recuperação**: Busca por similaridade no ChromaDB.
5.  **Geração Privada**:O contexto é injetado no prompt do Mistral para gerar a resposta fundamentada.

## Como executar

1. Ter o Ollama instalado e rodando.
2. Baixar os modelos necessários:
   ollama pull mistral
   ollama pull nomic-embed-text
   
* **Opção 1**: Via Docker
  docker build -t rag .
  docker run -p 8501:8501 rag
* **Opção 2**: Localmente
  1. Crie um ambiente virtual: python -m venv venv
  2. Ative o ambiente: .\venv\Scripts\activate (Windows) ou source venv/bin/activate (Linux/Mac)
  3. Instale as dependências: pip install -r requirements.txt
  4. Execute o app: streamlit run app_web.py
  
  

## Estrutura de Arquivos

```bash
 app_web.py        # Interface Web (Streamlit)
 main.py           # Interface de terminal (CLI)
 rag_module.py     # Lógica central da pipeline RAG
 Dockerfile        # Configuração para deploy em container
 requirements.txt  # Dependências enxutas do projeto
 .gitignore        # Proteção para não subir banco de dados ou arquivos temporários
