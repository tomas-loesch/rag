# RAG PDF Assistant com LangChain

> Um sistema de Perguntas e Respostas (QA) baseado em documentos PDF, utilizando a arquitetura RAG (Retrieval-Augmented Generation).

Este projeto demonstra a implementação de uma pipeline de IA Generativa capaz de ingerir documentos técnicos, vetorizá-los e permitir consultas em linguagem natural com alta precisão de contexto.

## Tecnologias Utilizadas

* **Python 3.10+**
* **LangChain**: Framework para orquestração de LLMs.
* **OpenAI API (GPT-3.5/4)**: Modelo de geração e embeddings.
* **ChromaDB**: Banco de dados vetorial (Vector Store) para busca semântica.
* **PyPDF**: Carregamento e parsing de arquivos PDF.
* **Ollama**: Modelo LLM local

## Arquitetura do Projeto

O sistema segue o fluxo padrão de RAG:

1.  **Ingestão**: Carregamento do PDF via `PyPDFLoader`.
2.  **Chunking**: Divisão do texto usando `RecursiveCharacterTextSplitter` para manter o contexto semântico entre quebras.
3.  **Embedding**: Conversão dos chunks em vetores numéricos usando `OpenAIEmbeddings`.
4.  **Armazenamento**: Indexação dos vetores no `ChromaDB`.
5.  **Recuperação (Retrieval)**: Busca por similaridade (Cosine Similarity) para encontrar os trechos mais relevantes à pergunta do usuário.
6.  **Geração**: Envio do contexto recuperado + pergunta para o LLM gerar a resposta final.

## Estrutura de Arquivos

```bash
 main.py           # Ponto de entrada da aplicação (CLI)
 rag_module.py     # Lógica encapsulada da pipeline RAG
 requirements.txt  # Dependências do projeto
 .env              # Variáveis de ambiente (API Keys)
 README.md         # Documentação