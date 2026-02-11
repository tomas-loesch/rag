# -*- coding: utf-8 -*-
from rag_module import RAGPipeline
import os

def main():
    # Defina sua chave da API aqui ou idealmente no arquivo .env
    # Se nao encontrar a chave, pede para o usuario digitar
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Insira sua OpenAI API Key: ")

    pdf_file = "exemplo.pdf"  # Certifique-se que o arquivo existe
    
    # Verifica se o PDF existe antes de tentar rodar
    if not os.path.exists(pdf_file):
        print(f"ERRO: O arquivo '{pdf_file}' nao foi encontrado na pasta.")
        print("Por favor, adicione um PDF e tente novamente.")
        return

    # Instancia a pipeline
    rag = RAGPipeline(pdf_file)
    
    # Ingestao (Indexacao) - Acentos removidos para evitar erro
    rag.ingest_data()
    
    # Loop de interacao
    print("\n?? Bot RAG pronto! Digite 'sair' para encerrar.")
    while True:
        try:
            user_input = input("\nPergunta: ")
            if user_input.lower() in ["sair", "exit"]:
                break
            
            resposta = rag.ask(user_input)
            print(f"Resposta: {resposta}")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()