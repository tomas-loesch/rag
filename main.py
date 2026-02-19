from rag_module import RAGPipeline
import os

def main():
    
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Insira sua OpenAI API Key: ")

    pdf_file = "exemplo.pdf"  
    
    # Verifica se o PDF existe antes de tentar rodar
    if not os.path.exists(pdf_file):
        print(f"ERRO: O arquivo '{pdf_file}' nao foi encontrado na pasta.")
        print("Por favor, adicione um PDF e tente novamente.")
        return

    rag = RAGPipeline(pdf_file)
    
    rag.ingest_data()
    
    print("\n Bot RAG pronto! Digite 'sair' para encerrar.")
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
