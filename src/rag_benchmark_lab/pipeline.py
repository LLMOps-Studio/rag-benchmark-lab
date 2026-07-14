from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llmops_common.client.ollama_client import OllamaClient
from llmops_common.vectorstore.chroma_client import ChromaClient

class RAGPipeline:
    """
    Configurable RAG Pipeline to test different chunking strategies locally.
    Uses shared llmops-common SDK components.
    """
    def __init__(self, collection_name: str = "rag_benchmark"):
        self.chroma = ChromaClient()
        self.ollama = OllamaClient()
        self.collection_name = collection_name
        
        # Ensure a clean slate for each independent pipeline configuration run
        try:
            self.chroma.client.delete_collection(name=collection_name)
        except Exception:
            pass
        self.collection = self.chroma.get_or_create_collection(collection_name)

    def ingest_documents(self, raw_text: str, chunk_size: int, chunk_overlap: int):
        """Splits text based on parameters and loads them into ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        # Add documents using the common vector store client
        self.chroma.add_documents(
            collection_name=self.collection_name,
            documents=chunks
        )
        return len(chunks)

    def answer_query(self, query: str, model_name: str, n_results: int = 2) -> Dict[str, Any]:
        """Retrieves context from ChromaDB and synthesizes an answer via Ollama."""
        # 1. Retrieve relevant chunks
        retrieval_results = self.chroma.query_documents(
            collection_name=self.collection_name,
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_chunks = retrieval_results.get("documents", [[]])[0]
        context = " ".join(retrieved_chunks)

        # 2. Construct the prompt
        system_prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer the query strictly using the context:"
        
        # 3. Generate response from local LLM
        response = self.ollama.generate(prompt=system_prompt, model=model_name)
        
        return {
            "query": query,
            "context": context,
            "answer": response
        }