
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    print("Direct import SUCCESS")
except ImportError as e:
    print(f"Direct import FAILED: {e}")
