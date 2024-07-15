from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain

class SimpleIndexToDocstoreId:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

def load_math_notes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def setup_qa_chain(llm, math_notes):
    splitter = CharacterTextSplitter()
    texts = splitter.split_text(math_notes)
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    embedded_texts = embeddings.embed_documents(texts)
    index = faiss.IndexFlatL2(embedded_texts.shape[1])
    index.add(embedded_texts)
    
    documents = [Document(page_content=text) for text in texts]
    docstore = InMemoryDocstore(documents)
    index_to_docstore_id = SimpleIndexToDocstoreId([str(i) for i in range(len(texts))])
    
    vectorstore = FAISS(embedding_function=embeddings.embed_documents, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    retriever = vectorstore.as_retriever()
    return create_retrieval_chain(llm, retriever)

def setup_calculator_tools(calculator):
    from langchain.tools import Tool
    tools = [
        Tool(name="calculator", func=calculator.evaluate_expression, description="Evaluate mathematical expressions"),
        Tool(name="integrator", func=calculator.integrate_expression, description="Integrate mathematical expressions"),
        Tool(name="differentiator", func=calculator.differentiate_expression, description="Differentiate mathematical expressions"),
        Tool(name="solver", func=calculator.solve_equation, description="Solve mathematical equations"),
        Tool(name="matrix_operations", func=calculator.matrix_operations, description="Perform matrix operations")
    ]
    return tools