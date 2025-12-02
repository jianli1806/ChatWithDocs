import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self):
        print("---------- 🚀 引擎初始化开始 ----------")
        
        # 1. Initialize LLM (Groq)
        print("1️⃣ 连接 Groq API...", end=" ", flush=True)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.1
        )
        print("✅ 完成")
        
        # 2. Initialize Embeddings
        print("2️⃣ 加载 Embedding 模型 (首次运行会下载 ~100MB)...")
        start_time = time.time()
        
        # 这里是可能卡住的地方
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        duration = time.time() - start_time
        print(f"✅ Embedding 模型加载完成! 耗时: {duration:.2f} 秒")
        
        # 3. Vector Store Path
        self.persist_directory = "./chroma_db"
        print("---------- 🏁 引擎初始化结束 ----------")
        
    def process_pdf(self, file_path):
        """Load PDF -> Chunk text -> Store in Vector DB"""
        print(f"📄 正在处理文件: {file_path}")
        
        # A. Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"   - 读取到 {len(docs)} 页")
        
        # B. Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        print(f"   - 切分为 {len(splits)} 个文本块")
        
        # C. Store in ChromaDB
        print("   - 正在写入向量数据库 (ChromaDB)...", end=" ", flush=True)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✅ 写入完成")
        
        return vectorstore

    def chat(self, query, vectorstore):
        """Retrieve context -> Generate Answer"""
        
        # 1. Define Retriever
        # Search for top 3 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. Define Prompt
        # Standard RAG prompt structure
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If the answer is not in the context, just say that you don't know. 
        Don't try to make up an answer.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """)
        
        # 3. Build Chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # 4. Execute
        response = retrieval_chain.invoke({"input": query})
        
        return response