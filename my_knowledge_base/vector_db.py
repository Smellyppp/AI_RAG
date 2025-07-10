import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def create_vector_db(metadata_path, embedding_model_path, vector_db_path, mode='create'):
    """创建或更新向量数据库"""
    # 加载分块数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 准备文档数据
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)
    
    # 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'}
    )
    
    # 处理数据库操作模式
    if mode == 'append' and os.path.exists(vector_db_path):
        # 追加模式：加载现有数据库并添加新文档
        vector_db = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        vector_db.add_documents(documents)
        print(f"已向现有数据库追加 {len(documents)} 个文档")
    else:
        # 创建模式（新建或覆盖）
        if mode == 'append':
            print("警告：未找到现有数据库，将创建新数据库")
        vector_db = FAISS.from_documents(documents, embeddings)
        print(f"已创建包含 {len(documents)} 个文档的新数据库")
    
    # 保存数据库
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    vector_db.save_local(vector_db_path)
    print(f"向量数据库已保存至: {vector_db_path}")

def search_vector_db(query, vector_db_path, embedding_model_path, k=3):
    """在向量数据库中搜索"""
    # 检查数据库是否存在
    if not os.path.exists(vector_db_path):
        print(f"错误：未找到向量数据库 {vector_db_path}")
        return []
    
    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'}
    )
    
    # 安全加载向量数据库
    vector_db = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 执行搜索
    results = vector_db.similarity_search(query, k=k)
    
    # 整理结果
    search_results = []
    for i, doc in enumerate(results):
        search_results.append({
            "rank": i+1,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": None
        })
    
    return search_results

if __name__ == "__main__":
    # 配置路径
    METADATA_PATH = "./chunk_output/docx/天照/metadata.json"
    EMBEDDING_MODEL_PATH = "../embedding_model/all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "./vector_db/faiss_index"
    
    # 用户选择操作模式
    while True:
        user_input = input("请选择操作模式:\n1. 新建/覆盖数据库\n2. 追加到现有数据库\n输入选项(1/2): ")
        if user_input in ['1', '2']:
            mode = 'create' if user_input == '1' else 'append'
            break
        print("无效输入，请重新输入")
    
    # 创建/更新向量数据库
    create_vector_db(
        metadata_path=METADATA_PATH,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        vector_db_path=VECTOR_DB_PATH,
        mode=mode
    )
    
    # # 示例搜索
    # query = "月读是谁？"
    # results = search_vector_db(query, VECTOR_DB_PATH, EMBEDDING_MODEL_PATH)
    
    # print(f"\n搜索查询: '{query}'")
    # print("="*50)
    # for res in results:
    #     print(f"\n[结果 #{res['rank']}]")
    #     print(f"内容: {res['content'][:500]}...")
    #     print("-"*50)