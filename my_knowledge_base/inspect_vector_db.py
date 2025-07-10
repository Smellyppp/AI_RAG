import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def inspect_vector_db(vector_db_path, embedding_model_path, max_docs=5, show_content=True):
    """
    查看向量数据库中的存储内容
    
    参数:
        vector_db_path: 向量数据库的存储路径
        embedding_model_path: 嵌入模型的路径
        max_docs: 最大显示文档数量 (默认5)
        show_content: 是否显示文档内容 (默认True)
    """
    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'}
    )
    
    # 安全加载向量数据库
    try:
        vector_db = FAISS.load_local(
            folder_path=vector_db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ 加载向量数据库失败: {str(e)}")
        return []
    
    # 获取所有存储的文档
    all_docs = vector_db.docstore._dict
    
    print(f"\n🔍 向量数据库位置: {vector_db_path}")
    print(f"📊 总存储文档数: {len(all_docs)}")
    print("=" * 60)
    
    # 打印文档信息
    print(f"📄 文档内容示例 (显示前 {max_docs} 条):")
    docs_list = []
    for i, (doc_id, doc) in enumerate(list(all_docs.items())):
        if i >= max_docs:
            break
            
        doc_info = {
            "id": doc_id,
            "metadata": doc.metadata,
            "content": doc.page_content
        }
        docs_list.append(doc_info)
        
        print(f"\n[文档 {i+1}] | ID: {doc_id}")
        print(f"📌 元数据: {json.dumps(doc.metadata, indent=2, ensure_ascii=False)}")
        
        if show_content:
            # 显示前200个字符
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"📝 内容: {content_preview}")
        
        print("-" * 60)
    
    print(f"\n✅ 成功加载 {len(docs_list)}/{len(all_docs)} 个文档")
    return docs_list

if __name__ == "__main__":
    # 配置路径 - 根据你的实际路径修改
    EMBEDDING_MODEL_PATH = "../embedding_model/all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "./vector_db/faiss_index"
    
    # 查看向量数据库内容
    inspect_vector_db(
        vector_db_path=VECTOR_DB_PATH,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        max_docs=3,  # 查看前3个文档
        show_content=True
    )