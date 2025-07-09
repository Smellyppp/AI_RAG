import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorizationSystem:
    def __init__(self, 
                 embedding_model_path=r'C:\Users\G1581\Desktop\GitHub\local_rag\modelscope\nlp_gte_sentence-embedding_chinese-large',
                 vector_db_path='./vector_db',
                 embedding_cache_path='./embeddings'):
        """
        初始化向量化系统
        :param embedding_model_path: 嵌入模型路径
        :param vector_db_path: 向量数据库存储路径
        :param embedding_cache_path: 向量缓存路径（可选）
        """
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # 设置路径
        self.vector_db_path = vector_db_path
        self.embedding_cache_path = embedding_cache_path
        
        # 确保目录存在
        os.makedirs(vector_db_path, exist_ok=True)
        if embedding_cache_path:
            os.makedirs(embedding_cache_path, exist_ok=True)
        
        # 初始化向量数据库
        self.vector_db = self._init_vector_db()
    
    def _init_vector_db(self):
        """初始化或加载向量数据库"""
        index_file = os.path.join(self.vector_db_path, 'faiss.index')
        metadata_file = os.path.join(self.vector_db_path, 'metadata.json')
        
        # 创建新的向量数据库
        index = faiss.IndexFlatL2(self.dimension)
        metadata = []
        
        # 如果存在保存的数据库，则加载
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                index = faiss.read_index(index_file)
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"已加载现有向量数据库: {len(metadata)} 条记录")
            except Exception as e:
                print(f"加载向量数据库失败: {e}. 创建新数据库")
        
        return {'index': index, 'metadata': metadata}
    
    def _get_embedding_cache_path(self, chunk_path):
        """获取向量缓存路径"""
        if not self.embedding_cache_path:
            return None
        
        # 保留原始目录结构
        relative_path = os.path.relpath(chunk_path, start='chunk_output')
        cache_path = os.path.join(self.embedding_cache_path, relative_path)
        cache_path = os.path.splitext(cache_path)[0] + '.npy'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        return cache_path
    
    def embed_text(self, text):
        """将单个文本转换为向量"""
        return self.embedding_model.encode([text])[0]
    
    def embed_batch(self, texts):
        """批量转换文本为向量"""
        return self.embedding_model.encode(texts)
    
    def process_chunks(self, chunk_dir='./chunk_output'):
        """
        处理分块文本目录，生成向量并添加到数据库
        :param chunk_dir: 分块文本的根目录
        """
        total_chunks = 0
        processed_files = 0
        
        # 遍历所有文件类型目录
        for file_type in os.listdir(chunk_dir):
            type_dir = os.path.join(chunk_dir, file_type)
            if not os.path.isdir(type_dir):
                continue
            
            print(f"\n处理 {file_type.upper()} 文件:")
            
            # 遍历每个文件的分块目录
            for file_name in os.listdir(type_dir):
                file_dir = os.path.join(type_dir, file_name)
                if not os.path.isdir(file_dir):
                    continue
                
                # 读取元数据文件
                metadata_path = os.path.join(file_dir, 'metadata.json')
                if not os.path.exists(metadata_path):
                    print(f"跳过缺少元数据的目录: {file_dir}")
                    continue
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    try:
                        chunks_metadata = json.load(f)
                    except json.JSONDecodeError:
                        print(f"元数据文件格式错误: {metadata_path}")
                        continue
                
                # 准备批量处理
                texts = []
                metadata_list = []
                
                for chunk_meta in chunks_metadata:
                    # 检查是否已处理过
                    if any(m['source_path'] == chunk_meta['metadata']['source'] 
                           and m['chunk_id'] == chunk_meta['chunk_id']
                           for m in self.vector_db['metadata']):
                        continue
                    
                    # 添加文本和元数据
                    texts.append(chunk_meta['text'])
                    metadata_list.append({
                        'text': chunk_meta['text'],
                        'chunk_id': chunk_meta['chunk_id'],
                        'source_path': chunk_meta['metadata']['source'],
                        'file_type': chunk_meta['metadata']['file_type'],
                        'original_name': chunk_meta['metadata']['original_name'],
                        'start_index': chunk_meta['metadata']['start_index'],
                        'end_index': chunk_meta['metadata']['end_index']
                    })
                
                # 批量处理文本
                if texts:
                    vectors = self.embed_batch(texts)
                    self._add_to_vector_db(vectors, metadata_list)
                    total_chunks += len(texts)
                    print(f"已处理 {file_name}: 添加 {len(texts)} 个分块")
                    processed_files += 1
        
        # 保存数据库
        self.save_vector_db()
        
        print(f"\n处理完成! 共处理 {processed_files} 个文件，添加 {total_chunks} 个新分块")
        return total_chunks
    
    def _add_to_vector_db(self, vectors, metadata_list):
        """添加向量到数据库"""
        # 转换为numpy数组
        vectors = np.array(vectors).astype('float32')
        
        # 添加到索引
        self.vector_db['index'].add(vectors)
        
        # 添加元数据
        self.vector_db['metadata'].extend(metadata_list)
    
    def save_vector_db(self):
        """保存向量数据库"""
        # 保存FAISS索引
        faiss.write_index(self.vector_db['index'], 
                         os.path.join(self.vector_db_path, 'faiss.index'))
        
        # 保存元数据
        with open(os.path.join(self.vector_db_path, 'metadata.json'), 
                 'w', encoding='utf-8') as f:
            json.dump(self.vector_db['metadata'], f, ensure_ascii=False, indent=2)
        
        print(f"向量数据库已保存至: {self.vector_db_path}")
    
    def search(self, query, k=5):
        """
        语义搜索
        :param query: 查询文本
        :param k: 返回结果数量
        :return: 搜索结果列表
        """
        # 将查询文本转换为向量
        query_vector = self.embed_text(query)
        query_vector = np.array([query_vector]).astype('float32')
        
        # 执行搜索
        distances, indices = self.vector_db['index'].search(query_vector, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # 有效索引
                metadata = self.vector_db['metadata'][idx]
                results.append({
                    'score': float(distances[0][i]),
                    'text': metadata['text'],
                    'source': metadata['source_path'],
                    'file_type': metadata['file_type'],
                    'original_name': metadata['original_name'],
                    'position': f"{metadata['start_index']}-{metadata['end_index']}"
                })
        
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化系统
    vector_system = VectorizationSystem()
    
    # 处理分块文本并构建向量数据库
    vector_system.process_chunks('./chunk_output')
    
    # # 执行搜索
    # query = "郭丽珍"
    # results = vector_system.search(query, k=3)
    
    # print(f"\n搜索查询: '{query}'")
    # for i, result in enumerate(results):
    #     print(f"\n结果 #{i+1} (相似度: {result['score']:.4f}):")
    #     print(f"来源: {result['source']}")
    #     print(f"内容: {result['text'][:200]}...")