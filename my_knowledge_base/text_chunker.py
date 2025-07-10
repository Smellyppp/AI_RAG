import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader

class StructuredTextLoader(BaseLoader):
    """自定义结构化文本加载器，保留标题信息"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self):
        """加载文档并保留结构信息"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        documents = []
        current_section1 = ""
        current_section2 = ""
        current_content = []
        
        # 正则表达式匹配标题标签
        section1_pattern = re.compile(r'^\[SECTION_1\](.+?)\[/SECTION_1\]$')
        section2_pattern = re.compile(r'^\[SECTION_2\](.+?)\[/SECTION_2\]$')
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # 检查一级标题
            section1_match = section1_pattern.match(line)
            if section1_match:
                # 保存当前块（如果有内容）
                if current_content:
                    metadata = {
                        'section1': current_section1,
                        'section2': current_section2,
                        'start_line': i - len(current_content),
                        'end_line': i - 1
                    }
                    documents.append(Document(
                        page_content='\n'.join(current_content),
                        metadata=metadata
                    ))
                    current_content = []
                
                current_section1 = section1_match.group(1).strip()
                current_section2 = ""  # 重置二级标题
                continue
            
            # 检查二级标题
            section2_match = section2_pattern.match(line)
            if section2_match:
                # 保存当前块（如果有内容）
                if current_content:
                    metadata = {
                        'section1': current_section1,
                        'section2': current_section2,
                        'start_line': i - len(current_content),
                        'end_line': i - 1
                    }
                    documents.append(Document(
                        page_content='\n'.join(current_content),
                        metadata=metadata
                    ))
                    current_content = []
                
                current_section2 = section2_match.group(1).strip()
                continue
            
            # 普通内容行
            if line.strip():  # 跳过空行
                current_content.append(line)
        
        # 处理最后一个块
        if current_content:
            metadata = {
                'section1': current_section1,
                'section2': current_section2,
                'start_line': len(lines) - len(current_content),
                'end_line': len(lines) - 1
            }
            documents.append(Document(
                page_content='\n'.join(current_content),
                metadata=metadata
            ))
        
        return documents

def chunk_and_save_parsed_files(parsed_dir="parsed_document", output_dir="chunk_output", 
                               chunk_size=500, chunk_overlap=100):
    """使用LangChain处理已解析的文档文件，进行结构感知分块并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建支持结构感知的文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    # 遍历所有解析后的文件目录
    for file_type in ['docx', 'pdf', 'txt']:
        type_dir = os.path.join(parsed_dir, file_type)
        if not os.path.exists(type_dir):
            print(f"目录不存在: {type_dir}，跳过...")
            continue
            
        print(f"\n处理 {file_type.upper()} 文件:")
        for file_name in os.listdir(type_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(type_dir, file_name)
                original_name = os.path.splitext(file_name)[0]
                
                # 使用自定义加载器加载文档
                loader = StructuredTextLoader(file_path)
                documents = loader.load()
                
                # 为每个文件创建单独的输出目录
                file_output_dir = os.path.join(output_dir, file_type, original_name)
                os.makedirs(file_output_dir, exist_ok=True)
                
                # 保存分块结果
                chunk_data = []
                for doc_idx, doc in enumerate(documents):
                    # 使用LangChain分割器处理每个结构化文档块
                    chunks = text_splitter.split_documents([doc])
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        # 获取标题信息
                        section1 = chunk.metadata.get('section1', '')
                        section2 = chunk.metadata.get('section2', '')
                        
                        # 添加标题前缀到文本内容
                        if section1 and section2:
                            title_prefix = f"{section1}-{section2}\n\n"
                        elif section1:
                            title_prefix = f"{section1}\n\n"
                        else:
                            title_prefix = ""
                        
                        chunk_text = title_prefix + chunk.page_content
                        
                        # 创建分块文件
                        chunk_file = os.path.join(file_output_dir, f"chunk_{doc_idx+1}_{chunk_idx+1}.txt")
                        with open(chunk_file, 'w', encoding='utf-8') as f:
                            f.write(chunk_text)
                        
                        # 创建分块信息（混合方案）
                        chunk_info = {
                            'chunk_id': f"{doc_idx+1}_{chunk_idx+1}",
                            'text': chunk_text,  # 直接存储文本内容
                            'metadata': {
                                'source': file_path,  # 原始文档路径
                                'chunk_path': chunk_file,  # 分块文件路径
                                'file_type': file_type,
                                'original_name': original_name,
                                'section1': section1,
                                'section2': section2,
                                'start_index': chunk.metadata.get('start_index', 0),
                                'end_index': chunk.metadata.get('end_index', len(chunk_text))
                            }
                        }
                        
                        # 添加到JSON数据
                        chunk_data.append(chunk_info)
                
                # 保存元数据JSON
                metadata_file = os.path.join(file_output_dir, "metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
                print(f"已处理 {file_name}: 生成 {len(chunk_data)} 个分块")

def main():
    # 对已解析的文件进行分块
    parsed_dir = "parsed_document"  # 已解析文档目录
    chunk_output = "chunk_output"  # 分块结果输出目录
    
    print(f"开始处理解析后的文件分块...")
    print(f"输入目录: {parsed_dir}")
    print(f"输出目录: {chunk_output}")
    
    # 设置分块参数
    chunk_size = 500
    chunk_overlap = 100
    
    chunk_and_save_parsed_files(parsed_dir, chunk_output, chunk_size, chunk_overlap)
    print("\n所有文件分块处理完成!")

if __name__ == "__main__":
    main()