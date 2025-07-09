import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_and_save_parsed_files(parsed_dir="parsed_document", output_dir="chunk_output", chunk_size=500, chunk_overlap=100):
    """处理已解析的文档文件，进行结构感知分块并保存"""
    os.makedirs(output_dir, exist_ok=True)
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
                
                # 读取解析后的文本内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 创建文档对象
                document = {
                    'text': content,
                    'metadata': {
                        'source': file_path,
                        'file_type': file_type,
                        'original_name': os.path.splitext(file_name)[0]
                    }
                }
                
                # 结构感知分块处理
                chunks = structure_aware_chunking(document['text'], text_splitter)
                
                # 为每个文件创建单独的输出目录
                file_output_dir = os.path.join(output_dir, file_type, os.path.splitext(file_name)[0])
                os.makedirs(file_output_dir, exist_ok=True)
                
                # 保存分块结果
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        'chunk_id': i + 1,
                        'text': chunk['text'],
                        'metadata': {
                            **document['metadata'],
                            'section1': chunk.get('section1', ''),
                            'section2': chunk.get('section2', ''),
                            'start_index': chunk.get('start_index', 0),
                            'end_index': chunk.get('end_index', len(document['text']))
                        }
                    }
                    
                    # 保存为文本文件
                    chunk_file = os.path.join(file_output_dir, f"chunk_{i+1}.txt")
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(chunk['text'])
                    
                    # 添加到JSON数据
                    chunk_data.append(chunk_info)
                
                # 保存元数据JSON
                metadata_file = os.path.join(file_output_dir, "metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
                print(f"已处理 {file_name}: 生成 {len(chunks)} 个分块")

def structure_aware_chunking(text, text_splitter):
    """根据文档结构进行智能分块"""
    chunks = []
    lines = text.split('\n')
    
    current_section1 = ""
    current_section2 = ""
    current_content = []
    
    # 正则表达式匹配标题标签
    section1_pattern = re.compile(r'^\[SECTION_1\](.+?)\[/SECTION_1\]$')
    section2_pattern = re.compile(r'^\[SECTION_2\](.+?)\[/SECTION_2\]$')
    
    for line in lines:
        # 检查一级标题
        section1_match = section1_pattern.match(line)
        if section1_match:
            # 保存当前块（如果有内容）
            if current_content:
                chunks.append(create_chunk(current_section1, current_section2, current_content, text_splitter))
                current_content = []
            
            current_section1 = section1_match.group(1).strip()
            current_section2 = ""  # 重置二级标题
            continue
        
        # 检查二级标题
        section2_match = section2_pattern.match(line)
        if section2_match:
            # 保存当前块（如果有内容）
            if current_content:
                chunks.append(create_chunk(current_section1, current_section2, current_content, text_splitter))
                current_content = []
            
            current_section2 = section2_match.group(1).strip()
            continue
        
        # 普通内容行
        if line.strip():  # 跳过空行
            current_content.append(line)
    
    # 处理最后一个块
    if current_content:
        chunks.append(create_chunk(current_section1, current_section2, current_content, text_splitter))
    
    return chunks

def create_chunk(section1, section2, content_lines, text_splitter):
    """创建结构感知的文本块"""
    # 组合内容
    content_text = '\n'.join(content_lines)
    
    # 如果内容很短，直接作为一个块
    if len(content_text) <= 500:
        # 添加标题信息
        title_prefix = ""
        if section1 and section2:
            title_prefix = f"{section1}-{section2}\n\n"
        elif section1:
            title_prefix = f"{section1}\n\n"
        
        return {
            'text': title_prefix + content_text,
            'section1': section1,
            'section2': section2
        }
    
    # 如果内容较长，使用文本分割器
    chunks = text_splitter.split_text(content_text)
    
    # 为每个分割块添加标题信息
    structured_chunks = []
    for chunk in chunks:
        title_prefix = ""
        if section1 and section2:
            title_prefix = f"{section1}-{section2}\n\n"
        elif section1:
            title_prefix = f"{section1}\n\n"
        
        structured_chunks.append({
            'text': title_prefix + chunk,
            'section1': section1,
            'section2': section2
        })
    
    return structured_chunks

def main():
    # 对已解析的文件进行分块
    parsed_dir = "parsed_document"  # 已解析文档目录
    chunk_output = "chunk_output"  # 分块结果输出目录
    
    print(f"开始处理解析后的文件分块...")
    print(f"输入目录: {parsed_dir}")
    print(f"输出目录: {chunk_output}")
    
    chunk_and_save_parsed_files(parsed_dir, chunk_output)
    print("\n所有文件分块处理完成!")

if __name__ == "__main__":
    main()