"""
从 GO OBO 文件构建层级图
将 GO 术语的 DAG 结构转换为边列表并保存为 pkl
边存储为 (child_idx, parent_idx) 元组列表，供后续 DGL 图构建使用
"""
import pickle
import numpy as np
from collections import defaultdict
import argparse
import pandas as pd
import os



def parse_obo_file(obo_file):
    """
    解析 GO OBO 文件 (修正版：支持 alt_id)
    """
    print(f"Parsing OBO file: {obo_file}")
    
    go_terms = {}
    relationships = defaultdict(list)
    
    current_term = None
    
    with open(obo_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line == "[Term]":
                if current_term is not None:
                    go_terms[current_term['id']] = current_term
                current_term = {
                    'id': None,
                    'name': None,
                    'namespace': None,
                    'is_obsolete': False,
                    'parents': [],
                    'alt_ids': []  # <--- 1. 新增 alt_ids 列表
                }
            
            elif line == "[Typedef]":
                if current_term is not None:
                    go_terms[current_term['id']] = current_term
                current_term = None
            
            elif current_term is not None:
                if line.startswith('id:'):
                    current_term['id'] = line.split('id:')[1].strip()
                
                elif line.startswith('alt_id:'):  # <--- 2. 解析 alt_id
                    alt_id = line.split('alt_id:')[1].strip()
                    current_term['alt_ids'].append(alt_id)
                
                elif line.startswith('name:'):
                    current_term['name'] = line.split('name:')[1].strip()
                
                elif line.startswith('namespace:'):
                    current_term['namespace'] = line.split('namespace:')[1].strip()
                
                elif line.startswith('is_obsolete:'):
                    current_term['is_obsolete'] = 'true' in line.lower()
                
                elif line.startswith('is_a:'):
                    parent_id = line.split('is_a:')[1].split('!')[0].strip()
                    current_term['parents'].append(parent_id)
                    
                elif line.startswith('relationship:'):
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] == 'part_of':
                        parent_id = parts[2]
                        current_term['parents'].append(parent_id)
    
    # 保存最后一个 term
    if current_term is not None and current_term['id'] is not None:
        go_terms[current_term['id']] = current_term
    
    # <--- 3. 关键步骤：建立 alt_id 到 term 的映射
    # 我们需要遍历一遍，把 alt_id 也加到 go_terms 字典里，指向同一个对象
    alt_id_map = {}
    for go_id, term in go_terms.items():
        for alt_id in term['alt_ids']:
            alt_id_map[alt_id] = term
            
    # 合并映射（注意：不要在遍历字典时修改字典，所以分两步）
    go_terms.update(alt_id_map)

    # 构建关系字典 (使用主ID构建，因为父子关系里存的都是主ID)
    for go_id, term_info in go_terms.items():
        # 即使是 alt_id 进来的，term_info['parents'] 也是共享的，没问题
        if not term_info['is_obsolete'] and term_info['parents']:
            relationships[go_id] = term_info['parents']
    
    print(f"Parsed {len(go_terms)} GO terms (including alt_ids)")
    
    return go_terms, relationships


def filter_go_terms(go_terms, relationships, namespace=None, min_depth=0, list_go=None):
    filtered = []
    missing_count = 0
    
    # 如果提供了 list_go，我们以 list_go 为主循环
    if list_go is not None:
        for go_id in list_go:
            if go_id in go_terms:
                filtered.append(go_id)
            else:
                # 打印前5个找不到的ID，看看是不是格式问题
                if missing_count < 5:
                    print(f"Warning: GO term {go_id} from CSV not found in OBO file.")
                missing_count += 1
    else:
        for go_id in go_terms:
            filtered.append(go_id)
            
    print(f"Total requested: {len(list_go) if list_go else 'All'}")
    print(f"Found in OBO: {len(filtered)}")
    print(f"Missing: {missing_count}")
    
    return filtered


def build_edge_list(go_terms, relationships, go_list=None):
    """
    构建边列表（存储为元组）
    Args:
        go_terms: GO 术语字典
        relationships: {child_id: [parent_ids]}
        go_list: 要包含的 GO ID 列表（如果为 None，使用所有非废弃术语）
    Returns:
        edges: [(child_idx, parent_idx), ...] 边的列表
        go_to_idx: {go_id: index} 映射
        idx_to_go: {index: go_id} 映射
    """
    # 确定要使用的 GO 术语列表
    if go_list is None:
        go_list = [go_id for go_id, info in go_terms.items() if not info['is_obsolete']]
    
    # go_list = sorted(go_list)  # 排序保证一致性
    num_terms = len(go_list)
    
    # 创建映射
    go_to_idx = {go_id: idx for idx, go_id in enumerate(go_list)}
    idx_to_go = {idx: go_id for idx, go_id in enumerate(go_list)}
    
    # 收集边（child -> parent）
    edges = []
    
    for child_id in go_list:
        if child_id not in relationships:
            continue
        
        child_idx = go_to_idx[child_id]
        
        for parent_id in relationships[child_id]:
            if parent_id in go_to_idx:
                parent_idx = go_to_idx[parent_id]
                edges.append((child_idx, parent_idx))
    
    print(f"\nEdge list created:")
    print(f"  Number of nodes: {num_terms}")
    print(f"  Number of edges: {len(edges)}")
    
    return edges, go_to_idx, idx_to_go


def compute_go_depths(edges, num_nodes):
    """
    计算每个 GO 术语的深度（从边列表）
    Args:
        edges: [(child_idx, parent_idx), ...] 边列表
        num_nodes: 节点总数
    Returns:
        depths: [num_nodes] 每个术语的深度
    """
    depths = np.zeros(num_nodes, dtype=np.int32)
    
    # 构建邻接表（用于 BFS）
    children_dict = {i: [] for i in range(num_nodes)}
    parent_count = np.zeros(num_nodes, dtype=np.int32)
    
    for child, parent in edges:
        children_dict[parent].append(child)
        parent_count[child] += 1
    
    # 找到根节点（没有父节点）
    roots = np.where(parent_count == 0)[0]
    print(f"\nFound {len(roots)} root nodes")
    
    # BFS 计算深度
    visited = set()
    queue = [(int(root), 0) for root in roots]
    
    while queue:
        node, depth = queue.pop(0)
        
        if node in visited:
            continue
        visited.add(node)
        
        depths[node] = max(depths[node], depth)
        
        # 访问所有子节点
        for child in children_dict[node]:
            queue.append((child, depth + 1))
    
    print(f"Depth statistics:")
    print(f"  Min depth: {depths.min()}")
    print(f"  Max depth: {depths.max()}")
    print(f"  Mean depth: {depths.mean():.2f}")
    
    return depths


def save_go_graph(edges, go_to_idx, idx_to_go, go_terms, depths, output_path):
    """
    保存 GO 图数据到 pkl 文件（存储边列表）
    Args:
        edges: [(child_idx, parent_idx), ...] 边列表
        go_to_idx: GO ID 到索引的映射
        idx_to_go: 索引到 GO ID 的映射
        go_terms: GO 术语信息
        depths: GO 术语深度
        output_path: 输出文件路径
    """
    num_nodes = len(idx_to_go)
    
    go_graph_data = {
        'edges': edges,  # 边列表：[(child_idx, parent_idx), ...]
        'num_nodes': num_nodes,
        'go_to_idx': go_to_idx,
        'idx_to_go': idx_to_go,
        'go_terms': {idx_to_go[i]: go_terms[idx_to_go[i]] for i in range(num_nodes)},
        'depths': depths
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(go_graph_data, f)
    
    print(f"\nGO graph data saved to: {output_path}")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of edges: {len(edges)}")
    print(f"  Storage format: edge list (tuples)")

    
def load_csv(filePath):    
    import pandas as pd
    df = pd.read_csv(filePath, header=None)
    go_list = df[1].tolist()
    return go_list

            
def quick_format_go_terms(input_csv, obo_file, output_csv):
    """
    1. 解析 OBO 获取有效的 GO ID
    2. 读取原 CSV
    3. 过滤并按顺序生成带索引的 CSV
    """
    print(f"开始处理: {input_csv}")
    
    # --- 1. 快速提取 OBO 中的有效 ID (含 alt_id) ---
    valid_go_ids = set()
    if not os.path.exists(obo_file):
        print(f"错误: 找不到文件 {obo_file}")
        return

    with open(obo_file, 'r', encoding='utf-8') as f:
        current_id = None
        is_obsolete = False
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
                is_obsolete = False
            elif line.startswith("id:"):
                current_id = line.split("id:")[1].strip()
            elif line.startswith("alt_id:"):
                alt_id = line.split("alt_id:")[1].strip()
                valid_go_ids.add(alt_id)
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
            elif line == "" and current_id and not is_obsolete:
                valid_go_ids.add(current_id)
    
    print(f"OBO 解析完成，共有 {len(valid_go_ids)} 个有效术语。")

    # --- 2. 读取并过滤你的 CSV ---
    # 假设输入 CSV 第一行是 'GO_ID'，或者直接就是 ID
    try:
        # 这里用 header=None 兼容性更强，读入后我们过滤掉可能的字符串表头
        raw_df = pd.read_csv(input_csv, header=None)
        all_input_ids = raw_df[1].tolist()
    except Exception as e:
        print(f"读取 CSV 出错: {e}")
        return

    filtered_list = []
    for go_id in all_input_ids:
        go_id = str(go_id).strip()
        if go_id in valid_go_ids:
            filtered_list.append(go_id)
        elif go_id == "GO_ID": # 跳过表头
            continue
        else:
            print(f"跳过无效/废弃 ID: {go_id}")

    # --- 3. 生成带索引的 DataFrame 并保存 ---
    # 这一步会生成你要求的格式： 0,GO:XXXXX
    output_df = pd.DataFrame({'GO_ID': filtered_list})
    
    # index=True 会把 0, 1, 2... 写入第一列
    # header=False 保证不出现 'GO_ID' 这个单词
    output_df['GO_ID'].to_csv(output_csv, index=True, header=False)

    print("-" * 30)
    print(f"处理成功！")
    print(f"原始 ID 数量: {len(all_input_ids)}")
    print(f"过滤后 ID 数量: {len(filtered_list)}")
    print(f"结果已保存至: {output_csv}")
    print("预览前 3 行:")
    print(pd.read_csv(output_csv, header=None).head(3))

    

def main():
    parser = argparse.ArgumentParser(description='Build GO graph from OBO file')
    parser.add_argument('--input', type=str, default='bp_terms.csv',
                                help='region GO ID CSV')
    parser.add_argument('--obo', type=str, default='./go.obo',
                        help='Path to GO OBO file (download from http://purl.obolibrary.org/obo/go/go-basic.obo)')
    parser.add_argument('--filtered-output', type=str, default='bp_terms_filter.csv',
                            help='filtered GO ID CSV')
    parser.add_argument('--output', type=str, default='./bp/go_parents_pair.pkl',
                        help='Output pkl file path')
    parser.add_argument('--namespace', type=str, default='biological_process',
                        choices=['biological_process', 'molecular_function', 'cellular_component'],
                        help='Filter by GO namespace (optional)')
    parser.add_argument('--min-depth', type=int, default=0,
                        help='Minimum depth to include (default: 0)')
    args = parser.parse_args(args=[])
    
    go_terms, relationships = parse_obo_file(args.obo)

    quick_format_go_terms(args.input, args.obo, args.filtered_output)

    list_go = load_csv(args.filtered_output)
    print(len(list_go))
    filtered_go_list = filter_go_terms(go_terms, relationships, 
                                       namespace=args.namespace, 
                                       min_depth=args.min_depth,
                                      list_go=list_go)
    edges, go_to_idx, idx_to_go = build_edge_list(
        go_terms, relationships, filtered_go_list
    )
    num_nodes = len(idx_to_go)
    depths = compute_go_depths(edges, num_nodes)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_go_graph(edges, go_to_idx, idx_to_go, go_terms, depths, args.output)

    print("\n" + "="*60)
    print("Sample GO Terms:")
    print("="*60)
    for i in range(min(5, len(idx_to_go))):
        go_id = idx_to_go[i]
        term_info = go_terms[go_id]
        print(f"{i}: {go_id} - {term_info['name']}")
        print(f"   Namespace: {term_info['namespace']}")
        print(f"   Depth: {depths[i]}")
        print(f"   Parents: {term_info['parents']}")
        print()


if __name__ == "__main__":
    main()
