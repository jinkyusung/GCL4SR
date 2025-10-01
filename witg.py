import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict




def read_item_sequences(file_path: str) -> Tuple[List[List[int]], int]:
    """
    사용자별 아이템 시퀀스 파일을 읽어 파싱합니다.

    Args:
        file_path (str): 'user_id item1,item2,...' 형식의 텍스트 파일 경로.

    Returns:
        Tuple[List[List[int]], int]: 
            - 모든 사용자의 아이템 시퀀스 목록 (e.g., [[1,2,3], [4,5]])
            - (가장 큰 아이템 ID + 1)을 의미하는 전체 노드 개수
    """
    user_sequences = []
    all_items = set()

    with open(file_path, 'r') as f:
        for line in f:
            # "user_id item1,item2,..." -> ["user_id", "item1,item2,..."]
            _, items_str = line.strip().split(' ', 1)
            items = [int(item) for item in items_str.split(',')]
            
            user_sequences.append(items)
            all_items.update(items)

    num_nodes = max(all_items) + 1 if all_items else 0
    return user_sequences, num_nodes




def convert_to_pyg_data(adjacency_list: List[Dict[int, float]], num_nodes: int) -> Data:
    """
    인접 리스트 형태의 그래프를 PyTorch Geometric(PyG)의 Data 객체로 변환합니다.

    Args:
        adjacency_list (List[Dict[int, float]]): 
            - 그래프의 인접 리스트. 
            - e.g., adj[source_node] = {target_node_1: weight_1, ...}
        num_nodes (int): 그래프의 전체 노드 개수.

    Returns:
        Data: PyG 모델에서 사용할 수 있는 그래프 데이터 객체.
    """
    edge_list = []
    weight_list = []

    # 각 노드에 대해, 이웃 노드들을 가중치(weight)가 높은 순으로 정렬
    for source_node, neighbors in enumerate(adjacency_list):
        # neighbors.items() -> [(target_node, weight), ...]
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)
        
        for target_node, weight in sorted_neighbors:
            edge_list.append([source_node, target_node])
            weight_list.append(weight)

    # PyG가 요구하는 텐서 형태로 변환
    # edge_index: [2, num_edges] 형태의 LongTensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # edge_attr: [num_edges, 1] 형태의 FloatTensor
    edge_attr = torch.tensor(weight_list, dtype=torch.float).view(-1, 1)
    
    # node_features: [num_nodes, 1] 형태, 각 노드의 ID를 특징으로 사용
    node_features = torch.arange(num_nodes, dtype=torch.long).view(-1, 1)

    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data




def build_weighted_item_transition_graph(train_sequence_file: str) -> Data:
    """
    아이템 시퀀스 데이터로부터 가중치가 있는 아이템 관계 그래프(WITG)를 생성합니다.

    Args:
        train_sequence_file (str): 학습 데이터로 사용할 시퀀스 파일 경로.

    Returns:
        Data: 완성된 가중치 그래프의 PyG Data 객체.
    """
    user_sequences, num_nodes = read_item_sequences(train_sequence_file)
    
    # 인접 리스트: adj[i]는 아이템 i와 연결된 (이웃 아이템, 가중치) 딕셔너리
    adjacency_list: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]

    # 모든 사용자의 행동 시퀀스를 순회
    for sequence in user_sequences:
        # 한 시퀀스 내에서 아이템 쌍을 추출 (윈도우 사이즈: 1, 2, 3)
        for window_size in range(1, 4):
            for i in range(len(sequence) - window_size):
    # ======================= TODO: 이 부분을 구현하세요 ========================= #
                source_item = sequence[i]
                target_item = sequence[i + window_size]
                
                # 가중치는 거리(window_size)의 역수: 가까울수록 높은 가중치
                weight = 1.0 / window_size
                
                # 양방향(undirected)으로 엣지 가중치를 더해줌
                # .get(key, 0.0)은 키가 없으면 0.0을 반환하여 KeyError 방지
                adjacency_list[source_item][target_item] = adjacency_list[source_item].get(target_item, 0.0) + weight
                adjacency_list[target_item][source_item] = adjacency_list[target_item].get(source_item, 0.0) + weight
    # =========================================================================== #
    
    # 완성된 인접 리스트를 PyG 데이터 객체로 변환
    graph_data: Data = convert_to_pyg_data(adjacency_list, num_nodes)
    return graph_data
