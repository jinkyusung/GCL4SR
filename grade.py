import torch
import math
from torch_geometric.data import Data
from numpy import lexsort



def grade_witg(student_graph: Data, answer_graph: Data):
    print("\n--- WITG SANITY CHECK ---")
    
    if student_graph.num_nodes == answer_graph.num_nodes:
        print(f"Number of nodes: Match ({student_graph.num_nodes})")
    else:
        print(f"Number of nodes: Mismatch (Student: {student_graph.num_nodes}, Answer: {answer_graph.num_nodes})")

    try:
        # Sort student graph edges
        student_perm = lexsort(keys=(student_graph.edge_index[1].cpu().numpy(), student_graph.edge_index[0].cpu().numpy()))
        student_sorted_edges = student_graph.edge_index[:, student_perm]
        student_sorted_attrs = student_graph.edge_attr[student_perm]

        # Sort answer graph edges
        answer_perm = lexsort(keys=(answer_graph.edge_index[1].cpu().numpy(), answer_graph.edge_index[0].cpu().numpy()))
        answer_sorted_edges = answer_graph.edge_index[:, answer_perm]
        answer_sorted_attrs = answer_graph.edge_attr[answer_perm]

        # Compare edge structures
        if torch.equal(student_sorted_edges, answer_sorted_edges):
            print("Edge connectivity structure: Match")
        else:
            print("Edge connectivity structure: Mismatch")

        # Compare edge attributes (weights)
        if torch.allclose(student_sorted_attrs, answer_sorted_attrs):
            print("Edge weights: Match")
        else:
            print("Edge weights: Mismatch")
            
    except Exception as e:
        print(f"Error during edge comparison: {e}")



def grade_loss(student_loss):
    answer_loss = (9.9495, 15.3257, 0.2743)
    ans_main_loss, ans_gcl_loss, ans_mmd_loss = answer_loss
    
    try:
        stu_main_loss, stu_gcl_loss, stu_mmd_loss = student_loss
        float(stu_main_loss), float(stu_gcl_loss), float(stu_mmd_loss)
    except (TypeError, ValueError):
        print("\n--- LOSS SANITY CHECK FAILED ---")
        print("Error: student_loss must contain three numerical values.")
        print(f"Received input: {student_loss}")
        return

    print("\n--- LOSS SANITY CHECK ---")
    print("Loss 값은 GPU 하드웨어에 따라 약간에 오차가 발생할 수 있습니다. 이는 감안하여 채점할 것이니 걱정하지 않으셔도 됩니다.")
    print("이 함수는 구현이 잘되었는지 간단히 확인하는 용도로 제공하는 함수입니다!")

    TOLERANCE = 1e-3

    if math.isclose(ans_main_loss, stu_main_loss, rel_tol=TOLERANCE):
        print(f"Main Loss: Match (Answer: {ans_main_loss:.4f}, Student: {stu_main_loss:.4f})")
    else:
        print(f"Main Loss: Mismatch (Answer: {ans_main_loss:.4f}, Student: {stu_main_loss:.4f})")

    if math.isclose(ans_gcl_loss, stu_gcl_loss, rel_tol=TOLERANCE):
        print(f"GCL Loss:  Match (Answer: {ans_gcl_loss:.4f}, Student: {stu_gcl_loss:.4f})")
    else:
        print(f"GCL Loss:  Mismatch (Answer: {ans_gcl_loss:.4f}, Student: {stu_gcl_loss:.4f})")

    if math.isclose(ans_mmd_loss, stu_mmd_loss, rel_tol=TOLERANCE):
        print(f"MMD Loss:  Match (Answer: {ans_mmd_loss:.4f}, Student: {stu_mmd_loss:.4f})")
    else:
        print(f"MMD Loss:  Mismatch (Answer: {ans_mmd_loss:.4f}, Student: {stu_mmd_loss:.4f})")



def grade_eval(recall_10: float, recall_20: float, ndcg_10: float, ndcg_20: float):
    answer_metrics = {
        "Recall@10": 0.0082,
        "Recall@20": 0.0143,
        "NDCG@10":   0.0040,
        "NDCG@20":   0.0056
    }

    student_metrics = {
        "Recall@10": recall_10,
        "Recall@20": recall_20,
        "NDCG@10":   ndcg_10,
        "NDCG@20":   ndcg_20
    }
    print("\n--- EVALUATION METRICS SANITY CHECK ---")
    print("마찬가지로 Metric 값도 GPU 하드웨어에 따라 약간에 오차가 발생할 수 있습니다. 이는 감안하여 채점할 것이니 걱정하지 않으셔도 됩니다.")
    print("이 함수는 구현이 잘되었는지 간단히 확인하는 용도로 제공하는 함수입니다!")

    TOLERANCE = 1e-3

    for metric_name, ans_value in answer_metrics.items():
        stu_value = student_metrics[metric_name]
        display_name = f"{metric_name}:".ljust(12)
        if math.isclose(ans_value, stu_value, rel_tol=TOLERANCE):
            print(f"{display_name} Match   (Answer: {ans_value:.4f}, Student: {stu_value:.4f})")
        else:
            print(f"{display_name} Mismatch (Answer: {ans_value:.4f}, Student: {stu_value:.4f})")
