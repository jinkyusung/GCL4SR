import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Set




def recall_at_k(actual: List[list], predicted: List[list], topk: int) -> float:
    """
    추천 시스템의 평균 Recall@k를 계산합니다.

    Recall@k는 사용자가 실제로 관심 있었던 전체 아이템 중에서,
    모델이 추천한 상위 K개의 아이템이 얼마나 포함하고 있는지를 나타내는 지표입니다.

    Args:
        actual (List[list]): 사용자별 실제 정답 아이템 목록.
                             e.g., [[1, 2, 3], [4, 5]]
        predicted (List[list]): 모델이 추천한 사용자별 아이템 목록 (점수 순으로 정렬됨).
                                e.g., [[1, 5, 2], [4, 8, 9]]
        topk (int): 평가에 사용할 상위 K개의 추천 개수.

    Returns:
        float: 모든 사용자의 Recall@k 점수를 산술 평균한 값.
    """
    total_recall_score = 0.0
    num_evaluated_users = 0  # 평가가 가능한 사용자(정답 아이템이 있는) 수

    # ============================== TODO: 이 부분을 구현하세요 ================================ #
    # 모든 사용자에 대해 반복
    for user_id in range(len(predicted)):
        ground_truth_items: Set[int] = set(actual[user_id])
        
        # 평가를 위해선 사용자의 정답 아이템이 최소 1개 이상이어야 함
        if not ground_truth_items:
            continue
            
        num_evaluated_users += 1
        
        # 모델이 추천한 상위 topk개의 아이템 목록
        recommended_items: Set[int] = set(predicted[user_id][:topk])
        
        # 추천된 아이템 중 정답 아이템의 개수 (적중 개수)
        num_hits = len(ground_truth_items.intersection(recommended_items))
        
        # 현재 사용자의 Recall@k 점수 계산: (적중 개수) / (전체 정답 개수)
        recall_score = num_hits / len(ground_truth_items)
        total_recall_score += recall_score
    # ======================================================================================= #
    
    # 평가된 모든 사용자의 점수를 평균내어 최종 점수 계산
    mean_recall = total_recall_score / num_evaluated_users if num_evaluated_users > 0 else 0.0
    return mean_recall




def ndcg_at_k(actual: List[list], predicted: List[list], topk: int) -> float:
    """
    추천 시스템의 평균 NDCG@k를 계산합니다.

    NDCG@k는 추천 결과의 순서까지 고려하는 정밀한 평가 지표입니다.
    정답 아이템이 추천 목록의 앞쪽에 있을수록 높은 점수를 받습니다.

    Args:
        actual (List[list]): 사용자별 실제 정답 아이템 목록.
        predicted (List[list]): 모델이 추천한 사용자별 아이템 목록 (점수 순으로 정렬됨).
        topk (int): 평가에 사용할 상위 K개의 추천 개수.

    Returns:
        float: 모든 사용자의 NDCG@k 점수를 산술 평균한 값.
    """
    total_ndcg_score = 0.0
    num_evaluated_users = 0  # 평가가 가능한 사용자 수
    # ============================== TODO: 이 부분을 구현하세요 ================================ #
    for user_id in range(len(predicted)):
        ground_truth_items: Set[int] = set(actual[user_id])
        
        if not ground_truth_items:
            continue
            
        num_evaluated_users += 1
        
        # 1. DCG@k (Discounted Cumulative Gain) 계산
        dcg_score = 0.0
        # 추천 순위(rank)는 1부터 시작
        for rank, item_id in enumerate(predicted[user_id][:topk], 1):
            # 추천한 아이템이 정답 목록에 있다면, 순위에 따라 점수를 할인하여 더함
            if item_id in ground_truth_items:
                dcg_score += 1.0 / math.log2(rank + 1)
        
        # 2. IDCG@k (Ideal DCG) 계산 (가장 이상적인 추천일 경우의 DCG 값)
        idcg_score = _calculate_idcg(min(topk, len(ground_truth_items)))
        
        # 3. NDCG@k 계산 및 누적 (0으로 나누는 경우 방지)
        if idcg_score > 0:
            ndcg_score = dcg_score / idcg_score
            total_ndcg_score += ndcg_score
    # ======================================================================================= #
    mean_ndcg = total_ndcg_score / num_evaluated_users if num_evaluated_users > 0 else 0.0
    return mean_ndcg




def _calculate_idcg(k: int) -> float:
    """
    IDCG@k (Ideal Discounted Cumulative Gain) 값을 계산하는 헬퍼(helper) 함수.
    가장 이상적인 추천, 즉 상위 k개의 아이템이 모두 정답이라고 가정했을 때의 DCG 점수입니다.

    Args:
        k (int): IDCG를 계산할 목록의 길이.

    Returns:
        float: IDCG@k 점수.
    """
    # 순위(rank) 1부터 k까지의 할인된 점수를 모두 더함
    ideal_dcg = sum([1.0 / math.log2(rank + 1) for rank in range(1, k + 1)])
    return ideal_dcg




class Trainer:
    def __init__(self, model, optimizer, sample_size, hidden_size, train_matrix):
        self.model = model
        self.optimizer = optimizer
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.train_matrix = train_matrix
        self.model.to(self.model.device)

    def get_scores(self, answers, pred_list):
        recall_10 = recall_at_k(answers, pred_list, 10)
        recall_20 = recall_at_k(answers, pred_list, 20)
        ndcg_10 = ndcg_at_k(answers, pred_list, 10)
        ndcg_20 = ndcg_at_k(answers, pred_list, 20)
        return recall_10, recall_20, ndcg_10, ndcg_20

    def predict(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def train_step(self, epoch, train_dataloader):
        self.model.train()
        main_loss_sum = 0.0
        gcl_loss_sum = 0.0
        mmd_loss_sum = 0.0

        for _, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            batch = tuple(t.to(self.model.device) for t in batch)
            loss, main_loss, gcl_loss, mmd_loss = self.model.calculate_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            main_loss_sum += main_loss.item()
            gcl_loss_sum += gcl_loss.item()
            mmd_loss_sum += mmd_loss.item()

        main_loss_avg = main_loss_sum / len(train_dataloader)
        gcl_loss_avg = gcl_loss_sum / len(train_dataloader)
        mmd_loss_avg = mmd_loss_sum / len(train_dataloader)

        return main_loss_avg, gcl_loss_avg, mmd_loss_avg

    def eval_step(self, dataloader, test_matrix):
        self.model.eval()
        pred_list = None
        answer_list = None

        for i, batch in tqdm(enumerate(dataloader), desc="Evaluate", total=len(dataloader)):
            batch = tuple(t.to(self.model.device) for t in batch)
            user_ids = batch[0]
            answers = batch[2]
            recommend_output = self.model.eval_stage(batch)
            answers = answers.view(-1, 1)

            rating_pred = self.predict(recommend_output)
            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            rating_pred[test_matrix[batch_user_index].toarray() > 0] = 0
            ind = np.argpartition(rating_pred, -20)[:, -20:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

        recall_10, recall_20, ndcg_10, ndcg_20 = self.get_scores(answer_list, pred_list)
        return recall_10, recall_20, ndcg_10, ndcg_20



