import math
import torch
import numpy as np
from tqdm import tqdm



def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users



def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))



def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res



class Trainer:
    def __init__(self, model):
        self.model = model
        if self.model.cuda_condition:
            self.model.to(self.model.device)

    def get_scores(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.model.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def predict(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred



class GCL4SR_Train(Trainer):
    def __init__(self, model, optimizer, sample_size, hidden_size, train_matrix):
        super(GCL4SR_Train, self).__init__(model)
        self.optimizer = optimizer
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.train_matrix = train_matrix

    def train_stage(self, epoch, train_dataloader):

        train_data_iter = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader))

        self.model.train()

        loss_sum = 0.0

        for _, batch in train_data_iter:
            batch = tuple(t.to(self.model.device) for t in batch)

            loss = self.model.loss_fn(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

        loss_avg = loss_sum / len(train_data_iter)
        print(f"loss_avg: {loss_avg:.4f}")

    def eval_stage(self, epoch, dataloader, test=True):
        str_code = "test" if test else "eval"
        rec_data_iter = tqdm(enumerate(dataloader), desc="Recommendation EP_%s:%d" % (str_code, epoch), total=len(dataloader))
        
        self.model.eval()
        pred_list = None
        answer_list = None

        for i, batch in rec_data_iter:
            batch = tuple(t.to(self.model.device) for t in batch)
            user_ids = batch[0]
            answers = batch[2]
            recommend_output = self.model.eval_stage(batch)
            answers = answers.view(-1, 1)

            rating_pred = self.predict(recommend_output)

            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            rating_pred[self.train_matrix[batch_user_index].toarray() > 0] = 0
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
                
        return self.get_scores(epoch, answer_list, pred_list)
