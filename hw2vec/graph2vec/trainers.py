
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))

import torch.optim as optim
import numpy as np

import warnings

warnings.filterwarnings('ignore')

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from hw2vec.graph2vec.models import *


class BaseTrainer:
    def __init__(self, cfg):
        self.config = cfg
        self.best_test_loss = np.Inf
        self.min_test_loss = np.Inf
        self.best_accuracy = 0
        self.task = None
        self.metrics = {}
        self.model = None
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def predict(self, data_loader_ast, data_loader_dfg):
        self.model.eval()
        all_labels, all_probs = [], []
        device = self.device

        with torch.no_grad():
            for batch_ast, batch_dfg in zip(data_loader_ast, data_loader_dfg):
                batch_ast = batch_ast.to(device)
                batch_dfg = batch_dfg.to(device)

                logits, _ = self.model(batch_ast, batch_dfg)  # forward
                probs = torch.softmax(logits, dim=-1)[:, 1]  # 取类别=1的概率

                all_labels.append(batch_ast.y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)

        return y_true, y_prob

    def build(self, model, path=None):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=5e-4)

    def metric_calc(self, loss, labels, preds, header):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        conf_mtx = str(confusion_matrix(labels, preds)).replace('\n', ',')
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")

        self.metric_print(loss, acc, f1, conf_mtx, precision, recall, header)

        if header == "test " and (self.best_test_loss >= loss):
            self.min_test_loss = loss
            self.metrics["acc"] = acc
            self.metrics["f1"] = f1
            self.metrics["conf_mtx"] = conf_mtx
            self.metrics["precision"] = precision
            self.metrics["recall"] = recall

    def metric_print(self, loss, acc, f1, conf_mtx, precision, recall, header):
        print("%s loss: %4f" % (header, loss) +
              ", %s accuracy: %.4f" % (header, acc) +
              ", %s f1 score: %.4f" % (header, f1) +
              ", %s confusion_matrix: %s" % (header, conf_mtx) +
              ", %s precision: %.4f" % (header, precision) +
              ", %s recall: %.4f" % (header, recall))



class GraphTrainer(BaseTrainer):

    def __init__(self, cfg, class_weights=None):
        super().__init__(cfg)
        self.task = "TJ"
        self.device = cfg.device
        if class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights.float().to(cfg.device)


    def predict(self, data_loader_ast, data_loader_dfg):
        self.model.eval()
        all_labels = []
        all_probs = []

        device = self.config.device

        with torch.no_grad():
            for data_ast in data_loader_ast:
                for data_dfg in data_loader_dfg:
                    # 确保AST和DFG匹配
                    if data_ast.name == data_dfg.name:
                        data_ast = data_ast.to(device)
                        data_dfg = data_dfg.to(device)

                        # 前向计算
                        output, _ = self.model.forward(data_ast.x, data_ast.edge_index,
                                                       data_dfg.x, data_dfg.edge_index,
                                                       data_ast.batch, data_dfg.batch)
                        output = self.model.mlp(output)
                        probs = torch.softmax(output, dim=1)  # 概率

                        all_probs.append(probs.cpu())
                        all_labels.append(data_ast.label.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()

        y_pred = y_prob.argmax(axis=1)

        return y_true, y_prob

    def train(self, data_loader, data_loader_graph, valid_data_loader, valid_data_loader_graph, epochs):
        tqdm_bar = tqdm(range(epochs))

        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            for data in data_loader:
                for data2 in data_loader_graph:
                    if data.name==data2.name:
                        self.optimizer.zero_grad()
                        data = data.to(device)
                        data2 = data2.to(device)

                        loss_train = self.train_epoch_tj(data,data2)
                        loss_train.backward()
                        self.optimizer.step()
                        acc_loss_train += loss_train.detach().cpu().numpy()

            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))

            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx, data_loader, data_loader_graph, valid_data_loader, valid_data_loader_graph)

    # @profileit
    def train_epoch_tj(self, data, data2):
        output, _ = self.model(data.x, data.edge_index, data2.x, data2.edge_index, data.batch, data2.batch)
        output = self.model.mlp(output)
        loss_train = self.loss_func(output, data.label)
        return loss_train

    # @profileit
    def inference_epoch_tj(self, data, data2):
        output, attn = self.model.forward(data.x, data.edge_index, data2.x, data2.edge_index, data.batch, data2.batch)
        output = self.model.mlp(output)
        loss = self.loss_func(output, data.label)
        return loss, output, attn

    def inference(self, data_loader, data_loader_graph):
        labels = []
        outputs = []
        node_attns = []
        total_loss = 0
        folder_names = []

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(data_loader):
                for j, data2 in enumerate(data_loader_graph):
                    if data.name == data2.name:
                        data = data.to(self.config.device)
                        data2 = data2.to(self.config.device)
                        loss, output, attn = self.inference_epoch_tj(data, data2)
                        total_loss += loss.detach().cpu().numpy()

                        outputs.append(output.cpu())

                        labels += np.split(data.label.cpu().numpy(), len(data.label.cpu().numpy()))

            outputs = torch.cat(outputs).reshape(-1, 2).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = torch.LongTensor(labels).detach()
            outputs_tensor = torch.FloatTensor(outputs).detach()
            preds = outputs_tensor.max(1)[1].type_as(labels_tensor).detach()

        return avg_loss, labels_tensor, outputs_tensor, preds, node_attns

    def evaluate(self, epoch_idx, data_loader, data_loader_graph, valid_data_loader, valid_data_loader_graph):
        train_loss, train_labels, train_outputs_tensor, train_preds, train_node_attns = self.inference(data_loader, data_loader_graph)
        test_loss, test_labels, test_outputs_tensor, test_preds, test_node_attns = self.inference(valid_data_loader,
                                                                                valid_data_loader_graph)

        print(f"\nMini Test for Epochs {epoch_idx}:")

        self.metric_calc(train_loss, train_labels, train_preds, header="train")
        self.metric_calc(test_loss, test_labels, test_preds, header="test ")


class Evaluator(BaseTrainer):
    def __init__(self, cfg, task):
        super().__init__(cfg)
        self.task = task




