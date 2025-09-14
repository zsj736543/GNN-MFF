
import os, sys
from pathlib import Path

sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *
cfg1 = Config(sys.argv[1:])

cfg2 = cfg1
cfg2.data_pkl_path = Path("ast_tj_rtl1111.pkl").resolve()
cfg2.graph_type = "AST"

''' prepare dataset '''
TROJAN = 1
NON_TROJAN = 0

all_graphs_DFG = data_proc1.get_graphs()
print(all_graphs_DFG)
for data in all_graphs_DFG:
    if "TjFree" == data.hw_type:
        data.label = NON_TROJAN
    else:
        data.label = TROJAN

all_graphs_AST = data_proc2.get_graphs()
print(all_graphs_AST)
for data in all_graphs_AST:
    if "TjFree" == data.hw_type:
        data.label = NON_TROJAN
    else:
        data.label = TROJAN

train_graphs_AST=[]
test_graphs_AST=[]
for data_graph in all_graphs_AST:
    if "RS232" in data_graph.name:
        test_graphs_AST.append(data_graph)
    else:
        train_graphs_AST.append(data_graph)


train_graphs_DFG=[]
for data_tree in train_graphs_AST:
    for data_graph in all_graphs_DFG:
        if data_tree.name == data_graph.name:
            train_graphs_DFG.append(data_graph)

test_graphs_DFG=[]
for data_tree in test_graphs_AST:
    for data_graph in all_graphs_DFG:
        if data_tree.name == data_graph.name:
            test_graphs_DFG.append(data_graph)

train_loader_DFG = DataLoader(train_graphs_DFG, shuffle=False, batch_size=1)
valid_loader_DFG = DataLoader(test_graphs_DFG, shuffle=False, batch_size=1)
train_loader_AST = DataLoader(train_graphs_AST, shuffle=False, batch_size=1)
valid_loader_AST = DataLoader(test_graphs_AST, shuffle=False, batch_size=1)

model = AFFGCN(num_wx_features=cfg1.hidden, num_for_predict=cfg1.hidden, num_nodes=cfg1.hidden, cfg1=cfg1, cfg2=cfg2, data_proc1=data_proc1, data_proc2=data_proc2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)  #  to cuda
trainer = GraphTrainer(cfg2, class_weights=data_proc2.get_class_weights(train_graphs_AST))
trainer.build(model)
trainer.train(train_loader_AST, train_loader_DFG, valid_loader_AST, valid_loader_DFG, epochs=150)
trainer.evaluate(cfg2.epochs, train_loader_AST, train_loader_DFG, valid_loader_AST, valid_loader_DFG)
