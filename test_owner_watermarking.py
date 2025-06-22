from datasets import Cora
from models.defense.OwnerWatermarkingDefense import OwnerWatermarkingDefense
from utils.dglTopyg import dgl_to_pyg_data

dataset = Cora()

pyg_data = dgl_to_pyg_data(dataset.graph)

defense = OwnerWatermarkingDefense(dataset)
metrics = defense.defend(pyg_data)

print(metrics)
