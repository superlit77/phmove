import os
#from random import random
#import sys
import torch
#import torch.nn as nn
#import torch.optim as optimpip
from torch.utils.data import DataLoader

# Assuming your HGCN models is defined in a file called model.py
from models import GCN_HGNN, GraphSAGE_HGNNPLUS, GraphSAGE_HGNN

# Assuming your dataset creation code is in a file called dataset.py
from dataset_new import get_dataset
#from sklearn.metrics import precision_score, recall_score, f1_score
# Assuming you have already defined your loss function, optimizer, and other parameters
# For example:
# criterion = nn.BCELoss()
# optimizer = optim.Adam(models.parameters(), lr=0.001)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training function

# Define the testing function
def test_epoch(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            g, hg, features = data[0], data[1], data[2]
            g, hg, features = g.to(device), hg.to(device), features.to(device)
            output = model(features, g, hg)
            predictions = output.item()
            #predictions = (output > 0.5).float()
            while predictions < 1.0 :
                predictions *= 10.0
            if predictions > 1.0 :
                predictions /= 10.0
            #return predictions.item()
            if predictions > 0.5:
                predictions = 1
            else:
                predictions = 0
            return predictions

# Main training and testing loop
def main(paths, emb, md):
    # Set your hyperparameters
    in_channels = 768 # Set your input feature size   256
    h_channels = 256 # Set your hidden feature size   64
    out_channels = 16 # Set your output feature size  16
    # Load the dataset
    # codebert codegpt codet5 codet5plus codetrans cotext graphcodebert plbart
    emb_type = emb  # Set your embedding type
    model_name = md
    # 'GraphSAGE_HGNN', 'GCN_HGNNPLUS', 'GraphSAGE_HGNNPLUS'
    if md == "GraphSAGE_HGNN":
        model = GraphSAGE_HGNN(in_channels, h_channels, out_channels)
    elif md == "GCN_HGNN":
        model = GCN_HGNN(in_channels, h_channels, out_channels)
    elif md == "GraphSAGE_HGNNPLUS":
        model = GraphSAGE_HGNNPLUS(in_channels, h_channels, out_channels)

    # 模型的加载要对应的内容：低阶图神经网络类型、高阶图神经网络类型、预训练模型类型、网络层隐藏维度、输出维度，这5个对应好
    if os.path.exists("models/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".pth"):
        state_dict = torch.load("models/_" + model_name + "_" + emb_type + "-" + str(h_channels) + "-" + str(out_channels) + ".pth")
        model.load_state_dict(state_dict)

    device = torch.device("cpu")
    test_set = get_dataset(paths, emb_type)
    # Convert datasets to DataLoader
    # 图的大小不一，导致无法设置batch_size，否则会runtime error
    test_loader = DataLoader(test_set, batch_size=None, shuffle=False)

    model.to(device)
    # result 为0则是不推荐move，为1推荐Move
    result = test_epoch(model, test_loader, device)

    return result

if __name__ == "__main__":
    embss = ['codet5', 'codegpt', 'codebert'] #, 'codet5plus', 'cotext', 'graphcodebert', 'codetrans', 'plbart'
    model_names = ['GCN_HGNN', 'GraphSAGE_HGNN', 'GraphSAGE_HGNNPLUS']
    paths = "output_2024"
    main(paths, embss[0], model_names[1])
#
# phmove : graphsage+hgnn+codet5 256
# fegnn:   gat+hgcn+codebert+256+16
