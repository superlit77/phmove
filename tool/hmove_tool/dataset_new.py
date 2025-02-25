# -*- encoding = utf-8 -*-
"""
@description: 构造数据集
@date: 2023/10/17
@File : dataset.py
@Software : PyCharm
"""
import os
import dhg
import torch

"""
思路：
1、首先，以一个class为中心，和它相关的move method的class都找出来，组成一个graph
2、然后，需要获取跨class的依赖结构，由此组成低阶图
3、move method之前，每一个class组成高阶图是负样本；move method之后组成高阶图是正样本，这样正负样本是平衡的
4、使用pre-trained model获取图中每个节点的特征(field是否考虑，考虑可能会变麻烦)
5、将图数据转化为DHG框架的格式
"""

test_set = []
# 构造低阶图
def create_digraph(folder_name):
    edges = []
    v = 0
    with open(folder_name + '\\graph_node.csv', "r", encoding="utf-8") as gr:
        # 查看graph_node.csv一共有几行数据
        lines = gr.readlines()
        v += len(lines)
    if not os.path.exists(folder_name + '\\relation.csv'):
        return dhg.Graph(v, edges)
    with open(folder_name + '\\relation.csv', "r", encoding="utf-8") as pr:
        for line in pr:
            line = line.strip('\n').split(',')
            edges.append([int(line[0]), int(line[1])])
    # edges = [[36,0],[37,0],...]
    return dhg.Graph(v, edges)

# 构造高阶图和要预测的超链，把它们放一起组成一个hypergraph，超链会放在超边集的最后一个位置
def create_hypergraph(folder_name):
    v = 0
    # 超边集合：edges
    edges = []
    class_groups = {} # 用于存储具有相同no_class值的行数据
    links = []
    with open(folder_name + "\\graph_node.csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            parts = line.strip('\n').split(',')
            no_id = int(parts[0])
            no_class = parts[1].split('-')[0]
            no_tag = int(parts[8])  # 假设no_tag是整数，提前转换
            v = max(v, no_id + 1)  # 确保v是节点集中最大的ID+1
            # 处理类别的节点列表
            if no_class not in class_groups:
                class_groups[no_class] = []
            class_groups[no_class].append(no_id)
            # 处理链接（假设1为需要添加到超边中的标记）
            if no_tag == 1:
                links.append(no_id)
    # 遍历字典，为每个类别的节点列表创建超边并添加到edges列表中
    for no_class, node_ids in class_groups.items():
        edge = tuple(node_ids)  # 直接将同一类别的所有节点ID作为超边
        edges.append(edge)
    # 构建基于类别的超边
    #for node_ids in class_groups.values():
     #   edge = []
      #  edge.append(tuple(node_ids))
       # edges.append(edge)
        # 将链接作为额外的超边添加到末尾，这里假设单个节点也可以构成超边，或者需要进一步处理逻辑
    edges.append(tuple(links))  # 注意，这里是将单个节点作为列表添加，根据实际需求调整
    '''
        for line in gr:
            v += 1
            line = line.strip('\n').split(',')
            no_class = line[1].split('-')[0]
            no_id = int(line[0])
            # 写逻辑去构造一条超边
            #edge = []
            # 如果no_class尚未作为键存在于字典中，则创建一个新的列表
            if no_class not in class_groups:
                class_groups[no_class] = []
            # 将节点ID添加到对应no_class的列表中
            class_groups[no_class].append(no_id)
    # 遍历字典，为每个类别的节点列表创建超边并添加到edges列表中
    for no_class, node_ids in class_groups.items():
        edge = list(node_ids)  # 直接将同一类别的所有节点ID作为超边
        print(no_class)
        edges.append(edge)
    # edges = [[0,1,2,3],[4]]
    # 构造link
    with open(folder_name + "\\graph_node.csv", 'r', encoding="utf-8") as gr2:
        for line in gr2:
            line = line.strip('\n').split(',')
            no_tag = line[8]
            link = []
            # 写逻辑去构造超链
            if (no_tag == 1):
                link.append(int(line[0]))
            edges.append(link)
    '''
    # link = [0,4]
    # 把构造的要预测的超链放到边集的末尾
    # edges = [ [0,1,2,3],[4],[1,4] ]

    #for i, edge in enumerate(edges):
        #print(f"Edge {i + 1}: {edge}")
    return dhg.Hypergraph(v, edges)


def l1_mormalize(features):
    return torch.nn.functional.normalize(features, p=1, dim=1)


# codebert codegpt codet5 codet5plus codetrans cotext graphcodebert plbart
def get_features(path, name):
    features = []
    with open(path + "\\" + name + ".csv", 'r', encoding="utf-8") as gr:
        for line in gr:
            line = [float(x) for x in line.strip('\n').split(',')]
            features.append(line)
    features_tensor = torch.tensor(features)
    # if name != "codet5plus":
    #
    features_tensor = l1_mormalize(features_tensor)
    return features_tensor


def solve(folder_name, emb_type):
    global test_set
    emb_path = folder_name + "\\embedding"
    g = create_digraph(folder_name)
    hg = create_hypergraph(folder_name)
    features = get_features(emb_path, emb_type)
    # 修改 device 为 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    test_set.append([g, hg, features])

# emb 是哪种预训练模型
def get_dataset(folder_name, emb):
    global test_set
    test_set.clear()
    #print("Prepare dataset……")
    solve(folder_name, emb)
    return test_set
