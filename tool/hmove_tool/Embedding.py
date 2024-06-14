import csv
import os
#import sys
#import time
import shutil
import pandas as pd
import torch
import numpy as np
#from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
#from sklearn import preprocessing
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import tkinter as tk
#import json

#bert_model_path = 'E:\\pythonproject\\Pretrain\\codebert-base'  # sys.argv[1]  # CodeBERT预训练模型目录E:\pythonproject\Pretrain
#gpt_model_path = 'E:\\pythonproject\\Pretrain\\CodeGPT-small-java-adaptedGPT2'
#det5plus_model_path = 'E:\\pythonproject\\Pretrain\\codet5+'   ###
#det5_model_path = 'Salesforce/codet5-base-multi-sum' #E:\\pythonproject\\Pretrain\\codet5-base-multi-sum
#trans_model_path = 'E:\\pythonproject\\Pretrain\\codeTrans'   ###
#text_model_path = 'E:\\pythonproject\\Pretrain\\cotext-2-cc'
#graph_model_path = 'E:\\pythonproject\\Pretrain\\graphcodebert-base'
#plbart_model_path = 'E:\\pythonproject\\Pretrain\\plbart-base'
# dataset_path = sys.argv[2]  # dataset目录

# 预训练
#print("开始测试")
#arr = []
#np.set_printoptions(suppress=True)
#ss = preprocessing.StandardScaler()
#print('Start To Load Pretrain Model ... ...')

#bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
#bert_model = AutoModel.from_pretrained(bert_model_path)

#gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_path)
#gpt_model = AutoModel.from_pretrained(gpt_model_path)

#det5plus_tokenizer = AutoTokenizer.from_pretrained(det5plus_model_path, trust_remote_code=True)
#det5plus_model = AutoModel.from_pretrained(det5plus_model_path, trust_remote_code=True)

#det5_tokenizer = AutoTokenizer.from_pretrained(det5_model_path)
#det5_model = AutoModel.from_pretrained(det5_model_path)
#from transformers import RobertaTokenizer, T5ForConditionalGeneration
#det5_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
#det5_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

#trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_path, skip_special_tokens=True)
#trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_path, output_hidden_states=True)

#text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
#text_model = AutoModel.from_pretrained(text_model_path)

#graph_tokenizer = AutoTokenizer.from_pretrained(graph_model_path)
#graph_model = AutoModel.from_pretrained(graph_model_path)

#plbart_tokenizer = AutoTokenizer.from_pretrained(plbart_model_path)
#plbart_model = AutoModel.from_pretrained(plbart_model_path)

#device = torch.device("cpu")
#det5_model.to(device)

#bert_model.to(device)
#gpt_model.to(device)
#det5plus_model.to(device)
#trans_model.to(device)
#text_model.to(device)
#graph_model.to(device)
#plbart_model.to(device)
#print('Finish Loading Pretrain Model ! !')

'''
def get_embedding_codebert(text):
    cl_tokens = bert_tokenizer.tokenize(text)
    tokens = [bert_tokenizer.cls_token] + cl_tokens + [bert_tokenizer.sep_token]
    print(tokens)
    tokens_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    if len(tokens) > 512:
        print('bert',len(tokens), text) # 是
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        model_output = bert_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        model_output = bert_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding


def get_embedding_codegpt(text):
    code_tokens = gpt_tokenizer.tokenize(text)
    tokens = [gpt_tokenizer.bos_token] + code_tokens + [gpt_tokenizer.eos_token]
    tokens_ids = gpt_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    while (index + 512) < len(tokens_ids):
        tk = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        embedding.extend(gpt_model(tk[None, :]).last_hidden_state[0].detach().cpu().numpy().tolist())
        index += 512
    if index < len(tokens_ids):
        tk = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        embedding.extend(gpt_model(tk[None, :]).last_hidden_state[0].detach().cpu().numpy().tolist())
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding

def get_embedding_codet5plus(text): # 是
    inputs = det5plus_tokenizer.encode(text, return_tensors="pt").to(device)
    embedding = det5plus_model(inputs)[0]
    # 将embedding转成list格式并返回
    return embedding.detach().cpu().numpy().tolist()
'''
device = torch.device("cpu")



def get_embedding_codet5(text, det5_tokenizer, det5_model):
    tokens = det5_tokenizer.tokenize(text+"</s>")
    tokens_ids = det5_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    if len(tokens) > 512:
        print('codet5',len(tokens), text) # 是
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        output = det5_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, 512).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        output = det5_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, len(tokens_ids) - index).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding
'''
def get_embedding_codetrans(text):
    tokens = trans_tokenizer.tokenize(text)
    tokens_ids = trans_tokenizer.convert_tokens_to_ids(tokens+[trans_tokenizer.eos_token])
    embedding = []
    index = 0
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        output = trans_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, 512).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        output = trans_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, len(tokens_ids) - index).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding

def get_embedding_cotext(text):
    tokens = text_tokenizer.tokenize(text+"</s>")
    tokens_ids = text_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    if len(tokens) > 512:
        print('cotext',len(tokens), text)
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        output = text_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, 512).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        output = text_model.encoder(input_ids=model_input[None, :], attention_mask=torch.ones(1, len(tokens_ids) - index).to(device), return_dict=True)
        embedding.extend(output.last_hidden_state[0].detach().cpu().numpy().tolist())
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding

def get_embedding_graphcodebert(text):
    cl_tokens = graph_tokenizer.tokenize(text)
    tokens = [graph_tokenizer.cls_token] + cl_tokens + [graph_tokenizer.sep_token]
    tokens_ids = graph_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    if len(tokens) > 512:
        print('graph',len(tokens), text)
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        model_output = graph_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        model_output = graph_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding

def get_embedding_plbart(text):
    cl_tokens = plbart_tokenizer.tokenize(text)
    tokens = [plbart_tokenizer.cls_token] + cl_tokens + [plbart_tokenizer.sep_token]
    tokens_ids = plbart_tokenizer.convert_tokens_to_ids(tokens)
    embedding = []
    index = 0
    if len(tokens) > 512:
        print('plbart', len(tokens), text)
    while (index + 512) < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:(index + 512)]).to(device)
        model_output = plbart_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
        index += 512
    if index < len(tokens_ids):
        model_input = torch.tensor(tokens_ids[index:len(tokens_ids)]).to(device)
        model_output = plbart_model(model_input[None, :]).last_hidden_state[0].detach().cpu().numpy()[0].tolist()
        embedding.extend(
            model_output)
    embedding = np.array(embedding).reshape((-1, 768)).mean(axis=0).tolist()
    return embedding
'''

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

def write_embedding(file_path, emb):
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        datas = [float(x) for x in emb]
        writer.writerow(datas)


def solve1(path, move_file, moveto_file, det5_tokenizer, det5_model):
    # 读取graph_node.csv所有行数据
    #print(path)
    if not os.path.exists(path + '\\embedding'):
        os.mkdir(path + '\\embedding')
    if os.path.exists(path + '\\embedding\\codebert.csv'):
        os.remove(path + '\\embedding\\codebert.csv')
    if os.path.exists(path + '\\embedding\\codegpt.csv'):
        os.remove(path + '\\embedding\\codegpt.csv')
    if os.path.exists(path + '\\embedding\\codet5.csv'):
        os.remove(path + '\\embedding\\codet5.csv')
    if os.path.exists(path + '\\embedding\\codet5plus.csv'):
        os.remove(path + '\\embedding\\codet5plus.csv')
    if os.path.exists(path + '\\embedding\\codetrans.csv'):
        os.remove(path + '\\embedding\\codetrans.csv')
    if os.path.exists(path + '\\embedding\\cotext.csv'):
        os.remove(path + '\\embedding\\cotext.csv')
    if os.path.exists(path + '\\embedding\\graphcodebert.csv'):
        os.remove(path + '\\embedding\\graphcodebert.csv')
    if os.path.exists(path + '\\embedding\\plbart.csv'):
        os.remove(path + '\\embedding\\plbart.csv')
    # 写一段代码，建一个新文件夹并存入两个指定绝对路径的Java文件
    # 新文件夹的路径
    new_folder_path = path + '\\source'
    # 要复制的两个Java文件的绝对路径
    java_files = [
        move_file,
        moveto_file
    ]
    # 创建新文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    # 复制Java文件到新文件夹
    for java_file in java_files:
        if os.path.exists(java_file):
            shutil.copy(java_file, os.path.join(new_folder_path, os.path.basename(java_file)))
        else:
            print(f"Warning: File {java_file} not exist，copy is skipped.")
    print("The operation is complete. All java files have been copied successfully.")
    datas = pd.read_csv(path + '\\graph_node.csv', header=None)  # (cfg + pdg) edge.txt
    for index, row in datas.iterrows():
        #print(row[1]+'   '+str(row[5])+'   '+str(row[6]))
        filename = row[1]
        e_st = int(row[5])
        e_ed = int(row[6])
        #print(filename, e_st, e_ed)
        # 遍历文件夹path + "source\\"下的所有java文件
        for file in os.listdir(path + '\\source'):
            if file != filename:
                continue
            java_file = read_file(path + '\\source\\' + file)
            code = java_file.split('\n')
            codes = [''] + code
            vec_det5 = np.zeros(768)
            # codes[99] codes[100] codes[101]
            # vec_bert = np.zeros(768)
            # vec_gpt = np.zeros(768)
            # vec_det5plus = np.zeros(256)
            # vec_trans = np.zeros(768)
            # vec_text = np.zeros(768)
            # vec_graph = np.zeros(768)
            # vec_plbart = np.zeros(768)
            line_num = 0
            # 遍历java文件中某一method的每一行代码
            for i in range(e_st, e_ed + 1):
                if codes[i].strip() == '':
                    continue
                if codes[i].strip().startswith('//'):
                    continue
                line_num += 1
                vecs_det5 = get_embedding_codet5(codes[i].strip(), det5_tokenizer, det5_model)
                vec_det5 += vecs_det5
            vec_det5 /= line_num
            write_embedding(path + '\\embedding\\' + 'codet5.csv', vec_det5)
#               vecs_bert = get_embedding_codebert(codes[i].strip())
#               vecs_gpt = get_embedding_codegpt(codes[i].strip())
#               vecs_det5plus = get_embedding_codet5plus(codes[i].strip())
#               vecs_trans = get_embedding_codetrans(codes[i].strip())
#               vecs_text = get_embedding_cotext(codes[i].strip())
#               vecs_graph = get_embedding_graphcodebert(codes[i].strip())
#               vecs_plbart = get_embedding_plbart(codes[i].strip())
#               vec_bert += vecs_bert
#               vec_gpt += vecs_gpt
#               vec_trans += vecs_trans
#               vec_text += vecs_text
#               vec_graph += vecs_graph
#               vec_plbart += vecs_plbart
#               vec_det5plus += vecs_det5plus

#           vec_bert /= line_num
#           vec_gpt /= line_num
#           vec_det5plus /= line_num
#           vec_trans /= line_num
#           vec_text /= line_num
#           vec_graph /= line_num
#           vec_plbart /= line_num
#           write_embedding(path + '\\embedding\\' + 'codegpt.csv', vec_gpt)
#           write_embedding(path + '\\embedding\\' + 'codet5plus.csv', vec_det5plus)
#           write_embedding(path + '\\embedding\\' + 'codetrans.csv', vec_trans)
#           write_embedding(path + '\\embedding\\' + 'cotext.csv', vec_text)
#           write_embedding(path + '\\embedding\\' + 'graphcodebert.csv', vec_graph)
#           write_embedding(path + '\\embedding\\' + 'plbart.csv', vec_plbart)



def judge(path):
    if os.path.exists(path):
        return True
    return False


# path目录下创建一个finish.csv文件
def create_finish_file(path):
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['finish'])

# 批处理
def scan_dir(path):
    global num
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        if os.path.isdir(file_path):
            if file_name.startswith('1') or file_name.startswith('2'):
                num += 1
                if not judge(file_path + '\\finish.csv'):
                    solve1(file_path)
                    create_finish_file(file_path + '\\finish.csv')
                    print(file_path, '+++++++++finish+++++++++++', num)
                else:
                    print(file_path, '---------exists-----------', num)
            else:
                scan_dir(file_path)


num = 0
if __name__ == '__main__':
    #scan_dir('E:/datasets')
    #scan_dir('D:\\dataset\\test')
    solve1('output_2024-06-11_18-03-06', 'E:/cam/JsdocToEs6TypedConverterTest.java', 'E:/cam/TypeDeclarationsIRFactoryTest.java')
    #emb = get_embedding_codetrans(code)
    #print(emb)
    # vec_bert = [0.0 for _ in range(768)]
    # vec_gpt = [0.0 for _ in range(768)]
    # vec_det5 = [0.0 for _ in range(768)]
    # vec_det5plus = [0.0 for _ in range(256)]
    # vec_trans = [0.0 for _ in range(768)]
    # vec_text = [0.0 for _ in range(768)]
    # vec_graph = [0.0 for _ in range(768)]
    # vec_plbart = [0.0 for _ in range(768)]
    # line_num = 0
    # for i in range(e_st, e_ed + 1):
    #     if codes[i].strip() == '':
    #         continue
    #     # 如果codes[i]是java注释，则跳过
    #     if codes[i].strip().startswith('//'):
    #         continue
    #     line_num += 1
    #     vecs_bert = get_embedding_codebert(codes[i].strip())
    #     vecs_gpt = get_embedding_codegpt(codes[i].strip())
    #     vecs_det5 = get_embedding_codet5(codes[i].strip())
    #     vecs_det5plus = get_embedding_codet5plus(codes[i].strip())
    #     vecs_trans = get_embedding_codetrans(codes[i].strip())
    #     vecs_text = get_embedding_cotext(codes[i].strip())
    #     vecs_graph = get_embedding_graphcodebert(codes[i].strip())
    #     vecs_plbart = get_embedding_plbart(codes[i].strip())
    #     # 将vecs_的各个向量的每一维度值累加至对应的vec_向量中，比如vecs_bert的768维应该累加到vec_bert的对应768维中
    #     for j in range(768):
    #         vec_bert[j] += vecs_bert[j]
    #         vec_gpt[j] += vecs_gpt[j]
    #         vec_det5[j] += vecs_det5[j]
    #         vec_trans[j] += vecs_trans[j]
    #         vec_text[j] += vecs_text[j]
    #         vec_graph[j] += vecs_graph[j]
    #         vec_plbart[j] += vecs_plbart[j]
    #     for j in range(256):
    #         vec_det5plus[j] += vecs_det5plus[j]
    # vec_bert = [x / line_num for x in vec_bert]
    # vec_gpt = [x / line_num for x in vec_gpt]
    # vec_det5 = [x / line_num for x in vec_det5]
    # vec_det5plus = [x / line_num for x in vec_det5plus]
    # vec_trans = [x / line_num for x in vec_trans]
    # vec_text = [x / line_num for x in vec_text]
    # vec_graph = [x / line_num for x in vec_graph]
    # vec_plbart = [x / line_num for x in vec_plbart]

