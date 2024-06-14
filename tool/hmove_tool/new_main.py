import argparse
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import numpy as np
#import sys
#import time
import subprocess
import os
#from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
#import pandas as pd
import torch
from sklearn import preprocessing
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import tkinter as tk
#import json

from process_new import solve
from train_new import main
from Embedding import solve1
import train_new


def run_jar(jar_path, file_path):
    # 构建命令
    command = ['java', '-jar', jar_path, file_path]
    # 开启子进程执行命令
    result = subprocess.run(command, capture_output=True, text=True)
    # 打印子进程的输出
    print("Subprocess output:", result.stdout)

    # 返回子进程的返回码
    return result.returncode

if __name__ == "__main__":
    #生成graph_node.csv文件
    # python new_main.py --move_file E:\cam\WCWidth.java --start_line 13 --moveto_file E:\cam\ConsoleReader.java
    parser = argparse.ArgumentParser(description='Please enter the directory where the files are located.')
    parser.add_argument('--move_file', type=str, help='The path of the source file.')
    parser.add_argument('--start_line', type=int, help='The starting line of the method being tested.')
    parser.add_argument('--moveto_file', type=str, help='The path of the target file.')
    args = parser.parse_args()
    folder_name = solve(args.move_file, args.start_line, args.moveto_file)
    current_directory = os.getcwd()
    print('Intermediate files will be stored in ' + current_directory + '\\' + folder_name)
    print('----------------------------------------------------------------------------')

    #加载预训练模型，获得实体嵌入文件，放在与graph_node.csv同目录下
    #print("开始测试")
    arr = []
    np.set_printoptions(suppress=True)
    ss = preprocessing.StandardScaler()
    print('Start To Load Pretrain Model ... ...')
    # det5_model_path = 'Salesforce/codet5-base-multi-sum'
    # tokenizer = AutoTokenizer.from_pretrained(det5_model_path)
    # model = AutoModel.from_pretrained(det5_model_path)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum') # Salesforce/codet5-base-multi-sum
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    device = torch.device("cpu")
    model.to(device)
    print('Finish Loading Pretrain Model ! !')
    print('----------------------------------------------------------------------------')
    solve1(folder_name, args.move_file, args.moveto_file, tokenizer, model)

    #开启子进程，执行依赖提取工具ArchDiff，生成relation.csv
    # 调用函数运行JAR包
    jar_path = './ArchDiff-1.0.jar'
    return_code = run_jar(jar_path, folder_name)
    # 检查子进程的返回码
    if return_code == 0:
        print("Subprocess finished successfully")
    else:
        print(f"Subprocess failed with return code {return_code}")
    # 子进程执行完毕后，继续执行主程序
    print("Subprocess completed, continuing main program execution.")
    print('----------------------------------------------------------------------------')

    #构造混合图，进行嵌入
    embss = ['codet5', 'codegpt', 'codebert']  # , 'codet5plus', 'cotext', 'graphcodebert', 'codetrans', 'plbart'
    model_names = ['GCN_HGNN', 'GraphSAGE_HGNN', 'GraphSAGE_HGNNPLUS']
    #paths = "output_2024-06-11_18-03-06"
    result = main(folder_name, embss[0], model_names[0])
    if result == 0 :
        print('The method starting at line ' + str(args.start_line) + ' in the source class is not recommended to move to the target class.')
    else:
        print('We recommend to move the method starting at line ' + str(args.start_line) + ' in the source class to the target class.')
