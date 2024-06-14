# -*- encoding = utf-8 -*-
"""
@description: 使用javalang处理出数据集中相关类的属性
@date: 2023/10/30
@File : process.py
@Software : PyCharm
"""
#import csv
import os
import re
#import shutil

import javalang
import numpy as np
import pandas as pd
#import argparse
import datetime


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

all = 0
# Using Javalang to obtain the method/field code range in each class
def solve(file_path, st_line, target_path):
    datas = pd.DataFrame({
        'id': pd.Series([], dtype=np.int64),
        'filename':  pd.Series([], dtype='str'),
        'class-entity': pd.Series([], dtype='str'),
        'c_st': pd.Series([], dtype=np.int64),
        'c_ed': pd.Series([], dtype=np.int64),
        'e_st': pd.Series([], dtype=np.int64),
        'e_ed': pd.Series([], dtype=np.int64),
        'tag': pd.Series([], dtype=np.int64),
        'ph_move': pd.Series([], dtype=np.int64)
#        'label': pd.Series([], dtype=np.int64)
    }, columns=['id', 'filename', 'class-entity', 'c_st', 'c_ed', 'e_st', 'e_ed', 'tag', 'ph_move'])
    # tag 为1表示field, tag 为2表示method
    # print(file_path)
    # 读取Java文件内容
    idx = 0
    def split(path, idx, filename, type):
        global all
        data = pd.DataFrame({
            'id': pd.Series([], dtype=np.int64),
            'filename':  pd.Series([], dtype='str'),
            'class-method': pd.Series([], dtype='str'),
            'c_st': pd.Series([], dtype=np.int64),
            'c_ed': pd.Series([], dtype=np.int64),
            'e_st': pd.Series([], dtype=np.int64),
            'e_ed': pd.Series([], dtype=np.int64),
            'tag': pd.Series([], dtype=np.int64),
            'ph_move': pd.Series([], dtype=np.int64)
            #'label': pd.Series([], dtype=np.int64)
        }, columns=['id', 'filename', 'class-entity', 'c_st', 'c_ed', 'e_st', 'e_ed', 'tag', 'ph_move'])
        java_file = read_file(path)

        # 解析Java代码
        try:
            tree = javalang.parse.parse(java_file)
        except javalang.parser.JavaSyntaxError as e:
            print(f"A syntax error occurred while parsing Java code: {e}")
            all += 1
            return data, idx, False
        code = java_file.split('\n')
        codes = [''] + code
        have_target = False
        no_class_name = '-1'
        # 筛选出类成员中的方法和字段
        for path, node in tree.filter(javalang.tree.TypeDeclaration):
            # 找到类声明
            # print(path, node.body)
            if isinstance(node, javalang.tree.ClassDeclaration) or isinstance(node, javalang.tree.InterfaceDeclaration) or isinstance(node, javalang.tree.EnumDeclaration):
                # 遍历类成员
                enc = node.body.declarations if isinstance(node, javalang.tree.EnumDeclaration) else node.body
                c_st = node.position.line
                c_ed = c_st
                l, r = 0, 0
                for i in range(c_st, len(codes)):
                    l += codes[i].count('{')
                    r += codes[i].count('}')
                    if l == r and l != 0:
                        c_ed = i
                        break
                for member in enc:
                    # 找到方法声明
                    no_class = node.name
                    if no_class_name == '-1':
                        no_class_name = no_class
                    if isinstance(member, javalang.tree.MethodDeclaration) or isinstance(member, javalang.tree.ConstructorDeclaration):
                        no_entity = member.name
                        start_line = member.position.line
                        end_line = member.position.line
                        l, r = 0, 0
                        for i in range(start_line, len(codes)):
                            l += codes[i].count('{')
                            r += codes[i].count('}')
                            if l == r and l != 0:
                                end_line = i
                                break
                        flag = 0
                        if have_target and ((lines[0] >= start_line and lines[1] <= end_line) or (
                                    lines[0] <= start_line and lines[1] >= end_line)):
                            continue
                        if type == 0:
                            if st_line == start_line:
                                new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 2, 'ph_move': 1}
                            else:
                                new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 2, 'ph_move': 0}
                        else:
                            new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 2, 'ph_move': 1}
                        data.loc[len(data)] = new_row
                        idx += 1
                    elif isinstance(member, javalang.tree.FieldDeclaration):
                        no_entity = member.declarators[0].name
                        start_line = member.position.line
                        end_line = member.position.line
                        lc = 0
                        rc = 0
                        for i in range(start_line, len(codes)):
                            lc += len(re.findall('{', codes[i]))
                            rc += len(re.findall('}', codes[i]))
                            if re.search(';', codes[i]) and lc == rc:
                                end_line = i
                                break
                        if type == 0:
                            if st_line == start_line:
                                new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 1, 'ph_move': 1}
                            else:
                                new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 1, 'ph_move': 0}
                        else:
                            new_row = {'id': idx, 'filename': filename, 'class-entity': no_class + '-' + no_entity, 'c_st': c_st, 'c_ed': c_ed, 'e_st': start_line, 'e_ed': end_line, 'tag': 1, 'ph_move': 1}
                        data.loc[len(data)] = new_row
                        idx += 1
        return data, idx, have_target, no_class_name

    try:
        with open(file_path, 'r') as file:
            filename = os.path.basename(file_path)
            data, idx, target_tag, _ = split(file_path, idx, filename, 0)
            datas = pd.concat([datas, data], ignore_index=True)
    except FileNotFoundError:
        print("The source file was not found, check the file path.")
    except PermissionError:
        print("There is no access to the source file.")
    except Exception as e:
        print(f"An unknown error occurred while reading the source file: {e}")
    try:
        with open(target_path, 'r') as file:
            filename = os.path.basename(target_path)
            data, idx, target_tag, _ = split(target_path, idx, filename, 1)
            datas = pd.concat([datas, data], ignore_index=True)
    except FileNotFoundError:
        print("The target file was not found, check the file path.")
    except PermissionError:
        print("There is no access to the target file.")
    except Exception as e:
        print(f"An unknown error occurred while reading the target file: {e}")
    # datas.to_csv(outfile_path + '\\graph_node.csv', sep=',', header=False, index=False, encoding='utf-8')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"output_{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    datas.to_csv(folder_name+'\\graph_node.csv', sep=',', header=False, index=False, encoding='utf-8')
    datas.drop(datas.index, inplace=True)
    return folder_name

#parser = argparse.ArgumentParser(description='输入文件所处目录')
#parser.add_argument('--move_file', type=str, help='被移动文件的路径')
#parser.add_argument('--start_line', type=int, help='代码起始行')
#parser.add_argument('--moveto_file', type=str, help='移动至文件的路径')
#args = parser.parse_args()
#solve(args.move_file,args.start_line,args.moveto_file)
#print('Finish!')