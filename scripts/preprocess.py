#coding: utf-8
import sys
import os

'''
全半角转换
大小写转换
只留下 query\t title1 \t title2 \n
'''


def strQ2B(ustring):
    '''全转半'''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring



root = '/search/ffz/projects/bigdata_qt_match/data/train_data/'
new_root = '/search/ffz/projects/bigdata_qt_match/data/train_data_new/'
file_list = os.listdir(root)
for i in file_list:
    with open(root+i, "r") as in_f:
        out_f = open(new_root+i, 'w')
        for line in in_f:
            line = line.strip().split('\t')
            query = strQ2B(line[0].strip().lower())
            title1= strQ2B(line[1].strip().lower())
            title2 = strQ2B(line[5].strip().lower())
            out_f.write(query + '\t' + title1 + '\t' + title2 + '\n')
        out_f.close()

