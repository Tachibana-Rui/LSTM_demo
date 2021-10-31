import pandas as pd
import re
import string
import numpy as np
import rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

def bleu(candidate_token, reference_token):
    """
    :param candidate_set:
    :param reference_set:
    :description:
    最简单的计算方法是看candidate_sentence 中有多少单词出现在参考翻译中, 重复的也需要计算. 计算出的数量作为分子
    分母是候选句子中的单词数量
    :return: 候选句子单词在参考句子中出现的次数/候选句子单词数量
    """
    # 分母是候选句子中单词在参考句子中出现的次数 重复出现也要计算进去
    candidate_token=str(candidate_token)
    reference_token=str(reference_token)
    count = 0
    for token in candidate_token:
        if token in reference_token:
            count += 1
    a = count
    # 计算候选翻译的句子中单词的数量
    b = len(candidate_token)
    return a/b


def calculate_average(precisions, weights):
    """Calculate the geometric weighted mean."""
    tmp_res = 1
    for id, item in enumerate(precisions):
        tmp_res = tmp_res*np.power(item, weights[id])
    tmp_res = np.power(tmp_res, np.sum(weights))
    return tmp_res


def calculate_candidate(gram_list, candidate):
    """Calculate the count of gram_list in candidate."""
    gram_sub_str = ' '.join(gram_list)
    return len(re.findall(gram_sub_str, candidate))


def calculate_reference(gram_list, references):
    """Calculate the count of gram_list in references"""
    gram_sub_str = ' '.join(gram_list)
    gram_count = []
    for item in references:
        # calculate the count of the sub string
        gram_count.append(len(re.findall(gram_sub_str, item)))
    return gram_count

f_read=open(r'D:\Documents\DS105\NEW\summary.csv',encoding = 'utf-8')
data=pd.read_csv(f_read)
print(data.tail())
#data['tags']=data['视频标签(article_topics)'].fillna('')
f_read=open(r'D:\Documents\DS105\NEW\amazon-fine-food-reviews\Reviews_short.csv',encoding = 'utf-8')
refdata=pd.read_csv(f_read,nrows=100)
ref=refdata['Summary']
rv=data['Review']
os=data['Original_summary']
ps=data['Predicted_summary']
s1=[]
s2=[]
s3=[]
for i in range(10):
    s1.append(bleu(rv[i],ref[i]))
    s2.append(bleu(os[i],ref[i]))
    s3.append(bleu(ps[i],ref[i]))

c={"s1" : s1,
   "s2" : s2,
   "s3" : s3}#将列表a，b转换成字典
bleu_scores=pd.DataFrame(c)#将字典转换成为数据框
bleu_scores.to_csv(r'recommend_sheet.csv',mode='w+',index=False,header=True,encoding= 'utf-8')
'''
print(data)
print(s1)
print(s2)
print(s3)
'''


