import pickle
import numpy as np
import pandas as pd
import csv
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

'''PARAMETERS'''
CLUSTERS = 5
WORD_NUM = 10
'''PARAMETERS'''

t = Tokenizer()

# データの読み込み
def wakati(csv_name):
    texts = []
    with open(csv_name, 'r', encoding="utf_8_sig") as f:
        reader = csv.reader(f)
        for row in reader:
            texts.append(row)
        # text_list = pd.DataFrame(texts).T.values.tolist()
    res = []
    for text in texts[0]:
        res_sub = []
        lines = text.split("\n")
        for line in lines:
            malist = t.tokenize(line)
            for tok in malist:
                ps = tok.part_of_speech.split(',')[0]
                if not ps in ['名詞', '動詞', '形容詞']: continue
                w = tok.base_form
                if w == '*' or w == '': w = tok.surface
                if w == '' or w == '\n': continue
                res_sub.append(w)
        res_wakati = (' '.join(res_sub))
        res.append(res_wakati)
    return np.array(res)

def vectorize(wakati_array):
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(wakati_array)
    return vecs.toarray(),list(vectorizer.vocabulary_.keys())

def python_list_add(in1, in2):
    wrk = np.array(in1) + np.array(in2)
    return wrk.tolist()

def python_list_divide(in1, in2):
    wrk = np.array(in1) / np.array(in2)
    return wrk.tolist()

def clustering(vecs,word_list,n_clusters):
    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)
    for num in range(n_clusters):
        index_list = [i for i, x in enumerate(clusters) if x == num]
        sum_list = [0 for i in range(len(word_list))]
        for index in index_list:
            sum_list = python_list_add(sum_list, vecs[index])
        total_sum = [sum(sum_list) for i in range(len(word_list))]
        sum_list = python_list_divide(sum_list,total_sum)
        result = [word_list,sum_list]
        result = pd.DataFrame(result).T.values.tolist()
        result = sorted(result, key=lambda x: x[1],reverse=True)
        for j in range(WORD_NUM):
            print('classification:{}, word:{}, percentage:{}%'.format(num,result[j][0],round(result[j][1]*100, 2)))

# 初回にPickle化
wakati_array = wakati('amazon_review.csv')
vecs = vectorize(wakati_array)
with open('amazon_review.pickle', 'wb') as f:
    pickle.dump(vecs, f)

# 非Pickle化
# with open('wakati_array.pickle', 'rb') as f:
    # vecs = pickle.load(f)

clustering(vecs[0],vecs[1],CLUSTERS)


    
