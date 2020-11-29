import pandas as pd
import re
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from konlpy.tag import Hannanum

hannanum = Hannanum()

df = pd.read_excel('information_theory_news_data.xlsx')

def preprocess(a):
    target = a['body']
    
    # remove email address
    _email_list = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", target)
    if _email_list:
        target = target.split(_email_list[0])[0]
        
    # diversify 0.2 vs ~(sent). (sent)~ by replacing '.' to '. '
    result = ''
    for idx in range(len(target)):
        if target[idx] == '.':
            if not target[idx-1].isdigit() or not target[idx+1].isdigit():
                result += '. '
            else:
                result += '.'
        else:
            result += target[idx]
            
    target = result

    # remove pattern I - " XXX 기자 =  금융당국.... "
    target = re.sub(r'[가-힣]+\ 기자\ =', '', target)
    # remove stuffs in parenthesis
    target = re.sub(r'\[([^\]]+)\]', '', target) # []
    target = re.sub(r'\(([^\)]+)\)', '', target) # ()
    # remove all special characters
    target = re.sub(r'[^a-zA-Z0-9가-힣一-龥\ .]', ' ', target).strip()
    
    return target

df['preprocessed'] = df.apply(preprocess, axis=1)

# 본문 내용이 없는 기사
df = df.drop(df[ df['preprocessed'].str.contains('본문 내용이 없는') ].index )
# 내용이 이상한 내용
df = df.drop(df[ df['preprocessed'] == '' ].index)


def check_korean(x):
    return re.sub(r'[^가-힣]', '', x).strip() == x

def extract_key_sentencce(a):
    _preprocessed_sent = []
    for sent in a['preprocessed'].split('. '):
        _preprocessed_sent.append( ' '.join( [hannanum.nouns(x)[0] if check_korean(x) and hannanum.nouns(x) else x for x in sent.strip().split()] ) )
        
    __tmp = _preprocessed_sent
    _title =  re.sub(r'[^a-zA-Z0-9가-힣一-龥\ ]', ' ', a['title']).strip()
    corpus = [
        ' '.join( [hannanum.nouns(x)[0] if check_korean(x) and hannanum.nouns(x) else x for x in _title.split()] ) 
    ] + __tmp
    tfidfv = TfidfVectorizer().fit(corpus)
    cosine_sim = linear_kernel(tfidfv.transform(corpus).toarray(), tfidfv.transform(corpus).toarray())
    return a['preprocessed'].split('. ')[ cosine_sim[0][1:].argmax() ].strip()


# 결과 파일 저장
df['key_sentence'] = df.apply(extract_key_sentencce, axis=1)
df.to_excel('result.xlsx', index=None)

_tmp = df[['title', 'key_sentence']]
_tmp.to_excel('title_key_sent.xlsx', index=None)


df_1 = pd.read_excel('title_key_sent.xlsx')
corpus = []
for doc in df_1.itertuples():
    title = doc[-2]
    key = doc[-1]
    
    pre_title =  re.sub(r'[^a-zA-Z0-9가-힣一-龥\ ]', ' ', title).strip()
    pre_key = re.sub(r'[^a-zA-Z0-9가-힣一-龥\ ]', ' ', key).strip()
    print( pre_title + ' ' + pre_key)
    print( '*' * 20 )
    corpus.append( pre_title + ' ' + pre_key)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.shape)

data = X.toarray()

num_of_clusters = 50

kmeans = KMeans(n_clusters=num_of_clusters)
kmeans.fit(data)

for c_idx in range(50):
    cat_1 = kmeans.cluster_centers_[c_idx].copy()

    i = 0
    while True:
        cat_1[ cat_1 < (cat_1.min()+i*0.0001) ] = 0
        if len(cat_1) - np.count_nonzero(cat_1==0) <= 20:
            break
        i += 1
    
    print( "Top 20 words of cluster {} : ".format(c_idx))
    print( vectorizer.inverse_transform( cat_1 ) )
    
    labels = kmeans.labels_
    
    c = Counter()
    for idx in np.argwhere(labels==c_idx):
        c += Counter(corpus[idx[0]].split())
    
    sorted( c )
    print( "Top 20 frequent word in cluster {} data".format(c_idx))
    print( list(c)[:20])
    
    print( "10 article of given cluster : " )
    art_count = 0
    for idx in np.argwhere(labels==c_idx):
        print(df.iloc[idx[0]][0])
#         print('*' * 30)
        if art_count == 9:
            break
        art_count += 1
    print('**********************************************************************************************')