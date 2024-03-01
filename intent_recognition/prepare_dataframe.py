import pandas as pd
import random

def preproc(string):
    string = str(string)
    if random.uniform(0, 1) > 0.51:
        string = string+'?'
    return string
def read_txt(path):
    with open(path, encoding='utf-8') as t:
        text = t.read()
        return text
def normalize(text):
    text_list = text.split('\n')
    text_list = [text.strip('"') for text in text_list]
    text_list = [text.lower() if random.uniform(0, 1)<0.5 else text for text in text_list]
    text_list = [text.upper() if random.uniform(0, 1)<0.1 else text for text in text_list]
    text_list = [text.strip('?') if random.uniform(0, 1)<0.4 else text for text in text_list]
    return text_list

mapping = {'chroma_request': 0, 'paraphrase etc': 1, 'nonsense': 2}
df_chroma = pd.read_excel('data/Q_A.xlsx').Theme
df_chroma = df_chroma.apply(preproc)
chroma_class = pd.Series([mapping['chroma_request'] for _ in range(len(df_chroma))])
Dataset_chroma = pd.DataFrame({'query': df_chroma, 'cls': chroma_class})

nonsense = normalize(read_txt('data/unrelevant.txt'))
nonsense_cls = [mapping['nonsense'] for _ in range(len(nonsense))]
dosp = normalize(read_txt('data/доспрашивание.txt'))
dosp_cls = [mapping['paraphrase etc'] for _ in range(len(dosp))]
nonsense.extend(dosp)
nonsense_cls.extend(dosp_cls)

dataset = pd.DataFrame({'query': nonsense, 'cls': nonsense_cls})
dataset = dataset[:1673] #куча лишних строк попало

dataset = pd.concat([dataset, Dataset_chroma], axis=0)
dataset.to_csv('intent.csv', index = 0)

    