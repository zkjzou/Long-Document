import torch
from ignite.metrics import Rouge
from tqdm import tqdm
import json
import spacy
import numpy as np
import itertools

nlp = spacy.load('en_core_web_sm')


def read_json(path):
    with open(path, 'r') as f:
        json_str = list(f)
    res = []
    for str in json_str:
        res.append(json.loads(str))
    return res
def split_sents(doc):
    doc = nlp(doc)
    res = []
    for sent in doc.sents:
        res.append(sent)
    return res
def mean_rouge(sent, summary):
    #compute the mean ROUGE-{1,2,L}
    rouge = Rouge(variants=[1,2,"L"])
    sent = sent.split()
    summary = [s.split() for s in summary]
    rouge.update(([sent],[summary]))
    scores = rouge.compute()
    return (scores['Rouge-1-F'] + scores['Rouge-2-F'] + scores['Rouge-L-F'])/3
def extract_oracle(paragraphs):
    rouge = Rouge(variants=[1,2,"L"])
    oracles = []
    for i in tqdm(range(len(paragraphs))):
        paragraph = paragraphs[i]
        text = paragraph['text']
        summary = paragraph['summary']
        sents = split_sents(text)
        scores = np.zeros(len(sents))
        for k, sent in enumerate(sents):
            scores[k] = mean_rouge(sent.text, summary)
        #select five highest score
        indices = np.argsort(scores)[-5:]
        highest = []
        for index in indices:
            highest.append(sents[index].text)
        #all combinations of 1, 2, and 3 sentences
        candidates = []
        for k in range(1,4):
            for subset in itertools.combinations(highest,k):
                candidates.append(" ".join([k for k in subset]))
        #final orcale the highest mean ROUGE
        highest_score = -1
        highest = ""
        for candidate in candidates:
            score = mean_rouge(candidate, summary)
            if highest_score<score:
                highest_score = score
                highest = candidate
        rouge.update(([highest.split()],[[s.split() for s in summary]]))
        paragraphs[i]['oracles'] = highest
    print(rouge.compute())
    return oracles
if __name__=='__main__':
    train = read_json("train.jsonl")
    train_orcales = extract_oracle(train)
    with open ('train_oracles.jsonl','w') as f:
        for paragraph in train:
            f.write(json.dumps(paragraph)+"\n")
    val = read_json("val.jsonl")
    val_orcales = extract_oracle(val)
    with open ('val_oracles.jsonl','w') as f:
        for paragraph in val:
            f.write(json.dumps(paragraph)+"\n")
