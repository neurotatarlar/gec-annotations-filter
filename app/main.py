from transformers import pipeline


    
if __name__ == "__main__":
    clf = pipeline("text-classification", model="textdetox/bert-multilingual-toxicity-classifier", device=-1)
    texts = [
        "Син бик матур.",  # should be non-toxic
        "Ты отвратительный идиот!",  # should be toxic
    ]
    for t in texts:
        out = clf(t, truncation=True)
        print(t, "=>", out)