import spacy
import pandas as pd
nlp = spacy.load("en")

data = pd.read_csv('output1.txt')

doc = nlp(data)

