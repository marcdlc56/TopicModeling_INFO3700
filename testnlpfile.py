data = []

with open("output1.txt", "r") as myfile:
    for line in myfile:
        line = line.replace('\n','')
        if line == '':
            continue
        else:
            data.append(str(line))
myfile.close()

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))

import string
exclude = set(string.punctuation)

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(wordnet_lemmatizer.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(line).split() for line in data]


import gensim
from gensim import corpora

dictionary = corpora.Dictionary(doc_clean)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(doc_term_matrix, num_topics=20, id2word=dictionary, passes=50)
for idx, topic in ldamodel.print_topics(-1):
    print(f'Topic: {idx} Word: {topic}')

example = doc_clean[100]

example_lda = dictionary.doc2bow(example)

for index, score in sorted(ldamodel[example_lda], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 5)))

# Visualize the lda model with interactive plots
import pyLDAvis
import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)

print(pyLDAvis.display(lda_display))
