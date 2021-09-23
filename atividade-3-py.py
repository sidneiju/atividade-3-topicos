import bibtexparser
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def open_parse_bibtex_file(path):
    file = open(path, encoding='utf8')
    return bibtexparser.loads(file.read())

data_acm = open_parse_bibtex_file('./datasets/slr/mdwe/round1-acm.bib')
data_ieee = open_parse_bibtex_file('./datasets/slr/mdwe/round1-ieee.bib')
data_sciencedirect = open_parse_bibtex_file('./datasets/slr/mdwe/round1-sciencedirect.bib')

dataset_entries = data_acm.entries + data_ieee.entries + data_sciencedirect.entries

X = [el['abstract'] for el in dataset_entries]
y = [el['inserir'] for el in dataset_entries]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

classifiers_list = [
    {
        'classifier': DecisionTreeClassifier(), 
        'params': {
            'criterion':['gini','entropy'],
            'max_depth': [None, 1, 3, 5, 10]
        }
    },
    
    { 
        'classifier': LinearSVC(),
        'params': {
            'C': range(1, 10)
        }
    },
    
    {
        'classifier': SVC(),
        'params': {
            'kernel': ['linear', 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10]
        }
    },
    
    {
        'classifier': RandomForestClassifier(),
        'params': {
            'criterion':['gini','entropy'],
            'max_depth': [None, 1, 3, 5, 10]
        }
    },
    
    {
        'classifier': LogisticRegression(),
        'params': {
            'solver':  ['newton-cg','lbfgs','liblinear'],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }
    },
    
    {
        'classifier': GaussianNB(),
        'params': {
            'var_smoothing': np.logspace(0,-9, num=100)
        }
    }
]


for item in classifiers_list:
    print(item['classifier'])
    
    grid = GridSearchCV(item['classifier'], item['params'], scoring='accuracy', cv=10)
    grid.fit(X.toarray(), y)
    
    for param in item['params']:
        print(f"{param}: {grid.best_params_[param]}")
        
    print(f"Acurácia: {grid.best_score_}")
    print('')