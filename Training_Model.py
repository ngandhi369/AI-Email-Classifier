import numpy as np
import pandas as pd
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

emails = pd.read_csv('preprocessed1.csv')
em = emails.dropna(axis=0)
em.sample(3)

em['Category'].value_counts()

def pre_process_text(textArray):
    #If using stemming...
    #stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()
    processed_text = []
    for text in textArray:
        words_list = (str(text).lower()).split()
        final_words = [wnl.lemmatize(word) for word in words_list if word not in stopwords.words('english')]
        #If using stemming...
        #final_words = [stemmer.stem(word) for word in words_list if word not in stopwords.words('english')]
        final_words_str = str((" ".join(final_words)))
        processed_text.append(final_words_str)
    return processed_text

em['Subject'] = pre_process_text(em['Subject'])

categories = [ 'logistics','tw-commercial group','california','bill williams iii','deal discrepancies','management','calender','esvl','tufco','resumes','e-mail bin','ces','online trading','junk','junk file','ooc','genco','projects','corporate','archives']

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
]);

# Every additional parameter value here will increase the training time by orders of magnitude.
# I'm running on a relatively slow computer, hence reduced the values

parameters = {
    'vect__max_df': (0.5, 1.0),#0.6, 0.7, 0.8, 0.9, 1.0),
    'vect__max_features': (None, 1000, 5000),#2000, 3000, 4000, 5000, 6000, 10000, 20000, 30000, 40000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),#, (1, 3)),  # unigrams or bigrams or trigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.1, 0.01, 0.001),#, 0.0001, 0.00001, 0.000001, 0.0000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (10, 50),#, 100, 200, 300, 400, 500, 100),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, refit=True)

print("Grid Search started\n---------------------------------------")
print("Pipeline:", [name for name, _ in pipeline.steps])
print("Grid Search Parameters:")
print(parameters)
t0 = time()
grid_search.fit(np.array(em['Subject']), np.array(em['Category']))
print("done in %0.3fs\n----------------------------------------------" % (time() - t0))

print("Best Score: %0.3f\n-------------------------------------------" % grid_search.best_score_)
print("Best Parameters:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

param_grid = {
                'sgdclassifier__learning_rate':['constant','optimal','invscaling'],
                'sgdclassifier__eta0':[0.0,0.01,0.1,0.3,0.5,0.7],
                'sgdclassifier__alpha':[0.0001,0.001,0.01,0.1]
}

pipeline.get_params().keys()

import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(grid_search, open(filename, 'wb'))

test_set = [
    'hey there',
    'california',
    'movie tickets for sale',
    'Advice needed for treatment of hair fall',
    'Moving out sale',
    'RE: Selling Honda City',
    'want to grab some offers'
]

loaded_model.best_estimator_.predict(np.array(test_set))



