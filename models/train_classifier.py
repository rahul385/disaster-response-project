# import libraries

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,classification_report,f1_score,confusion_matrix,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: everything esle
        category names.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from disaster',engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
    
    # listing the columns
    category_names = list(np.array(y.columns))
    
    return X,y,category_names
    
def tokenize(text):
    """
    Normalize and tokenize
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize (text)
    stemmed = [PorterStemmer().stem(w) for w in words]
    return stemmed

def build_model():
    """
    pipe line construction
    """
    
    # Creating pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),
                     ('tfidf',TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'vect__min_df': [1],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 25], 
              'clf__estimator__min_samples_split':[2, 5, 10]
              }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
    
    return cv

def eval_metrics(ArrayL, ArrayP, col_names):

    """Evalute metrics of the ML pipeline model
    
    inputs:
    ArrayL: array. Array containing the real labels.
    ArrayP: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the ArrayP fields.
       
    Returns:
    data_metrics: Contains accuracy, precision, recall 
    and f1 score for a given set of ArrayL and ArrayP labels.
    """
    metrics = []
    
    # Evaluate metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(ArrayL[:, i], ArrayP[:, i])
        precision = precision_score(ArrayL[:, i], ArrayP[:, i],average='macro')
        recall = recall_score(ArrayL[:, i], ArrayP[:, i],average='macro')
        f1 = f1_score(ArrayL[:, i], ArrayP[:, i],average='macro')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # store metrics
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return data_metrics
    
def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)
    col_names = list(y_test.columns.values)
    print(eval_metrics(np.array(y_test), y_pred, col_names))


def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()