# import libraries

import sys
import nltk
nltk.download(['punkt','stopwords'])
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,classification_report,recall_score,f1_score
from sklearn.model_selection import GridSearchCV
import pickle
from utils import tokenize

def load_data(database_filepath):
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: everything esle
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('disaster_response',engine)
    X = df.message.head(10)
    Y = df.drop(['id','message','original','genre'],axis=1).head(10)
    return X,Y
    
def build_model():
    """
    pipe line construction
    """
    
    # Creating pipeline
    pipeline = Pipeline([
                        ('vect',CountVectorizer(tokenizer=tokenize)),
                        ('tfidf',TfidfTransformer()),
                        ('clf',MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters = {'tfidf__use_idf':[True, False]
              ,'clf__estimator__n_estimators':[10, 25]
              ,'clf__estimator__min_samples_split':[2, 5, 10]
              }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
    
    return cv
    
def evaluate_model(model,X_test, y_test):
    """
    inputs
        X_test
        y_test
    output:
        scores
    """
    y_pred=model.predict(X_test)
    y_pred_df=pd.DataFrame(y_pred)
    y_pred_df.columns=y_test.columns
    
    ## Creating an evaluation matrix of precision scores and recall scores for each column
    eval_matrix=[]
    for column in y_test.columns:    
        eval_matrix.append(str(precision_score(y_test[column], y_pred_df[column])) +','+ str(recall_score(y_test[column], y_pred_df[column])) +','+ str(f1_score(y_test[column], y_pred_df[column])))
    
    # Converting eval matrix to data frame for ease of readability
    df=pd.DataFrame(eval_matrix)
    eval_df=df[0].str.split(',',expand=True)
    eval_df.columns=['Precision','Recall','F1']
    for col in eval_df.columns:
        eval_df[col]=eval_df[col].astype(float)

    print(eval_df.shape)
    print(eval_df)
    print(eval_df.describe())

def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y = load_data(database_filepath)
        
        # Split data into train and test data set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model,X_test, Y_test)

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