# Import libraries
from utils import tokenize
import json
import plotly
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt','stopwords'])
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import pickle

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
with open('models/classifier.pkl','rb') as f:
    model = pickle.load(f)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100*gen_count/gen_count.sum(), 2)
    gen = list(gen_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)

    colors = ['yellow', 'green', 'red']
    
    # create visuals
    
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_per,
                  "y": gen
                },
                "marker": {
                  "colors": [
                    "#7fc97f",
                    "#beaed4",
                    "#fdc086"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": gen,
                "values": gen_count
              }
            ],
            "layout": {
              "title": "Messages Count by Genre"
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "x": cat,
                "y": cat_num,
                "marker": {
                  "color": 'blue'}
                }
            ],
            "layout": {
              "title": "Message Count by Category",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "Category"
              },
              'barmode': 'group'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    pass

if __name__ == '__main__':
    main()