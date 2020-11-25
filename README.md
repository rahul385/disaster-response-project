# Disaster Response Pipeline Project

## Installation
The following python libraries are required to run this project. 
pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.

## Project Overview
This code is designed and developed to run a web app which helps in disaster management (e.g. Flood, Earthquake etc.). In the event of a natural disaster, thousands of messages are sent on social media seeking help of various categories. This disaster response pipeline will classify those text messages into several categories which can then be monitored by respective government bodies.

The app built to have an ML model to categorize every message received
## File Description:
* **process_data.py**: This file takes csv files as input containing message data and message categories (labels), and then merges the data and exports it into a database table.
* **train_classifier.py**: This file imports the data from database table and trains the ML model on training set and evaluates accuracy.
* **ETL Pipeline Preparation.ipynb**:  Jupyter notebook for process_data.py
* **ML Pipeline Preparation.ipynb**: Jupyter notebook for train_classifier.py
* **data**: This folder contains real messages received during a natural disaster in csv format.
* **app**: This folder cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots

***Screenshot 1: Message Count by Category***
![Screenshot 1](https://github.com/rahul385/disaster-response-project/blob/master/visualizations/Message_Count_By_Category.PNG)

***Screenshot 2: Message Count by Genre***
![Screenshot 2](https://github.com/rahul385/disaster-response-project/blob/master/visualizations/Message_Count_By_Genre.PNG)


## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
