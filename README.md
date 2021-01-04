# Disaster Response Project
https://webapp-flask-disaster-response.herokuapp.com/

## Installation
The following python libraries are required to run this project:

pandas, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy.

## Project Overview
The objective of this Machine Learning project is to analyze thousands of real messages sent during natural disasters either via social media or directly to disaster response organizations. 

Machine learning is critical to helping different disaster response organizations understand which messages are relevant to them and which messages to prioritize. During these disasters is when they have the least capacity to filter out messages that matter, and find basic methods such as using key word searches to provide trivial results. This disaster response pipeline will classify those text messages into several categories which can easily be monitored by respective disaster management organizations.

## File Description:

* **data**: This folder contains real messages received during a natural disaster in csv format.
    * `disaster_categories.csv` : Contains name of the package and author's information
    * `disaster_messages.csv` : Contains name of the package and author's information
    * `DisasterResponse.db` : Contains name of the package and author's information
    * `process_data.py` : This python module takes csv files as input containing messages and categories (labels), cleans and processes the data and then exports it into a database table.
    
* **models**: This folder contains real messages received during a natural disaster in csv format.
    * `classifier.pkl` : Contains name of the package and author's information
    * `train_classifier.py` : This file imports the data from database table and trains a Machine Learnin model to classify messages among 36 different categories.
    * `utils.py` : Contains name of the package and author's information

* **templates**: This folder contains real messages received during a natural disaster in csv format.
    * `go.html` : Contains name of the package and author's information
    * `master.html` : This file imports the data from database table and trains a Machine Learnin model to classify messages among 36 different categories.

* **Visualizations**: This folder contains real messages received during a natural disaster in csv format.
    * `Message_Count_By_Category.PNG` : Contains name of the package and author's information
    * `Message_Count_By_Genre.PNG` : This file imports the data from database table and trains a Machine Learnin model to classify messages among 36 different categories.

* **ETL Pipeline Preparation.ipynb**:  Jupyter notebook for process_data.py

* **ML Pipeline Preparation.ipynb**: Jupyter notebook for train_classifier.py

* **run.py**: This folder cointains the run.py to iniate the web app.

* **utils.py**: This folder cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. The web application deployed on Heroku can be accessed through the below URL
https://webapp-flask-disaster-response.herokuapp.com/

## Screenshots

***Screenshot 1: Message Count by Category***
![Screenshot 1](https://github.com/rahul385/disaster-response-project/blob/master/visualizations/Message_Count_By_Category.PNG)

***Screenshot 2: Message Count by Genre***
![Screenshot 2](https://github.com/rahul385/disaster-response-project/blob/master/visualizations/Message_Count_By_Genre.PNG)


## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

Author: Rahul Gupta Copyright 2020

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
