# import libraries
import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Read the messsages and categories CSV files, and merges both files into
    one dataframe containing the text and the categories of each message.
    Parameters
    ----------
    message_filepath - path to the “disaster_messages.csv” file
    categories_filepath - path to the “disaster_categories.csv” file
    Returns
    -------
    pd.DataFrame
    """
    
    # Read files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge 2 dataframes to return a single dataframe with  messages and labels
    df = pd.merge(messages,categories,on='id')
    return df

def clean_data(df):
    """
    Perform data cleaning on the disaster response dataset
    Arguments
    ---------
    df - the dataframe returned by load_data
    Returns
    -------
    pd.DataFrame
    """
    
    categories = df.categories.str.split(';',expand=True)

    row = categories[:1]
    category_colnames=[]
    for i in row:
        category_colnames.append(row[i][0][0:-2])
        
    categories.columns = category_colnames

    def find_number(text):
        num = re.findall(r'[0-9]+',text)
        return " ".join(num)
        
    for i in categories:
        # set each value to be the last character of the string
        categories[i] = categories[i]=categories[i].apply(lambda x: find_number(x))
        
        # convert column from string to numeric
        categories[i] = categories[i].astype('int')
    
    # drop original column and add the column split into multiple categories
    df=df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicate rows   
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save a dataframe as a sqlite database.
    Arguments
    ---------
    df - a pandas dataframe
    database_filename - the database file name
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()