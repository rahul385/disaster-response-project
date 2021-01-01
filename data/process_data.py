# import libraries
import sys
import pandas as pd
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

    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    category_colnames=[]
    for cat in df.categories[0].split(';'):
        category_colnames.append(str(cat)[:-2])
    
    # rename the columns of categories df
    categories.columns = category_colnames
    
    for col in categories.columns:
        for row in range(df.shape[0]):
            categories[col][row]=categories[col][row][-1:]
    
    # Convert the column data typs from string to a int
    for col in categories.columns:
        categories[col]=categories[col].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)

    # Join df and categories data frames.
    df_final=df.join(categories)
    
    # Remove records where related=2
    df_final=df_final[df_final['related']!=2]
    
    # drop duplicates
    df_final.drop_duplicates(inplace=True)
    
    return df_final

def save_data(df, database_filename):
    """
    Save a dataframe as a sqlite database.
    Arguments
    ---------
    df - a pandas dataframe
    database_filename - the database file name
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, index=False,if_exists='replace')

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