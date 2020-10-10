import pandas as pd

data_file = './logistic_regression/'

digits_train = data_file + 'logistic_digits_train.txt'
digits_test = data_file + 'logistic_digits_test.txt'

news_train = data_file + 'logistic_news_train.txt'
news_test = data_file + 'logistic_news_test.txt'

df_train_digits = pd.read_csv(digits_train)
df_test_digits = pd.read_csv(digits_test)

df_train_news = pd.read_csv(news_train)
df_test_news = pd.read_csv(news_test)

def cleanCols(df:pd.DataFrame) -> pd.DataFrame:
    """
    Used to remove spaces in column names
    :param df: Dataframe with data
    :return: Dataframe with spaces in columns stripped
    """
    strip_spaces = lambda x: x.replace(' ', '')
    return df.rename(columns=strip_spaces)

# load the digits dataset
df_train_digits = cleanCols(df_train_digits)
df_test_digits = cleanCols(df_test_digits)

# load the news dataset
df_train_news = cleanCols(df_train_news)
df_test_news = cleanCols(df_test_news)