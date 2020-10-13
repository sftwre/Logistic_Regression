import pandas as pd
from sklearn.preprocessing import StandardScaler

data_file = './logistic_regression/'

digits_train = data_file + 'logistic_digits_train.txt'
digits_test = data_file + 'logistic_digits_test.txt'

news_train = data_file + 'logistic_news_train.txt'
news_test = data_file + 'logistic_news_test.txt'

# load the datasets
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

# remove spaces in column names
df_train_digits = cleanCols(df_train_digits)
df_test_digits = cleanCols(df_test_digits)

df_train_news = cleanCols(df_train_news)
df_test_news = cleanCols(df_test_news)

"""
Feature and label vectors digits
"""
X_digits = df_train_digits.loc[:, :'X_train_65'].to_numpy()
X_digits_test = df_test_digits.loc[:, :'X_test_65'].to_numpy()
y_digits = df_train_digits.loc[:, 'Var2'].to_numpy().reshape(-1, 1)
y_digits_test = df_test_digits.loc[:, 'Var2'].to_numpy().reshape(-1, 1)


"""
Feature and label vectors news
"""
X_news = df_train_news.loc[:, :'X_train_2001'].to_numpy()
X_news_test = df_test_news.loc[:, :'X_test_2001'].to_numpy()
y_news = df_train_news.loc[:, 'Var2'].to_numpy().reshape(-1, 1)
y_news_test = df_test_news.loc[:, 'Var2'].to_numpy().reshape(-1, 1)


"""
Normalize datasets
"""
X_digits = X_digits / 255.
X_digits_test = X_digits_test / 255.

scaler = StandardScaler()
X_news = scaler.fit_transform(X_news)
X_news_test = scaler.fit_transform(X_news_test)
