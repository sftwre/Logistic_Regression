import pandas as pd

data_file = './logistic_regression/'
train = data_file + 'logistic_digits_train.txt'
test = data_file + 'logistic_digits_test.txt'

df_train = pd.read_csv(train)
df_test = pd.read_csv(test)

def cleanCols(df:pd.DataFrame) -> pd.DataFrame:
    strip_spaces = lambda x: x.replace(' ', '')
    return df.rename(columns=strip_spaces)

# remove spaces in column names
df_train = cleanCols(df_train)
df_test = cleanCols(df_test)