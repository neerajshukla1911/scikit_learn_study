import numpy as np
from sklearn import decomposition, cross_validation
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

i_cols = []
t_col = ''
min_balance = 0
max_balance = 0
min_duration = 0
max_duration = 0


def preprocess(df, type='train'):
    global i_cols
    global t_col
    global min_balance
    global max_balance
    global min_duration
    global max_duration
    y_mapping = {'yes': 1, 'no': 0}
    default_mapping = {'yes': 1, 'no': 0}
    housing_mapping = {'yes': 1, 'no': 0}
    loan_mapping = {'yes': 1, 'no': 0}
    df['age_group'] = pd.cut(df['age'], [0, 20, 40, 60, 100], labels=['age1', 'age2', 'age3', 'age4'])
    df['age_group'] = df['age_group'].fillna(value='age4')
    # df['duration_group'] = pd.cut(df['duration'], [0, 100, 200, 300, 400, 500, 640, 1100, 1350, 3000,5000], labels=['duration1', 'duration2', 'duration3', 'duration4', 'duration5', 'duration6', 'duration7', 'duration8', 'duration9', 'duration10'])
    # df['duration_group'] = df['duration_group'].fillna(value='duration10')

    if type == 'train':
        min_balance = df.balance.min()
        max_balance = df.balance.max()

    max = max_balance
    min = min_balance

    if min < 0:
        balanceRange = max - min + 9999
    else:
        balanceRange = max + min + 9999
    bins = []
    group_names = []
    rangeStartPoint = min - 1
    count = 0
    while count < balanceRange/10000 + 1:
        bins.append(rangeStartPoint)
        rangeStartPoint += 10000
        count += 1
        if count < balanceRange/10000 + 1:
            group_names.append(count)
    df['balance_group'] = pd.cut(df['balance'], bins, labels=group_names)
    df['y'] = df['y'].map(y_mapping)
    df['default'] = df['default'].map(default_mapping)
    df['housing'] = df['housing'].map(housing_mapping)
    df['loan'] = df['loan'].map(loan_mapping)
    # df['age_group'] = pd.cut(df['age'], [0, 20, 40, 60, 80, 100], labels=['age1', 'age2', 'age3', 'age4', 'age5'])

    if type == 'train':
        min_duration = df.duration.min()
        max_duration = df.duration.max()

    duration_range = max_duration - min_duration
    duration_split = 100
    duration_bins = []
    duration_feature_count = duration_range/duration_split
    loopCounter = 0
    duration_group_name = []
    duration_start_point = 0
    while loopCounter < duration_feature_count:
        duration_bins.append(duration_start_point)
        duration_start_point += duration_split
        loopCounter += 1
        if loopCounter < duration_feature_count:
            duration_group_name.append('duration%s' % loopCounter)

    df['duration_group'] = pd.cut(df['duration'], duration_bins, labels=duration_group_name)
    df['duration_group'] = df['duration_group'].fillna(value=duration_group_name.pop())
    df = pd.get_dummies(df)
    df = df.drop(['id'], axis=1)
    t_col = 'y'
    columns = list(df.columns.values)
    i_cols = columns
    i_cols.remove('y')
    i_cols.remove('balance')
    i_cols.remove('age')
    i_cols.remove('duration')
    i_cols.remove('previous')
    i_cols.remove('pdays')

    return df

if __name__ == '__main__':

    train_df = pd.read_csv("data/bank_fd_train.csv")
    # test_df = pd.read_csv("data/bank_fd_test_X.csv")

    train_df = preprocess(train_df, type='train')
    # test_id = test_df['id']
    # test_df = preprocess(test_df, type='test')

    train_x, train_y = train_df[i_cols], train_df[t_col]
    # test_x, test_y = test_df[i_cols], test_df

    gaussian = GaussianNB()
    # gaussian.fit(train_x, train_y)
    # pred_y = gaussian.predict(test_x)

    scores = cross_validation.cross_val_score(gaussian, train_x, train_y, cv=5)
    print scores
