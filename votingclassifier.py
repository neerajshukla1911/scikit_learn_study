import pandas as pd
import seaborn as sns
from sklearn import cross_validation
from sklearn.metrics import f1_score

sns.set_style('whitegrid', {})
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB


def preprocess_data(df):
    yes_no_mapping = {'yes': 1, 'no': 0}
    df['default'] = df['default'].map(yes_no_mapping)
    df['housing'] = df['housing'].map(yes_no_mapping)
    df['loan'] = df['loan'].map(yes_no_mapping)
    df['age'] = pd.cut(df['age'], bins=12)
    df['balance'] = pd.cut(df['balance'], bins=9)
    df['duration'] = pd.cut(df['duration'], bins=40)
    df['day'] = pd.cut(df['day'], bins=13)
    df['pdays'] = pd.cut(df['pdays'], bins=25)
    df['previous'] = pd.cut(df['previous'], bins=4)
    df = df.drop(['id', 'y'], axis=1)
    df = pd.get_dummies(df)
    return df


if __name__ == '__main__':
    train_df = pd.read_csv("data/bank_fd_train.csv")
    # test_df = pd.read_csv("../data/bank_fd_test_X.csv")
    yes_no_mapping = {'yes': 1, 'no': 0}
    inverse_yes_no_mapping = {1: 'yes', 0: 'no'}

    train_df['y'] = train_df['y'].map(yes_no_mapping)
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(train_df, train_df['y'], test_size=0.20, random_state=45)
    # test_id = test_df['id']
    train_x = preprocess_data(train_x)
    test_x = preprocess_data(test_x)

    vc = VotingClassifier(estimators=[('gau', GaussianNB()), ('lr', LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')),
                                          ('rfc', RandomForestClassifier( n_estimators=40, max_features=0.25, random_state=10))]
                              , voting='soft', weights=[5, 16, 8])

    vc.fit(train_x, train_y)
    pred_y = vc.predict(test_x)

    print f1_score(test_y, pred_y)

