"""
Description: Loan Default predictor based on a Random Forest Model

Requirements: Scientific python stack (numpy, pandas, scikit-learn)

Usage: Two arguments are expected: training dataset and test dataset

            python solution.py puzzle_train_dataset.csv puzzle_test_dataset.csv

       An output file in .cvs format is generated
"""

import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

def get_data(file_name):
    df = pd.read_csv(file_name, parse_dates=['last_payment','end_last_loan'])
    return df

def feat_engineering(df, income_threshold, features):
    # Remove outliers
    df = df.query('income < %f' % income_threshold)
    # Handle missing values
    min_last_payment, min_end_last_loan  = df.last_payment.min(), df.end_last_loan.min()
    df = df.fillna( {'last_payment':min_last_payment, 'end_last_loan':min_end_last_loan} )
    # Create new features
    df['delta_pay']  = ( df.last_payment.max()  - df.last_payment  ).astype('timedelta64[D]')
    df['delta_loan'] = ( df.end_last_loan.max() - df.end_last_loan ).astype('timedelta64[D]')
    # Return good features
    return df.loc[:, features]

def feat_tidying(X):
    return X.drop('default', axis=1), X['default'].astype('bool')

def model_prediction(df, income_threshold, model, features):
    X = feat_engineering(df, income_threshold, ['ids'] + features)
    X['default'] = model.predict( X.iloc[:,1:] )
    return X.loc[:,['ids','default']]

def handle_outliers(df, income_threshold):
    outliers = df.loc[ df.income>income_threshold , ['ids'] ]
    outliers['default'] = np.zeros(outliers.shape[0], dtype=bool)
    return outliers

def handle_miss_income(df):
    miss_income = df.loc[ df.income.isnull() , ['ids'] ]
    miss_income['default'] = np.zeros(miss_income.shape[0], dtype=bool)
    return miss_income

def write_csv(prediction, outliers, miss_income, out_file):
    final_prediction = pd.concat([prediction, outliers, miss_income])
    final_prediction.to_csv(path_or_buf=out_file, index=False)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        sys.exit("\n Wrong arguments \n")

    train_data, test_data = sys.argv[1], sys.argv[2]

    df = get_data(train_data)
    df.dropna(axis=0, how='any', subset=['default'], inplace=True)

    income_threshold = 300000
    features = ['default',
                'amount_borrowed',
                'borrowed_in_months',
                'income',
                'delta_pay',
                'delta_loan']

    X = feat_engineering(df, income_threshold, features)
    X, y = feat_tidying(X)

    model = RandomForestClassifier(max_depth=10, random_state=1);
    model.fit(X, y);
    
    df = get_data(test_data)

    prediction = model_prediction(df, income_threshold, model, features[1:])
    outliers = handle_outliers(df, income_threshold)
    miss_income = handle_miss_income(df)

    write_csv(prediction, outliers, miss_income, 'predictions.csv')
