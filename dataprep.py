import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler


def preprocess_data(df):
    """
    Preprocess the data by converting date columns, creating a date difference feature, and scaling numerical features.
    """
    numerical_features = ['amount', 'date_difference']

    df['entered_date'] = pd.to_datetime(df['entered_date'], format='mixed')
    df['document_date'] = pd.to_datetime(df['document_date'], format='mixed')

    df['date_difference'] = (df['entered_date'] - df['document_date']).dt.days

    # No NaN filling - if NaN exist, let it error so we can fix it
    num_scaled = RobustScaler().fit_transform(df[numerical_features])
    num_scaled = MinMaxScaler().fit_transform(num_scaled)
    df[numerical_features] = num_scaled

    return df


cols_expand = ["amount", 
                "gl_account_name", 
                "cd_flag",
                #"gl_account",
                "promptly","weekend","nwh",'top_n','high_cash'
                ]
cols_constant = ["tax_rate","user","date_difference"
                 #,"promptly","weekend","nwh",'top_n','high_cash'
                 ]

MAX_LEN = 3
def expand_group(group):
    result = {}

    for col in cols_expand:
        values = group[col].tolist()
        values = values + [0] * (MAX_LEN - len(values))  # Padding mit 0
        for i in range(MAX_LEN):
            result[f"{col}_{i+1}"] = values[i]

    for col in cols_constant:
        result[col] = group[col].iloc[0]

    return pd.Series(result)
def expand_cols(df):

    df_out = (
        df
        .groupby("posting_id", as_index=False)
        .apply(expand_group)
        .reset_index(drop=True)
    )
    return df_out

def ohe_for_model(df_out):
    categorical_features = [
        'gl_account_name_1','gl_account_name_2', 'gl_account_name_3', 
        'cd_flag_1', 'cd_flag_2','cd_flag_3', 'tax_rate', 'user', 
        #'gl_account_1', 'gl_account_2', 'gl_account_3',
        #"promptly","weekend","nwh",'top_n','high_cash'
        'weekend_1','weekend_2', 'weekend_3', 'nwh_1', 'nwh_2', 'nwh_3','top_n_1','top_n_2', 'top_n_3', 'high_cash_1', 'high_cash_2', 'high_cash_3'
        ]

    # Alle in str konvertieren
    for col in categorical_features:
        df_out[col] = df_out[col].astype(str)

    # One-Hot
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df_out[categorical_features])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_features),
        index=df_out.index  # damit Reihenfolge passt
    )

    df_final = pd.concat([df_out.drop(columns=categorical_features), encoded_df], axis=1)
    return df_final

def full_dataprep():
    df = pd.read_csv("journal_entries.csv", sep=';')
    df = preprocess_data(df)
    labels = df.groupby('posting_id')['label'].max()
    df = expand_cols(df)
    df= ohe_for_model(df)
    return df, labels

if __name__ == "__main__":
    df, labels = full_dataprep()
    print(df.head())