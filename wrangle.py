import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

from env import host, user, password





def get_db_url(db_name):
    """
    This function uses env file to get the url to access the SQL database.
    It takes in a string identifying the database I want to connect to.
    """
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"




def get_data_from_sql(str_db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from SQL and return a DataFrame.
    """
    df = pd.read_sql(query, get_db_url('zillow'))
    return df

def get_zillow_mvp():
    """
    This Function pulls from SQL for Project two using the listed databases and col
    """
    query = """
select 
bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt
FROM properties_2017
join propertylandusetype using (propertylandusetypeid)
join predictions_2017 using (parcelid)
where propertylandusedesc = "Single Family Residential"
;
    """
    df = get_data_from_sql("zillow", query)
    return df


def optimize_types(df):
    """ 
    Convert columns to integers
     and bathroom and bedrooms to integers
    """
    df["bathroomcnt"] = df["bathroomcnt"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    return df

def handle_outliers(df, col_list, k = 1.5):
    """
    Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors
    """
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range

        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df
def handle_nulls(df):    
    df = df.dropna()
    return df

def filter_square_feet(x):
    if x > 0 and x < 1000:
        return 'Less Than 1000 Sqft'
    if x > 1001 and x < 2000:
        return 'More Than 1000 Sqft'
    if x > 2001:
        return 'More Than 2000 Sqft'
    
def wrangle_zillow(df):
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    changes name to be easier to handle
    returns a clean dataframe
    """
    
    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df, ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt'])
    
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'area',
                          'taxvaluedollarcnt':'tax_value'})
    df['Sqft'] = df['area'].apply(filter_square_feet)
   
    return df

def train_validate_test(df, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)

    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    x_train = train.drop(columns=[target])
    y_train = train[target]

    x_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    x_test = test.drop(columns=[target])
    y_test = test[target]

    return train, validate, test, x_train, y_train, x_validate, y_validate, x_test, y_test

def scale_data(train, validate, test, cols_scale):
    '''
    This function takes in train, validate, and test dataframes as well as a
    list of features to be scaled by MinMaxScalar. It then returns the 
    scaled versions of train, validate, and test in new dataframes. 
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[cols_scale])
    
    train_scaled[cols_scale] = pd.DataFrame(scaler.transform(train[cols_scale]),
                                               columns = train[cols_scale].columns.values).set_index([train.index.values])
    validate_scaled[cols_scale] = pd.DataFrame(scaler.transform(validate[cols_scale]),
                                               columns = validate[cols_scale].columns.values).set_index([validate.index.values])
    test_scaled[cols_scale] = pd.DataFrame(scaler.transform(test[cols_scale]),
                                               columns = test[cols_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled