import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
# for scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

# for feature selection verification and evaluation 
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df if col not in ['transactiondate']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
    
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
    
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

def get_rmse(y_train):
    """
    Finds RMSE and comparies mean against median and chooses which ever is lower
    """

    
    y_train['baseline_mean'] = y_train['tax_value'].mean()
    y_train['baseline_median'] = y_train['tax_value'].median()

    # scores:
    rmse_mean = mean_squared_error(y_train.tax_value,
                               y_train['baseline_mean'], squared=False)
    rmse_med = mean_squared_error(y_train.tax_value,
                               y_train['baseline_median'], squared=False)

    print("RMSE Mean:")
    print(rmse_mean)
    print("----------------")
    print("RMSE Median:")
    print(rmse_med)