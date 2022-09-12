import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import mean_squared_error,r2_score, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures

from env import host, user, password
"""
test
"""


def area_plot_chi2(train):
    alpha = 0.05
    null_hyp = 'The Sqft of a home and Tax Value are independent'
    alt_hyp = 'There is a relationship between tax value and Sqft'
    observed = pd.crosstab(train.tax_value, train.area)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between Tax Value and the Sqft')   
    print('P', p)
    print('Chi2', round(chi2, 2))
    
def sqft_fun(train):
    plt.figure(figsize = (12, 8))
    ax = sns.barplot(x='Sqft', y='tax_value', data=train,ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    tax_value_avg = train.tax_value.mean()
    plt.axhline(tax_value_avg, label='Tax Value Average')
    plt.legend()
    plt.show()
    area_plot_chi2(train)
    return



def bed_plot_chi2(train):
    alpha = 0.05
    null_hyp = 'The Number of Bedrooms and Tax Value are independent'
    alt_hyp = 'There is a relationship between tax value and Number of Bedrooms'
    observed = pd.crosstab(train.tax_value, train.bedrooms)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between Tax Value and the Number of Bedrooms')   
    print('P', p)
    print('Chi2', round(chi2, 2))
    
    
    
def bedroom_fun(train):
    plt.figure(figsize = (12, 8))
    ax = sns.barplot(x='bedrooms', y='tax_value', data=train,ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    tax_value_avg = train.tax_value.mean()
    plt.axhline(tax_value_avg, label='Tax Value Average')
    plt.legend()
    plt.show()
    bed_plot_chi2(train)
    return

def bath_plot_chi2(train):
    alpha = 0.05
    null_hyp = 'The Number of Bathrooms and Tax Value are independent'
    alt_hyp = 'There is a relationship between Tax Value and Number of Bathrooms'
    observed = pd.crosstab(train.tax_value, train.bathrooms)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between Tax Value and the Number of Bathrooms')   
    print('P', p)
    print('Chi2', round(chi2, 2))

    
def bathroom_fun(train):
    plt.figure(figsize = (12, 8))
    ax = sns.barplot(x='bathrooms', y='tax_value', data=train,ci=None)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    tax_value_avg = train.tax_value.mean()
    plt.axhline(tax_value_avg, label='Tax Value Average')
    plt.legend()
    plt.show()
    bath_plot_chi2(train)
    return

def evaluate_models(y_train, y_validate, x_train, x_validate, x_test):
    """
    This mess puts runs the models it pulls everthing in to one big mess 
    """
    
    x_train=x_train.drop(columns=['Sqft'])
    x_validate=x_validate.drop(columns=['Sqft'])
    x_test=x_test.drop(columns=['Sqft'])

    lm = LinearRegression(normalize=True)

    
    lm.fit(x_train, y_train.tax_value)


    y_train['tax_value_pred_lm'] = lm.predict(x_train)


    y_validate['tax_value_pred_lm'] = lm.predict(x_validate)


    replace_lm = y_validate['tax_value_pred_lm'].min()
    replace_lm_avg = y_validate['tax_value_pred_lm'].mean()
    y_validate['tax_value_pred_lm'] = y_validate['tax_value_pred_lm'].replace(replace_lm, replace_lm_avg)


    lars = LassoLars(alpha=1.0)

 

    lars.fit(x_train, y_train.tax_value)


    y_train['tax_value_pred_lars'] = lars.predict(x_train)


    y_validate['tax_value_pred_lars'] = lars.predict(x_validate)


    replace_lars = y_validate['tax_value_pred_lars'].min()
    replace_lars_avg = y_validate['tax_value_pred_lars'].mean()
    y_validate['tax_value_pred_lars'] = y_validate['tax_value_pred_lars'].replace(replace_lars, replace_lars_avg)


    pf = PolynomialFeatures(degree=2)


    x_train_degree2 = pf.fit_transform(x_train)


    x_validate_degree2 = pf.transform(x_validate)
    x_test_degree2 = pf.transform(x_test)

    lm2 = LinearRegression(normalize=True)

    
    lm2.fit(x_train_degree2, y_train.tax_value)

    y_train['tax_value_pred_lm2'] = lm2.predict(x_train_degree2)

    
    y_validate['tax_value_pred_lm2'] = lm2.predict(x_validate_degree2)

    replace_lm2 = y_validate['tax_value_pred_lm2'].min()
    replace_lm2_avg = y_validate['tax_value_pred_lm2'].mode()
    y_validate['tax_value_pred_lm2'] = y_validate['tax_value_pred_lm2'].replace(replace_lm2, replace_lm2_avg[0])

    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)
    print("RMSE for Lasso & Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lars), 2))
    print('_______________________________________________')
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lm), 2))
    print('_______________________________________________')
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2:", round(r2_score(y_train.tax_value, y_train.tax_value_pred_lm2), 2))
