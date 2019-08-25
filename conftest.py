# %%
import numpy as np
import pandas as pd
from pytest import fixture
import xlrd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from ml_studio.operations.callbacks import History
from ml_studio.supervised_learning.estimator import GradientDescent
from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.supervised_learning.regression import LassoRegression
from ml_studio.supervised_learning.regression import RidgeRegression
from ml_studio.supervised_learning.regression import ElasticNetRegression
from ml_studio.supervised_learning.regression import SGDRegression
from ml_studio.supervised_learning.regression import SGDLassoRegression
from ml_studio.supervised_learning.regression import SGDRidgeRegression
from ml_studio.supervised_learning.regression import SGDRidgeRegression
from ml_studio.supervised_learning.regression import SGDElasticNetRegression

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    y = np.log(y)
    return X, y

@fixture(scope="session")
def get_regression_data_w_validation(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test

@fixture(scope='module')
def train_linear_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = LinearRegression(epochs=2000, val_size=0, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test

@fixture(scope='module')
def train_lasso_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = LassoRegression(epochs=2000, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test    

@fixture(scope='module')
def train_ridge_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = RidgeRegression(epochs=2000, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test   

@fixture(scope='module')
def train_elasticnet_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = ElasticNetRegression(alpha = 0.01, ratio=0.5,
                              epochs=2000, val_size=0, 
                              seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test  



@fixture(scope='module')
def train_sgd_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDRegression(batch_size=1, epochs=200, val_size=0, 
                              seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test  

@fixture(scope='module')
def train_sgd_lasso_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDLassoRegression(batch_size=1, epochs=200, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test    

@fixture(scope='module')
def train_sgd_ridge_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDRidgeRegression(batch_size=1, epochs=200, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test   

@fixture(scope='module')
def train_sgd_elasticnet_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDElasticNetRegression(batch_size=1,
                              alpha = 0.01, ratio=0.5,
                              epochs=200, val_size=0, 
                              seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test  



@fixture(scope='module')
def train_mbgd_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDRegression(batch_size=32, epochs=200, val_size=0, 
                              seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test  

@fixture(scope='module')
def train_mbgd_lasso_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDLassoRegression(batch_size=32, epochs=200, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test    

@fixture(scope='module')
def train_mbgd_ridge_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDRidgeRegression(batch_size=32, epochs=200, val_size=0, 
                          alpha = 0.01, seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test   

@fixture(scope='module')
def train_mbgd_elasticnet_regression(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    gd = SGDElasticNetRegression(batch_size=32,
                              alpha = 0.01, ratio=0.5,
                              epochs=200, val_size=0, 
                              seed=50)
    gd.fit(X_train, y_train)
    return gd, X_train, X_test, y_train, y_test  

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'mean_absolute_error',
                                  'mean_squared_error',
                                  'neg_mean_squared_error',
                                  'root_mean_squared_error',
                                  'neg_root_mean_squared_error',
                                  'median_absolute_error'])
def regression_metric(request):
    return request.param

@fixture(scope='session', params=np.linspace(0.005,.01,5))
def get_learning_rate(request):
    return request.param

def least_squares(X,y):    
    X = np.insert(X, 0, 1, axis=1)
    w = np.linalg.lstsq(X,y)[0]
    return w

@fixture(scope='session')
def analytical_solution(get_regression_data):
    X, y = get_regression_data
    w = least_squares(X,y)    
    return w

@fixture(scope='session')
def analytical_solution_training_data(get_regression_data_w_validation):
    X_train, _, y_train, _ = get_regression_data_w_validation
    train_solution = least_squares(X_train, y_train)
    return train_solution


@fixture(scope='session')
def predict_y():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    gd = GradientDescent(epochs=5)
    gd.fit(X, y)
    y_pred = gd.predict(X)
    return y, y_pred

@fixture(scope='session')
def get_figure_path():
    return "tests/test_figures"

@fixture(scope='session')
def get_alpha():
    return 0.5


@fixture(scope='session')
def get_ratio():
    return 0.7


@fixture(scope='session')
def get_weights():
    return np.array([0.862485488, 0.113651277, 0.104906717, 0.213618469, 0.388680613,
                     0.98925317, 0.447474586, 0.935217135, 0.732583525, 0.076222466])


@fixture(scope='session')
def get_l1_cost():
    return 2.432046723


@fixture(scope='session')
def get_l1_grad():
    return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


@fixture(scope='session')
def get_l2_cost():
    return 1.780241816


@fixture(scope='session')
def get_l2_grad():
    return np.array([0.862485487970269,	0.113651277057817,	0.10490671672632,
                     0.213618469051441,	0.388680612571046,	0.989253169861413,
                     0.447474586387139,	0.935217134881623,	0.732583525054404,
                     0.0762224663412077])


@fixture(scope='session')
def get_elasticnet_cost():
    return 1.969468978


@fixture(scope='session')
def get_elasticnet_grad():
    return np.array([0.47937282319554,	0.367047691558673,	0.365736007508948,
                     0.382042770357716,	0.408302091885657,	0.498387975479212,
                     0.417121187958071,	0.490282570232243,	0.459887528758161,
                     0.361433369951181])


@fixture(scope='class')
def get_quadratic_y():
    return np.array([2257, 4744, 7040, 5488, 9755, 7435, 3812, 5296, 7300, 7041])


@fixture(scope='class')
def get_quadratic_y_pred():
    return np.array([8306, 6811, 1125, 4265, 1618, 3128, 2614, 2767, 3941, 4499])


@fixture(scope='class')
def get_quadratic_X():
    filename = "tests/test_operations/test_quadratic_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[2, 3, 4], skipfooter=2)
    X = df.values
    return X


@fixture(scope='class')
def get_quadratic_cost():
    return 9384127.6


@fixture(scope='class')
def get_quadratic_gradient():
    return np.array([-2109.4, -113092063.3, -34441317.9])


@fixture(scope='class')
def get_binary_cost_X():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[2, 3, 4])
    X = df.values
    return X


@fixture(scope='class')
def get_binary_cost_y():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0])
    y = df.values
    return y


@fixture(scope='class')
def get_binary_cost_y_pred():
    filename = "tests/test_operations/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[1])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_binary_cost():
    return 0.345424815


@fixture(scope='class')
def get_binary_cost_gradient():
    return np.array([-1.556243917, -960.1098781, -1342.758965])


@fixture(scope='class')
def get_categorical_cost_X():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[12, 13, 14])
    X = df.values
    return X


@fixture(scope='class')
def get_categorical_cost_y():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0, 1, 2])
    y = df.values
    return y


@fixture(scope='class')
def get_categorical_cost_y_pred():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[3, 4, 5])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_categorical_cost():
    return 0.367654163


@fixture(scope='class')
def get_categorical_cost_gradient():
    filename = "tests/test_operations/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[26, 27, 28], skipfooter=7)
    y_grad = df.values
    return y_grad


@fixture(scope='function')
def get_history():
    return History()



#%%
