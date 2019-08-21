# --------------------------------------------------------------------------- #
#                               TEST UTILITIES                                #
# --------------------------------------------------------------------------- #
#%%
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from ml_studio.supervised_learning.regression import LinearRegression
from ml_studio.utils.filemanager import save_csv

import warnings
warnings.filterwarnings('ignore')
def process_ames_data():
    train_file = "./tests/test_data/interim/train.csv"
    test_file = "./tests/test_data/interim/test.csv"
    variables = ['LotArea', 'OverallQual', 'YearBuilt', 'ExterQual', 'BsmtQual',
                'TotalBsmtSF', 'GrLivArea', 'FullBath', 'KitchenQual',
                'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']    
    features = variables[:-1]
    target = variables[-1]

    # Read the data
    train = pd.read_csv(train_file, usecols=variables)    
    test  = pd.read_csv(test_file, usecols=features)    
        
    # Get numeric feature names
    numeric_variables = train.select_dtypes(exclude=['object']).columns.values
    numeric_features = numeric_variables[:-1]

    # Standardize numeric columns
    scaler = StandardScaler()    
    scaler.fit(train[numeric_features])
    train[numeric_features] = scaler.transform(train[numeric_features])
    test[numeric_features] = scaler.transform(test[numeric_features])

    # Log transform target
    train[target] = np.log1p(train[target])

    # Onehot Encode Categorical Variables for Training Set    
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    # Separate features from target
    X_train = train.drop(columns=['SalePrice'])
    y_train = train[target]
    X_test = test
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=50)

    # Format y_ values into dataframes
    y_train = pd.DataFrame(data=y_train, columns=['SalePrice'])
    y_val = pd.DataFrame(data=y_val, columns=['SalePrice'])

    # Save the data
    directory = "./tests/test_data/processed/"
    objects = [X_train, y_train, X_val, y_val, X_test]
    filenames = ["X_train.csv", 'y_train.csv', "X_val.csv", "y_val.csv", "X_test.csv"]
    for f, o in zip(filenames, objects):
        save_csv(df = o, directory=directory, filename=f)    

    # Create data tuples
    train = {'X': X_train, 'y': y_train}
    validation = {'X': X_val, 'y': y_val}
    test = {'X': X_test}

    return train, validation, test    

#%%