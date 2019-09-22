# import numpy as np
# from sklearn import datasets
# from sklearn import linear_model
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# X, y = datasets.load_boston(return_X_y=True)
# scaler = StandardScaler()    
# X = scaler.fit_transform(X)
# y = np.log(y)
# X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=50)
# sgd = linear_model.SGDRegressor()
# sgd.fit(X_train,y_train)
# y_pred = sgd.predict(X_test)        
# score = sgd.score(X_test, y_test)

