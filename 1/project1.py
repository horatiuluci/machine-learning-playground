import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets, sklearn.model_selection, sklearn.linear_model, sklearn.metrics


def estimate_coef(df_x, df_y):
    # data interception
    df_x.insert(0,'intercept',1)
    # dataframe to numpy array
    X = df_x.to_numpy()

    y = np.array(df_y.iloc[:, -1])
    list_y = []
    for i in range(len(y)):
        list_y.append([y[i]])
    y = np.array(list_y)
    #y.reshape(y.shape[0], 1)
    #y = y.transpose()

    X_t = X.transpose()
    X_t_X = X_t.dot(X)
    X_t_X_inv = np.linalg.inv(X_t_X)
    inter = X_t_X_inv.dot(X_t)
    betas = np.dot(inter,y)
    # number of observations/points

    return betas


def get_lasso_coef(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model.coef_


def lasso_100_compare(X_train, y_train, X_test, y_test):
    alpha_range = np.linspace(0.01, 1.0, num=500)
    # print(alpha_range)
    predictions = []
    for i in alpha_range:
        #print(i)
        model = sklearn.linear_model.Lasso(alpha=i)
        model.fit(X_train, y_train)
        predictions.append((i, model.predict(X_test)))
    r_sq = []
    for pred in predictions:
        r_sq.append((pred[0], sklearn.metrics.r2_score(y_test, pred[1])))
    ualp,r2sc = map(list,zip(*r_sq))
    '''print("Alpha - R2Score")
    for i in range(len(ualp)):
        print("{} - {}".format(ualp[i], r2sc[i]))'''

    alpha_vals = {'alpha': alpha_range}
    lasso_vals = sklearn.model_selection.GridSearchCV(model, alpha_vals, scoring='neg_mean_squared_error', cv=10 )
    print('The best value of alpha is:',lasso_vals.fit(X_train,y_train).best_params_)

    plt.plot(ualp, r2sc, 'r')
    plt.xlabel('Alpha')
    plt.ylabel('R^2 Score')
    plt.show()





def main():
    df = sklearn.datasets.load_diabetes()

    data = np.c_[df.data, df.target]
    columns = np.append(df.feature_names, ["target"])
    df = pd.DataFrame(data, columns=columns)
    # data intercept


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.3, random_state=42)
    betas = estimate_coef(X_train, y_train)
    print('weights as computed using OLS:')
    print(betas)

    y_test_ = np.array(y_test.iloc[:, -1])
    x = []
    for i in range(0, X_test.shape[0]):
        x.append(np.array(X_test.iloc[i, :]))
    #last = b[1:]
    res = []
    last =[]
    for i in range(10):
        last.append(betas[i+1][0])
    for i in range (len(x)):
        res.append(betas[0][0]+ np.sum(x[i]*np.array(last)))

    # print('y_test_:\n', y_test_, '\n\n', 'res:\n', res, '\n\n')
    print("R^2 for OLS:   {}".format(sklearn.metrics.r2_score(y_test_, res)))


    # LASSO
    alp=0.5
    clf = sklearn.linear_model.Lasso(alpha=alp)
    clf.fit(X_train, y_train)
    X_test.insert(0,'intercept',1)
    res_lasso = clf.predict(X_test.to_numpy())
    print("weights as computed using Lasso:")
    lasso_coef = clf.coef_
    print(lasso_coef)

    '''res_lasso = []
    #print(x[0])
    for i in range(len(x)):
        res_lasso.append(clf.intercept_+np.sum(x[i]*lasso_coef))
    #print(res_lasso)'''
    print("R^2 for Lasso: {} (alpha={})".format(sklearn.metrics.r2_score(y_test_, res_lasso), alp))

    lasso_100_compare(X_train, y_train, X_test.to_numpy(), y_test_)

    # print(type(df))
    # df.to_csv('file1.csv', index=True)


if __name__ == '__main__':
    main()



'''
151.00818273, 29.25034582, -261.70768053, 546.29737263, 388.40077257, -901.95338706, 506.761149 , 121.14845948, 288.02932495, 659.27133846, 41.37536901
'''
