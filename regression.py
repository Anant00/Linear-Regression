from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def getMean(nums: List) -> float:
    total = sum(nums)
    readings = len(nums)
    mean = total / float(readings)
    return mean

'''
sxx 
'''
def getVariance(nums: List, mean) -> float:
    var = 0
    for i in range(0, len(nums)):
        var += pow((nums[i] - mean), 2)
    return var


'''
sxy 
'''
def getCovariance(nums_1, nums_2):
    mean_1 = getMean(nums_1)
    mean_2 = getMean(nums_2)
    size = len(nums_1)
    covariance = 0.0
    for i in range(0, size):
        covariance += (nums_1[i] - mean_1) * (nums_2[i] - mean_2)
    return covariance



def ssxreg(b1, sxx):
    return (b1 ** 2) * sxx


def getSlope(x_list, y_list):
    y_mean = getMean(y_list)
    x_mean = getMean(x_list)
    cov = getCovariance(x_list, y_list)
    var = getVariance(y_list, y_mean)
    print("=====\n")

    print('Cov', cov)
    print('Var', var)
    b1 = cov / var
    bo = x_mean - (b1 * y_mean)
    sreg = ssxreg(b1, var)
    stot = getVariance(x_list, x_mean)
    sreg = round(sreg)
    stot = round(stot)

    print('SS reg ', sreg)
    print('SS total ', stot)
    r2 = sreg / stot
    print('R-square ', r2)
    print('b0 ', bo)
    print('b1 ', b1)
    return b1, bo, r2

def predict(dataset, years, libname):
    x_list = dataset['population']
    y_list = dataset['year']
    b1, b0, r2 = getSlope(x_list, y_list)
    df_ylist = pd.DataFrame(y_list)
    df_xlist = pd.DataFrame(x_list)
    predX = []
    predY = []

    predDf = pd.DataFrame()
    for year in years:
        df_ylist = df_ylist.append({'year': int(year)}, ignore_index=True)

        pred = b0 + b1 * year
        df_xlist = df_xlist.append({'population': pred}, ignore_index=True)
        predX.append(pred)
        predY.append(year)
    predDf['year'] =  pd.DataFrame(predY,columns=['year'])
    predDf['prediction'] =  pd.DataFrame(predX,columns=['prediction'])
    xfit= [b0 + (b1 * year) for year in df_ylist['year']]
    print(predDf)
    plot_data(df_ylist['year'], df_xlist['population'], xfit, predY, predX, libname+'.jpg')
    return predDf
    


def plot_data(x, y, xfit, predX, predY, figname):
    plt.figure(dpi=120)
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.scatter(x, y)
    plt.scatter(predX, predY, color='red')
    plt.plot(x, xfit, color= 'black')
    plt.savefig('./figs/'+figname, dpi=800)
    plt.show()




def predictUsingLib(dataset, libname, years):
    lr = LinearRegression()
    y_list = dataset[['population']]
    x_list = dataset[['year']]
    model = lr.fit(x_list.values,y_list.values)
    xfit = model.predict(x_list.values)
    pred_X = pd.DataFrame(years,columns=['year'])
    pred_X['prediction'] = model.predict(pred_X[['year']].values)
    plot_data(x_list, y_list, xfit, years, pred_X['prediction'], libname+'.jpg')
    b1 = model.coef_
    b0 = model.intercept_

    print("Intercepts \n")
    print('b0 ', b1[0][0])
    print('b1 ', b0[0])

    print("\n")
    print("Prediction are: \n")
    print(pred_X)
    return pred_X

