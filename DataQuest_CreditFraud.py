
import pandas as pd
import numpy as np
import scipy.optimize as so


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


#function to train model
def get_mae(max_nodes, train_x, val_x, train_y, val_y):
    total = 0
    model = DecisionTreeRegressor(max_leaf_nodes = max_nodes, random_state = 13)
    model.fit(train_x, train_y)
    model_Predictions = model.predict(val_x)
    for item in model_Predictions:
        total += item
    results = total/len(model_Predictions)
    MAE = [mean_absolute_error(val_y, model_Predictions), results]

    return (MAE)

def simpleRegPredict(b,x):
    yp=b[0]+b[1]*x
    return yp

def simpleRegLossRSS(b,x,y):
    predY = simpleRegPredict(b,x)
    res = y-predY
    rss = sum(res**2)
    deriv = np.zeros(2)
    deriv[0]=-2*sum(res)
    deriv[1]=-2*sum(res*x)
    return (rss,deriv)

def simpleRegFit(x,y,lossFcn=simpleRegLossRSS):
    b0=[np.mean(y),0]
    RES = so.minimize(lossFcn,b0,args=(x,y),jac=True)
    b=RES.x # Results
    res = y-np.mean(y)
    TSS = sum(res**2)
    RSS,grad = simpleRegLossRSS(b,x,y)
    R2 = 1-RSS/TSS
    return (R2,b)

def addPrediction(b,x,predFunc,ax,linestyle='r:'):
    xrange = max(x)-min(x)
    xp = np.linspace(min(x)-xrange/30,max(x)+xrange/30,num=50)
    yp = predFunc(b,xp)
    ax.plot(xp,yp,linestyle)
    return(ax)

def multRegPredict(b,D,xname):
    yp=np.ones(len(D.index))*b[0]        
    for i in range(len(xname)):
        yp=yp+D[xname[i]]*b[i+1]        
    return yp

def multRegLossRSS(b,D,y,xname):
    predY = multRegPredict(b,D,xname)
    res = y-predY
    rss = sum(res**2)
    grad=np.zeros(len(b))
    grad[0]=-2*np.sum(res)
    for i in range(len(xname)):
        grad[i+1]=-2*np.sum(D[xname[i]]*res)
    return (rss,grad)

def multRegFit(D,y,xname):
    N=len(xname)
    b0=np.zeros((N+1,))
    RES = so.minimize(multRegLossRSS,b0,args=(D,y,xname),
                    jac=True,
                    tol=1e-3)
    if (not(RES.success)):
        print('unsuccessful fit')
        print(RES)
    b=RES.x # Results
    res = y-np.mean(y)
    TSS = sum(res**2)
    RSS,deriv = multRegLossRSS(b,D,y,xname)
    R2 = 1-RSS/TSS
    return (R2,b)

def bootstrapReg(D,y,args,fitfcn,numIter=1000):
    r2,b=fitfcn(D,y,args)
    numParams=len(b)
    N = len(D.index)
    ind  = np.arange(N)
    stats = np.zeros((numIter,numParams))
    for i in range(numIter):
        sample=np.random.choice(ind,N)
        r2,stats[i,:]=fitfcn(D.iloc[sample],y[sample],args)
    return stats

def confidenceInt(data,perc):
    Int=(np.percentile(data,(100-perc)/2),np.percentile(data,100-(100-perc)/2))
    return Int

train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

train_data = train.dropna()
test_data = test.dropna()

print(train_data.head())
print(test_data.head())

y = train_data.isFraud


