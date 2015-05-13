'''
This is a machine learning algorithm implemented similar to the first pairs trading assignment. The instructions for the assignment is to: 

1) Develop a machine learning approach and craft a trading strategy based on it.
2) You may explore on you own or use the Random Forest or Neural Network methods introduced in class.
3) Design rules for the trade including thresholds and any limits.
4) Back test from Jan 1, 2010 using Quantopian.

For this assignment I implemented a random forest classifier. I used this machine learning post to craft my answers: 

https://www.quantopian.com/posts/simple-machine-learning-example

Alpha = 115.20
Beta = -22.35
Sharpe = 1.59

'''

from sklearn.ensemble import RandomForestClassifier as rfc
from numpy import std as std
from numpy import mean as mean


def initialize(context):
    context.warmuptest = False
    context.years = 5
    context.workingDays = 250
    #stock
    context.stocks = [sid(698)]
    context.models = []
    context.historicalDays = 30
    context.predictionDays = 5
    context.percentChange = .05
    context.amountToBuy = 20
    context.daysTillSale = []
        
def handle_data(context, data):
    i = 0
    #iterate and sell stocks
    for j in range(len(context.daysTillSale)):
        context.daysTillSale[i][0] = context.daysTillSale[i][0] - 1
        if context.daysTillSale[i][0] <= 0:
            order(context.stocks[i], -(context.daysTillSale[i][1]))
    
    #training
    if context.warmuptest == False:
        print('training')
        currHist = history(bar_count=context.years * context.workingDays, frequency='1d', field='price')[context.stocks[i]]
        context.models.append(train_model(currHist, context))
        context.warmuptest = True
        
    #history
    testHist = history(bar_count=context.historicalDays, frequency='1d', field='price')[context.stocks[i]]
    testChanges = []
    
    #delta
    for j in range(len(testHist)-1):
        testChanges.append((testHist[j+1] - testHist[j])/testHist[j])
    
    prediction = context.models[i].predict(testChanges)[0]
    #trades
    if prediction == 1:
        order(context.stocks[i], context.amountToBuy)
        context.daysTillSale.append([context.predictionDays, context.amountToBuy])
    elif prediction == -1:
        order(context.stocks[i], -(context.amountToBuy))
        context.daysTillSale.append([context.predictionDays, -(context.amountToBuy)])
    
def train_model(currHist, context):
    #training datasets
    trainingX = []
    trainingY = []
    priceChanges = []
    #delta
    for i in range(len(currHist)-1):
        priceChanges.append((currHist[i+1] - currHist[i])/currHist[i])
    
    #dataset creation
    for i in range(len(currHist) - (context.historicalDays + context.predictionDays)):
        currDay = (i + context.historicalDays + context.predictionDays)
        currValue = 0
        if currHist[currDay] > currHist[currDay - context.predictionDays] * (1 + context.percentChange):
            currValue = 1
        elif currHist[currDay] < currHist[currDay - context.predictionDays] * (1 - context.percentChange):
            currValue = -1
        tempList = []
        for j in range(context.historicalDays - 1):
            tempList.append(priceChanges[i+j])
        trainingX.append(tempList)
        trainingY.append(currValue)
        
    #classifier
    clf = rfc()
    clf.fit(trainingX, trainingY)
    return(clf)