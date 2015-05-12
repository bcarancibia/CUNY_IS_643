'''
This is a machine learning algorithm implemented similar to the first pairs trading assignment. The instructions for the assignment is to: 

1) Develop a machine learning approach and craft a trading strategy based on it.
2) You may explore on you own or use the Random Forest or Neural Network methods introduced in class.
3) Design rules for the trade including thresholds and any limits.
4) Back test from Jan 1, 2010 using Quantopian.

For this assignment I implemented a random forest classifier. I used this machine learning post to craft my answers: 

https://www.quantopian.com/posts/simple-machine-learning-example



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
    # If you would like to trade on more than 1 stock, you could use the i to iterate
    i = 0
    # Iterate through and sell stocks that need to be sold
    for j in range(len(context.daysTillSale)):
        context.daysTillSale[i][0] = context.daysTillSale[i][0] - 1
        if context.daysTillSale[i][0] <= 0:
            order(context.stocks[i], -(context.daysTillSale[i][1]))
    
    # Training
    if context.warmuptest == False:
        print('training')
        currHist = history(bar_count=context.years * context.workingDays, frequency='1d', field='price')[context.stocks[i]]
        context.models.append(train_model(currHist, context))
        context.warmuptest = True
        
    # Getting days of history
    testHist = history(bar_count=context.historicalDays, frequency='1d', field='price')[context.stocks[i]]
    testChanges = []
    
    # Percent Change
    for j in range(len(testHist)-1):
        testChanges.append((testHist[j+1] - testHist[j])/testHist[j])
    
    # The prediction
    prediction = context.models[i].predict(testChanges)[0]
    # Trades on the prediction
    if prediction == 1:
        order(context.stocks[i], context.amountToBuy)
        context.daysTillSale.append([context.predictionDays, context.amountToBuy])
    elif prediction == -1:
        order(context.stocks[i], -(context.amountToBuy))
        context.daysTillSale.append([context.predictionDays, -(context.amountToBuy)])
    
def train_model(currHist, context):
    # Creates the training datasets, x being the variables, y being the classification
    trainingX = []
    trainingY = []
    priceChanges = []
    # Percent Change
    for i in range(len(currHist)-1):
        priceChanges.append((currHist[i+1] - currHist[i])/currHist[i])
    
    # Creates the dataset from the history
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
        
    # Trains the classifier
    clf = rfc()
    clf.fit(trainingX, trainingY)
    return(clf)