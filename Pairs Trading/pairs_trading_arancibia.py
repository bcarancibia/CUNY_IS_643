'''
This is the pairs trading algorithm implemented on the quantopian website
The pairs used to do the trading are Coco-Cola and Pepsi, Walmart and Target, and HP Billiton and BHP Billiton
what is interesting is that when HP Billiton and BHP Billiton are removed then the performance decreases. This makes sense
based on the whole premise of pairs trading. What was useful about this assignment and really drove the point home is that
pairs trading through different markets has more of an effect than pairs trading with a specific market, i.e. diversification
with Coco-Cola and Pepsi and Walmart and Target (instead of using like Lipton and Snapple as other set of pairs). 
As the assignment said there are three parts (create pairs, the trading, and testing cointegration).

Total Returns = 299.9%
Benchmark Returns = 73.9%
Alpha = 0.94
Beta = -1.62
Sharpe = 1.48

'''
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np

def initialize(context):
    # use following stocks from wikipedia page: http://en.wikipedia.org/wiki/Pairs_trade
    # Coco-Cola and Pepsi
    # Walmart and Target
    # HP Billiton Limited and BHP Billiton

    
    context.stocks = [[sid(4283), sid(5885)],
                      [sid(8229), sid(21090)],
                      [sid(863), sid(25165)]]
    
    # lag value/flag
    context.warmupDays = 60
    context.warmedUp = False
    
    # create ratio, historical, and current holdings
    lenCon = len(context.stocks)
    context.ratio = [[]] * lenCon
    context.historical = [[[],[]]] * lenCon
    context.currDays = 0
    
    # cointegration
    context.cointegrated = []
    
    # standard deviation amount results in buy
    context.SDDiff = 1
    
    # mean the stock was last bought at
    context.limits = [False] * lenCon
    
    context.spread = [[]] * lenCon
    
    context.amountStock = [[0,0]] * lenCon
 


# trade event for the securities specified. 
def handle_data(context, data):
  # Check flag for lag
    if context.warmedUp == False:
        for pair in range(len(context.stocks)):
            currPair = context.stocks[pair]
            context.ratio[pair].append(data[currPair[0]].price/data[currPair[1]].price)
            context.historical[pair][0].append(data[currPair[0]].price)
            context.historical[pair][1].append(data[currPair[1]].price)
            context.spread[pair].append(data[currPair[0]].price - data[currPair[1]].price)

        if len(context.ratio[0]) >= 60:
            context.warmedUp = True
            for pair in range(len(context.stocks)):
            #cointegration relationship testing. 
                context.cointegrated.append(test_coint(context.historical[pair]))
            if False in context.cointegrated:
                print("First pair that is not cointegrated:")
                # This could be built out to iterate if we are searching for pairs, but all of the pairs I have chosen cointegrate
                print(context.stocks[np.where([not i for i in context.cointegrated])[0][0]])
    else:
        #trade
        for pair in range(len(context.stocks)):
            currPair = context.stocks[pair]
    
            currX = currPair[0]
            currY = currPair[1]
            
            currXPrice = data[currX].price
            currYPrice = data[currY].price
            
            spreadMean = np.mean(context.spread[pair])
            spreadSD = np.std(context.spread[pair])
            currSpread = currXPrice - currYPrice
            context.spread[pair].append(currSpread)
            
            currRatio = currXPrice/currYPrice

            #number stocks
            stocksToOrderX = 1000 * currRatio
            stocksToOrderY = 1000
            
            # number stocks owned
            currOwnedX = context.portfolio.positions[currPair[0]]['amount']
            currOwnedY = context.portfolio.positions[currPair[1]]['amount']
       
            
            
            # if not all(i == False for i in context.limits):
            toCheck = [i for i, j in enumerate(context.limits) if j != False]
            if pair in toCheck:
                #pair, spreadMean, 'long'/'short'
                lim = context.limits[pair]
                if lim[2] == 'long':
                    if currSpread <= spreadMean:
                        order(currX, -currOwnedX)
                        order(currY, -currOwnedY)
                        context.limits[pair] = False
                else:
                    if currSpread >= spreadMean:
                        order(currX, -currOwnedX)
                        order(currY, -currOwnedY)
                        context.limits[pair] = False
            
            # first trade
            lowerLim = -100000
            # currOwnedX < upperLim and and currOwnedY < upperLim
            if currOwnedX > lowerLim  and currOwnedY > lowerLim:          
                if currSpread > spreadMean + context.SDDiff * spreadSD:
                        order(currX, -stocksToOrderX)
                        order(currY, stocksToOrderY)
                        print("Bought " + str(stocksToOrderY) + " stocks of " + str(currY))
                        print("Shorted " + str(stocksToOrderX) + " stocks of " + str(currX))
                        context.limits[pair] = [pair, spreadMean, 'long']
                elif currSpread < spreadMean - context.SDDiff * spreadSD:
                        order(currX, stocksToOrderX)
                        order(currY, -stocksToOrderY)
                        print("Bought " + str(stocksToOrderX) + " stocks of " + str(currX))
                        print("Shorted " + str(stocksToOrderY) + " stocks of " + str(currY))
                        context.limits[pair] = [pair, spreadMean, 'long']

            # data retention
            context.historical[pair][0].append(data[currPair[0]].price)
            context.historical[pair][1].append(data[currPair[1]].price)

            
def test_coint(pair):
    result = sm.OLS(pair[1], pair[0]).fit()   
    dfResult =  ts.adfuller(result.resid)
    return dfResult[0] >= dfResult[4]['10%']