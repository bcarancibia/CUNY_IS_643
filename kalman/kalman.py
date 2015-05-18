'''

'''

import numpy as np
import pandas as pd

def initialize(context):
    # The stocks I would like to trade
    context.stocks = [sid(4283), sid(5885),
                      sid(8229), sid(21090),
                      sid(8347), sid(23112),
                      sid(863), sid(25165)]
    
    context.daysTillSale = [None] * len(context.stocks)
    for i in range(len(context.daysTillSale)):
        context.daysTillSale[i] = []
    context.percent = 0.01
    context.amountToBuy = 200
    context.predictionDays = 60
    context.trainingHistLen = 30
        
def handle_data(context, data):
    for i in range(len(context.stocks)):
        for j in range(len(context.daysTillSale[i])):
            context.daysTillSale[i][j][0] = context.daysTillSale[i][j][0] - 1
            if context.daysTillSale[i][j][0] == 0:
                order(context.stocks[i], -(context.daysTillSale[i][j][1]))
                # print("cashed in")
        context.daysTillSale[i] = filter(lambda a: a[0] != 0, context.daysTillSale[i])
    
        currPrice = data[context.stocks[i]].price
        currHist = history(bar_count=context.trainingHistLen, frequency='1d', field='price')[context.stocks[i]]
        prediction = kalman_filter(currHist, currPrice)
        record(pred = prediction)
        record(price = currPrice)
        percentChange = (prediction - currPrice)/currPrice
        if context.portfolio.cash > currPrice * context.amountToBuy:
            if percentChange > context.percent:
                # Buy
                order(context.stocks[i], context.amountToBuy)
                # print("bought")
                context.daysTillSale[i].append([context.predictionDays, context.amountToBuy])
            elif  percentChange < -context.percent:
                # Short
                order(context.stocks[i], -(context.amountToBuy))
                context.daysTillSale[i].append([context.predictionDays, -(context.amountToBuy)])
                # print("shorted")
        else:
            print('out of cash!')
    #print(context.portfolio.cash)
    #print "-" * 25

    
def kalman_filter(currHist, currPrice):  
    # Kalman filter
    x = 0
    p = 1
    R = 0.1
    k = 0
    for day in range(len(currHist)):
        x_prev = x
        p_prev = p
        z_k = currHist[day]
        k = update_k(p_prev, R)
        x = update_x(x_prev, k, z_k)
        p = update_p(k, p_prev)
    # Having a hard time appending the price to the data frame, so just run it one more time on the current price
    #x_prev = x
    #p_prev = p
    #z_k = currPrice
    #k = update_k(p_prev, R)
    #x = update_x(x_prev, k, z_k)
    #p = update_p(k, p_prev)
    return(x)
    
def update_k(p_prev, R):
    return(p_prev/(p_prev + R))

def update_x(x_prev, k, z_k):
    return(x_prev + k * (z_k - x_prev))

def update_p(k, p_prev):
    return((1-k) * p_prev)