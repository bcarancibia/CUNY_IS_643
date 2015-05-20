'''
Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements observed over time, 
containing noise (random variations) and other inaccuracies, and produces estimates of unknown variables that tend to be more precise 
than those based on a single measurement alone. 
More formally, the Kalman filter operates recursively on streams of noisy input data to produce a statistically optimal estimate of the underlying system state.

http://en.wikipedia.org/wiki/Kalman_filter

Used this heavily to implement: https://hal.archives-ouvertes.fr/file/index/docid/433886/filename/Laaraiedh_PythonPapers_Kalman.pdf

used apple and google stock ids

Alpha = -0.02
Beta = 0.0
Benchmark Returns = 81.79%


From 2011-01-01 to 2015-05-18 with $500,000 initial capital (daily data)



'''

from numpy import diag, eye, zeros, pi, exp, dot, sum, tile, linalg, array
from numpy import log as logg
from numpy.linalg import inv, det
from numpy.random import randn
import numpy as np
import random

# Algorithm plots both the observed price and the signal produced

def initialize(context):
    #2 securities
    context.total_securites = 2
    #apple and google
    context.securites = [sid(26578), sid(24)]
    context.X = np.zeros((context.total_securites*2,1))
    context.P = np.diag(np.zeros(context.total_securites*2))
    context.A = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0,0]])    
    context.Q = eye(context.X.shape[0])
    context.B = eye(context.X.shape[0])
    context.U = zeros((context.X.shape[0],1))
    context.Y = array([[0], [0]])
    context.H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
    context.R = eye(context.Y.shape[0])

def handle_data(context, data):
   #prediction
   (context.X, context.P) = kalman_predict(context.X, context.P, context.A, context.Q, context.B, context.U)
   #update
   (context.X, context.P, K, IM, IS, LH) = kalman_update(context.X, context.P, context.Y, context.H, context.R)  
   #price data
   context.Y = array([[data[sid(4283)].price],[data[sid(5885)].price]])
      
   goog_price_observed = context.Y[0][0]    
   goog_price_filter = context.X[0][0]
   aapl_price_observed = context.Y[1][0]    
   aapl_price_filter = context.X[1][0]
   
   record(GOOG_actual = goog_price_observed, GOOG_filtered = goog_price_filter)
   record(AAPL_actual = aapl_price_observed, AAPL_filtered = aapl_price_filter) 

# X = mean state estimate of the previous step (k-1)
# P = state covariance matrix of the previous step (k-1)
# A = transition nxn matrix
# Q = process noise covariance matrix
# B = input effect matrix
# U = control input

def kalman_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X,P)


#update
# At time step k, this update computes the posterior mean X and covariance P of the system state given a new measurement Y. 
# Y = measurement vector
# H = measurement matrix
# R = measurement covariance matrix
# K = Kalman Gain matrix
# IM = mean of the predictive distribution of Y
# IS = covariance of the predictive mean of Y
# LH = predictive probability (lieklihood) of measurement

def kalman_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)
    
def gauss_pdf(X, M, S):
    if M.shape[1] == 1:
        DX = X - tile(M, X.shape[1])
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * logg(2 * pi) + 0.5 * logg(det(S))
        P = exp(-E)
    return (P[0],E[0])
    