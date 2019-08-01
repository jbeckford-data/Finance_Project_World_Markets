
# coding: utf-8

"""
# Final Project:
# 1: Download data  for 25 years for global markets
# 2: Check if prices follow a log-normal distribution
# 3: Check if returns follow a normal distribution
# 4: Random walk: Assuming geometric Brownian and 20% historical volatility write a sub-module
# to calculate the event like the 1987 crash.  Explain in simple terms what the results imply.
# 5: What does "Fat tail" mean? Plot the distribution of price movement.


# Use the following market indices:
# A) Dow Jones
# B) S&P 500
# C) NASDAQ
# D) DAX
# E) FTSE
# F) HANGSENG
# G) KOSPI (Korea)
# H) CNX NIFTY (INDIA)
# 
# 

YAHOO SYMBOLS FOR MARKETS:
DOW JONES: ^DJI
SP 500: ^GSPC
NASDAQ: ^IXIC
DAX: ^GDAXI
FTSE: ^FTSE
HANG SENG: ^HSI
KOSPI: ^KS11
CNX NIFTY: ^NSEI
"""

import pandas as pd
import numpy as np
from numpy.random import normal
from scipy import stats as sci
# from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.mlab as mlab

# Import random generators for Hurst Exponent formula
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
style.use('ggplot')

#YAHOO Fix
# yf.pdr_override() 

symbols = ['^DJI', '^GSPC', '^IXIC', '^GDAXI', '^HSI', '^KS11', '^NSEI']

start = '1/1/1994'

def get_prices(symbols, start):
            
            main_df = pd.DataFrame()
            for x in symbols:
                print(x)
                df = si.get_data(x,start_date=start)
                df.rename(columns={'adjclose': x}, inplace=True)
                df.drop(columns=['open','high','low','close','volume','ticker']
                       ,inplace=True)
                df[x+' Returns'] = df[x].pct_change()

                if main_df.empty:
                    main_df = df
                else:
                    main_df = main_df.join(df,how='outer') 
            return main_df

df = get_prices(symbols,start)
print(df.head(),df.columns)

#Separate Each Symbol
dow = df['^DJI']
spx = df['^GSPC']
ndx = df['^IXIC']
dax = df['^GDAXI']
# ftse = df['^FTSE']
hsi = df['^HSI']
ks = df['^KS11']
nsei = df['^NSEI']

# """ Part 1: Download All Data for Markets"""
# # download Panel
# # If download does not work, please simply run again.  Sometimes there may be an issue with the server
# data = pdr.get_data_yahoo(symbols, start="1993-01-01", end="2018-04-30")

# #Format DataFrame to minimize interference
# df = data.to_frame()
# df.drop(['Close', 'High', 'Low', 'Open', 'Volume'],1,inplace=True)
# df.reset_index(inplace=True)

# #['^DJI', '^GSPC', '^IXIC', '^GDAXI', '^FTSE', '^HSI', '^KS11', '^NSEI']
# #Separate Each Symbol
# dow = df[df['minor'] == '^DJI']
# spx = df[df['minor'] == '^GSPC']
# ndx = df[df['minor'] == '^IXIC']
# dax = df[df['minor'] == '^GDAXI']
# ftse = df[df['minor'] == '^FTSE']
# hsi = df[df['minor'] == '^HSI']
# ks = df[df['minor'] == '^KS11']
# nsei = df[df['minor'] == '^NSEI']

#Calculate the Returns
# dow['Return'] = dow['Adj Close'].pct_change()[1:]
# ndx['Return'] = ndx['Adj Close'].pct_change()[1:]
# spx['Return'] = spx['Adj Close'].pct_change()[1:]
# dax['Return'] = dax['Adj Close'].pct_change()[1:]
# ftse['Return'] = ftse['Adj Close'].pct_change()[1:]
# hsi['Return'] = hsi['Adj Close'].pct_change()[1:]
# ks['Return'] = ks['Adj Close'].pct_change()[1:]
# nsei['Return'] = nsei['Adj Close'].pct_change()[1:]



# syms = [dow,spx,ndx,dax,ftse,hsi,ks,nsei]
syms = [dow,spx,ndx,dax,hsi,ks,nsei]

"""Part 2: Do prices follow a log-normal distribution?"""

print("Part 2: Do Prices Follow a log-normal distribution?")
print("For this we can find a Hurst Exponent and test \
if the value is mean-reverting, trending, or log-normal(geometric brownian)\n")

#Hurst Exponent
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

# Create a Random Gometric Brownian Motion, Mean-Reverting and Trending Series
# To test formula for validation
gbm = log(cumsum(randn(100000))+1000)
mr = log(randn(100000)+1000)
tr = log(cumsum(randn(100000)+1)+1000)

#Hurst Exponent for each stock
# h_dow = hurst(dow['Adj Close'])
# h_spx = hurst(spx['Adj Close'])
# h_ndx = hurst(ndx['Adj Close'])
# h_dax = hurst(dax['Adj Close'])
# h_ftse = hurst(ftse['Adj Close'])
# h_hsi = hurst(hsi['Adj Close'])
# h_nsei = hurst(nsei['Adj Close'])
# h_ks = hurst(ks['Adj Close'])

h_dow = hurst(dow)
h_spx = hurst(spx)
h_ndx = hurst(ndx)
h_dax = hurst(dax)
h_hsi = hurst(hsi)
h_nsei = hurst(nsei)
h_ks = hurst(ks)

#['^DJI', '^GSPC', '^IXIC', '^GDAXI', '^FTSE', '^HSI', '^KS11', '^NSEI']
# hurst_exponent = [h_dow, h_spx, h_ndx, h_dax, h_ftse, h_hsi, h_ks, h_nsei]
hurst_exponent = [h_dow, h_spx, h_ndx, h_dax, h_hsi, h_ks, h_nsei]


# Output the Hurst Exponent for each of the above series

# Should be == .5
print('\nGeomtric Brownian should be ~= .5')
print ("Geometric Brownian Hurst:   %s" % hurst(gbm))

# Should be == 0
print('\nMean Reverting should be ~= 0')
print ("Mean-Reverting Hurst:    %s" % hurst(mr))

# Should be == 1
print('\nTrending should be ~= 1')
print ("Trending Hurst:    %s" % hurst(tr))
print()

#Print All Hurst values for Indices measured
for x in range(len(symbols)):
    print ("Hurst %s:  %s" % (symbols[x],hurst_exponent[x]))
print()


# No asymptotic distribution theory has been derived for most of the Hurst exponent estimators so far.
#However, Weron, Rafał (2002-09-01) used bootstrapping to obtain approximate functional 
#forms for confidence intervals of the two most popular methods, i.e.,
#for the Annis, A. A.; Lloyd, E. H. (1976-01-01) corrected R/S analysis:
# 
# Level	           Lower bound	                         Upper bound
# 90%	0.5 − exp(−7.35 log(log M) + 4.06)	       exp(−7.07 log(log M) + 3.75) + 0.5
# 95%	0.5 − exp(−7.33 log(log M) + 4.21)	       exp(−7.20 log(log M) + 4.04) + 0.5
# 99%	0.5 − exp(−7.19 log(log M) + 4.34)	       exp(−7.51 log(log M) + 4.58) + 0.5
#
# Where M=log_base 2(N) and N is the series length. 
    

#For all symbols run p-value test for Hurst exponent
for x in range(len(syms)):    
    n = len(syms[x])
    m = np.log2(n)
    
    # Values for a 5% or .05 p-value test -- From Above
    lower = 0.5 - np.exp(-7.33 * log(log(m)) + 4.21)
    upper = np.exp(-7.20 * log(log(m)) + 4.04) + 0.5
    bounds = [lower, upper]
        
    #Print Conclusion of Hypothesis Tests
    if hurst_exponent[x] < bounds[0] or hurst_exponent[x] > bounds[1]:
        print(symbols[x]+' NOT within 5% p-value range '+str(bounds))
        print('As Such, we CAN REJECT the null hypothesis that %s follows\
 Geometric Brownian Motion\n' % (symbols[x]))
    else:
        print(symbols[x]+' within 5% p-value range '+str(bounds))
        print('As Such, we cannot reject the null hypothesis that %s follows\
 Geometric Brownian Motion\n' % (symbols[x]))


""" Part 3: Plot Returns - Follow Normal Distribution"""
print('\nPart 3: Do returns follow a Normal Distribution with mean 0?\n\
To test this, we can use a one-sample t-test to discover if the true mean \
could be zero with the prices observed.  If our p-value of the test is less than 5% or .05 \
we can reject the null hypothesis that our true mean is zero based on the observed values.')
returns = [df['^DJI Returns'],
           df['^GSPC Returns'],
           df['^IXIC Returns'],
           df['^GDAXI Returns'],
           df['^HSI Returns'],
           df['^KS11 Returns'],
           df['^NSEI Returns']
           ]
#Method for Plotting Returns
def plot_ret(stock):
    mu = stock[1:].dropna().mean()
    sigma = stock[1:].dropna().std()
    sims = normal(mu,sigma,1000)
    norm = np.random.randn(1000)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Deprecated -- ax1.plot(x, mlab.normpdf(x, mu, sigma), color='blue')
    ax1.plot(x, sci.norm.pdf(x, mu, sigma), color='blue')
    ax2.hist(stock[1:], alpha=.5, bins=50, label=stock.name.split()[0])
    plt.legend()
    plt.show()

for stock in returns:
    plot_ret(stock)
    tvalue = sci.ttest_1samp(stock[1:].dropna(),0)
    #print(tvalue)
    if tvalue[1] < .05:
        print('For %s, the p-value of a one sample t-test is less than 5%%\n \
which leads us to conclude we can REJECT the null hypothesis that our sample mean is zero.\n'\
% stock.name)
    else:
        print('\nFor %s, the p-value of a one sample t-test is not less than 5%%\n\
and as such, we cannot reject the null hypothesis that our true mean is zero.\n' % stock.iloc[1])
    
""" Part 4: 1987 Crash Simulation"""

print('Part 4: 1987 Crash Simulation.\n\n\
This simulation runs 1 million random variables.  Please allow it some time.\n')
print("The 1987 Crash was a 22.61% move in 1 day, or a 20-sigma move. \
 Let's run a simulation to find out what the probability of that occurrence actually is.\n")
def run_geom_sim():
    #Geomtric Brownian Motion  Simulation for 30 years of daily data
    #T = 30 Years
    T = 30
    #Mu = Average Return -- Doesn't affect the test
    mu = .1
    #Sigma = .2 or 20% volatility
    sigma = .2
    #S0 Start Stock Price -- Again, not necessary for test
    S0 = 5
    #Day step
    dt = 1/360
    #Number of iterations
    N = round(T/dt)

    t = np.linspace(0, T, N)
    W = np.random.standard_normal(N) 
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    #print(S)
    
    #return daily movement over 30 years (10,800 observations)
    return S

pct = []
max_change = 0
count=0
while count < 1000000 and max_change < .2261:
    S = run_geom_sim()
    for x in range(1,len(S)):
        #Keep Count
        count += 1
        #Percent change of Geometric Brownian Simulation
        pct.append(1-(S[x]/S[x-1]))
    #Maximum of Percent change
    max_change = max(np.absolute(pct))
if count >= 1000000:
    print('Never occurs in simulation.\n')
    print('In conclusion, given a Geometric Brownian Motion and 20% volatility \
which also assumes a Normal Distribution (0,1), the probability of a 21% move \
in one day cannot be calculated by a simulation.\n \
In a 1 million count simulation the greatest 1-day change is: ' +str(max_change))
else:
    print("Occurred on trial %s or %f probability." % (str(count),float(1/count)))
    
"""Part 5: What does fat-tail mean?"""

print("\nPart 5: What does 'fat-tail' mean?\nPlot the distribution of price movement.\n")
print('To answer the question: a fat-tail represents data that has more observations on the outside \
or extremities (named: outliers) of the normal distribution than of the normal distribution.\n\
A statistical measure for the value is a measure of Kurtosis.\n\n')
print('We can use a statistical test to again determine the kurtosis of the observations and \
determine if the true distribution could be normal given the observed values.\n')

# Boxplot of Returns -- Green Diamonds Represent Outliers -- Should be few or 0 for Normal Distribution
def plot_boxes(df):
    green_diamond = dict(markerfacecolor='g', marker='D')
    plt.ylabel('Return')
    plt.xlabel(stock.name)
    plt.boxplot(stock[1:].dropna(),flierprops=green_diamond)
    plt.show()
    

# For each stock, run a Kurtosis test to determine if normally distributed and plot returns in boxplot.
for stock in returns:
    kurt = sci.kurtosistest(stock[1:].dropna())
    plot_boxes(stock)
    
    #If p-value is less than .05 we CAN REJECT the null that distribution is NORMALLY DISTRIBUTED
    if kurt[1] < .05:
        print('Based on the observed values of %s at a 5%% level we CAN REJECT the null hypothesis \
that this distribution is distributed normally.\n' % stock.name)
    #If p-value is greater than .05 we cannot reject the null that it is normally distibuted
    else:
        print('Based on the observed values of %s at a 5%% level we cannot reject the null hypothesis \
that this distribution is distributed normally.\n' % stock.name)
