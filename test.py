from scipy.special import comb
import math

def cal(k):
    return (k+2)/0.02-100/0.02*(1-math.pow(0.51/0.49,k+2))/(1-math.pow(0.51/0.49,100))

for i in range (0,10):
    print (i,cal(i))