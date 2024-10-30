"""
SIR using differential equations
This model also includes Birth, Death and Lockdown

Submitted by : Abhishek Shirsekar
Roll  Number : MT2130
"""

import matplotlib.pyplot as plt

#Common parameters
beta = 0.002
gamma = 0.4
del_birth = 0.01
delta = 0.7

def sus(S, I, R, lam):
    global beta, del_birth
    N = S + I + R
    return -beta*S*I*lam + (del_birth * N)


def inf(S, I, lam):
     global beta, gamma, delta
     return beta*S*I*lam-gamma*I

def rec(I):
     global gamma, delta
     return gamma*delta*I

def death(I):
     global gamma, delta
     return gamma*(1-delta)*I

def birth(S, I, R):
    global del_birth
    N = S + I + R
    return N*del_birth
    
#rk4 method
def rk4(S, I, R, n):

   #calculate step size
   h=0.002
   S_list = []
   I_list = []
   R_list = []
   D_list = []
   B_list = []
   D=0
   B=0
   total = S + I + R

   for i in range(1,n):
      N = S + I + R
      lam = 1
      if(i%2 == 0):
          lam = 0.5
      k1 = h * (sus(S, I, R, lam))
      l1 = h * (inf(S, I, lam))
      m1 = h * (rec(I))
      d1 = h * (death(I))
      b1 = h * (birth(S, I, R))
      
      k2 = h * (sus((S+(k1)/2), (I+(l1)/2), (R+(m1)/2), lam))
      l2 = h * (inf((S+(k1)/2), (I+(l1)/2), lam))
      m2 = h * (rec((I+(l1)/2)))
      d2 = h * (death((I+(l1)/2)))
      b2 = h * (birth((S+(k1)/2), (I+(l1)/2), (R+(m1)/2)))
      
      k3 = h * (sus((S+(k2)/2), (I+(l2)/2), (R+(m2)/2), lam))
      l3 = h * (inf((S+(k2)/2), (I+(l2)/2), lam))
      m3 = h * (rec((I+(l2)/2)))
      d3 = h * (death((I+(l2)/2)))
      b3 = h * (birth((S+(k2)/2), (I+(l2)/2), (R+(m2)/2)))
      
      k4 = h * (sus((S+(k3)), (I+(l3)), (R+(m3)), lam))
      l4 = h * (inf((S+(k3)), (I+(l3)), lam))
      m4 = h * (rec((I+(l3))))
      d4 = h * (death((I+(l3))))
      b4 = h * (birth((S+(k3)), (I+(l3)), (R+(m3))))
      
      k = (k1+(2*k2)+(2*k3)+k4)/6
      S = S + k
      
      l = (l1+(2*l2)+(2*l3)+l4)/6
      I = I + l
      
      m = (m1+(2*m2)+(2*m3)+m4)/6
      R = R + m
      
      d = (d1+(2*d2)+(2*d3)+d4)/6
      D = D + d
      
      b = (b1+(2*b2)+(2*b3)+b4)/6
      B = B + b

      S_list.append(S)
      I_list.append(I)
      R_list.append(R)
      D_list.append(D)
      B_list.append(B)
      
      if((S<0) or (I<0) or (R<0) or (D<0)):
         S = S - k
         I = I - l
         R = R - m
         D = D - d
         break
    
   peak_infections_index = I_list.index(max(I_list))
   text_to_add = "Peak infections = " + str(round(max(I_list), 2)) + " ; at time = " + str(peak_infections_index)
   plt.plot(list(range(len(S_list))), S_list, color = 'green')
   plt.plot(list(range(len(I_list))), I_list, color = 'red')
   plt.plot(list(range(len(R_list))), R_list, color = 'blue')
   plt.plot(list(range(len(D_list))), D_list, color = 'black')
   plt.plot(list(range(len(B_list))), B_list, color = 'orange')
   plt.text((n/5), (0.95 * total), text_to_add, color = 'red')
   plt.hlines(y = max(I_list), xmin = 0, xmax = peak_infections_index, color = 'r', linestyle = 'dashed', linewidth = 0.9)
   plt.vlines(x = peak_infections_index, ymin = 0, ymax = max(I_list), color = 'r', linestyle = 'dashed', linewidth = 0.9)  
   plt.gca().legend(['Susceptible', 'Infected', 'Recovered', 'Death', 'Birth'])
   plt.title('SIR using Differential equations - Lockdown')
   plt.show()


######### Main #########
S=990
I=10
R=0
n=5000

#rk4 method call
rk4(S, I, R, n)
