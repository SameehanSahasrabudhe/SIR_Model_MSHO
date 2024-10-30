"""
SIR using differential equations
This model also includes Birth, Death, Lockdown, Symptomatic and Asymptomatic

Submitted by : Abhishek Shirsekar
Roll  Number : MT2130
"""

import matplotlib.pyplot as plt

#Common parameters
beta = 0.0005
gamma = 0.35
del_birth = 0.001
delta = 0.7
zeta = 0.001
omega = 0.3
tau = 0.2

def sus(S, Is, Ia, R, lam):
    global beta, del_birth, zeta
    N = S + Is + Ia + R
    return (((-S * lam) * ((beta * Is) + (zeta * Ia))) + (del_birth * N))

def inf(S, Is, Ia, lam):
     global beta, gamma, delta, tau, omega
     return ((beta * S * Is * lam) + ((1 - tau) * omega * Ia) - (gamma * Is))  

def inf_asym(S, Ia, lam):
     global zeta, omega, tau
     return ((zeta * S * Ia * lam) - (omega * Ia))

def rec(Is, Ia):
     global gamma, delta, omega, tau
     return ((gamma * delta * Is) + (omega * tau * Ia))

def death(Is):
     global gamma, delta
     return gamma*(1-delta)*Is

def birth(S, Is, Ia, R):
    global del_birth
    N = S + Is + Ia + R
    return (N * del_birth)
    
#rk4 method
def rk4(S, Is, Ia, R, n):

   #calculate step size
   h=0.002
   S_list = []
   Is_list = []
   Ia_list = []
   I_total = []
   R_list = []
   D_list = []
   B_list = []
   D=0
   B=0
   total = S + Is + Ia + R

   for i in range(1, n):
      N = S + Is + Ia + R
      lam = 1
      
      if ((i % 2) == 0):
          lam = 0.5

      k1 = h * (sus(S, Is, Ia, R, lam))
      l1 = h * (inf(S, Is, Ia, lam))
      la1 = h * (inf_asym(S, Ia, lam))
      m1 = h * (rec(Is, Ia))
      d1 = h * (death(Is))
      b1 = h * (birth(S, Is, Ia, R))
      
      k2 = h * (sus((S+(k1)/2), (Is+(l1)/2), (Ia+(la1)/2), (R+(m1)/2), lam))
      l2 = h * (inf((S+(k1)/2), (Is+(l1)/2), (Ia+(la1)/2), lam))
      la2 = h * (inf_asym((S+(k1)/2), (Ia+(la1)/2), lam))
      m2 = h * (rec((Is+(l1)/2), (Ia+(la1)/2)))
      d2 = h * (death((Is+(l1)/2)))
      b2 = h * (birth((S+(k1)/2), (Is+(l1)/2), (Ia+(la1)/2), (R+(m1)/2)))
      
      k3 = h * (sus((S+(k2)/2), (Is+(l2)/2), (Ia+(la2)/2), (R+(m2)/2), lam))
      l3 = h * (inf((S+(k2)/2), (Is+(l2)/2), (Ia+(la2)/2), lam))
      la3 = h * (inf_asym((S+(k2)/2), (Ia+(la2)/2), lam))
      m3 = h * (rec((Is+(l2)/2), (Ia+(la2)/2)))
      d3 = h * (death((Is+(l2)/2)))
      b3 = h * (birth((S+(k2)/2), (Is+(l2)/2), (Ia+(la2)/2), (R+(m2)/2)))
      
      k4 = h * (sus((S+(k3)), (Is+(l3)), (Ia+(la3)), (R+(m3)), lam))
      l4 = h * (inf((S+(k3)), (Is+(l3)), (Ia+(la3)), lam))
      la4 = h * (inf_asym((S+(k3)), (Ia+(la3)), lam))
      m4 = h * (rec((Is+(l3)), (Ia+(la3))))
      d4 = h * (death((Is+(l3))))
      b4 = h * (birth((S+(k3)), (Is+(l3)), (Ia+(la3)), (R+(m3))))
      
      k = (k1 + (2 * k2) + (2 * k3) + k4) / 6
      S = S + k
      
      l = (l1 + (2 * l2) + (2 * l3) + l4) / 6
      Is = Is + l
      
      la = (la1 + (2 * la2) + (2 * la3) + la4) / 6
      Ia = Ia + la
      
      m = (m1 + (2 * m2) + (2 * m3) + m4) / 6
      R = R + m
      
      d = (d1 + (2 * d2) + (2 * d3) + d4) / 6
      D = D + d
      
      b = (b1 + (2 * b2) + (2 * b3) + b4) / 6
      B = B + b

      S_list.append(S)
      Is_list.append(Is)
      Ia_list.append(Ia)
      I_total.append((Is + Ia))
      R_list.append(R)
      D_list.append(D)
      B_list.append(B)
      
      if((S < 0) or (Is < 0) or (Ia < 0) or (R < 0) or (D < 0)):
         S = S - k
         Is = Is - l
         Ia = Ia - la
         R = R - m
         D = D - d
         break
    
   peak_infections_index = I_total.index(max(I_total))
   text_to_add = "Peak infections = " + str(round(max(I_total), 2)) + " ; at time = " + str(peak_infections_index)
   plt.plot(list(range(len(S_list))), S_list, color = 'green')
   plt.plot(list(range(len(I_total))), I_total, color = 'brown')
   plt.plot(list(range(len(Is_list))), Is_list, color = 'red')
   plt.plot(list(range(len(Ia_list))), Ia_list, color = 'orange')
   plt.plot(list(range(len(R_list))), R_list, color = 'blue')
   plt.plot(list(range(len(D_list))), D_list, color = 'black')
   plt.plot(list(range(len(B_list))), B_list, color = 'pink')
   plt.text((n/5), (0.95 * total), text_to_add, color = 'red')
   plt.hlines(y = max(I_total), xmin = 0, xmax = peak_infections_index, color = 'r', linestyle = 'dashed', linewidth = 0.9)
   plt.vlines(x = peak_infections_index, ymin = 0, ymax = max(I_total), color = 'r', linestyle = 'dashed', linewidth = 0.9)  
   plt.gca().legend(['Susceptible', 'Infected - Total', 'Infected - Symptomatic', 'Infected - Asymptomatic', 'Recovered', 'Death', 'Birth'])
   plt.title('SIR using Differential equations - Symptomatic and Asymptomatic Infected - Lockdown')
   plt.show()


######### Main #########
S = 990
Is = 1
Ia = 9
R = 0
n = 15000

#rk4 method call
rk4(S, Is, Ia, R, n)
