# SIR -- Deterministic model

import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S, I, R, beta, gamma, N):
    ''' 
    The simple SIR model :
    S = Susceptible
    I = Infected
    R = Recovered
    N = total population
    beta, gamma = infection rate, recovery rate respectively
    '''
    dSdt = -beta*S*I / N
    dIdt = beta*S*I / N - gamma*I
    dRdt = gamma*I
    return dSdt, dIdt, dRdt

# Runge kutta method for simple SIR model.
def rk4(SIR_model, S_values, I_values, R_values, t0, tf, dt, beta, gamma, N):
    steps = int((tf - t0) / dt) + 1

    t = np.linspace(t0, tf, steps)
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    S[0]=S_values
    I[0]=I_values
    R[0]=R_values

    for i in range(1,steps):
        k1S,k1I,k1R = SIR_model(S[i-1],I[i-1],R[i-1], beta, gamma, N)
        k2S,k2I,k2R = SIR_model(S[i-1] + 0.5 * dt * k1S, I[i-1] + 0.5 * dt * k1I, R[i-1] + 0.5 * dt * k1R, beta, gamma,N)
        k3S,k3I,k3R = SIR_model(S[i-1] + 0.5 * dt * k2S, I[i-1] + 0.5 * dt * k2I, R[i-1] + 0.5 * dt * k2R, beta, gamma,N)
        k4S,k4I,k4R = SIR_model(S[i-1] + dt * k3S, I[i-1] + dt * k3I, R[i-1] + dt * k3R, beta, gamma,N)

        S[i] = S[i-1] + (dt / 6) * (k1S + 2 * k2S + 2 * k3S + k4S)
        I[i] = I[i-1] + (dt / 6) * (k1I + 2 * k2I + 2 * k3I + k4I)
        R[i] = R[i-1] + (dt / 6) * (k1R + 2 * k2R + 2 * k3R + k4R)
    return t,S,I,R


S_values = 1998  # 90% susceptible
I_values = 2  # 10% infected
R_values = 0   # initially - 0 recovery
N = S_values+I_values

# Time parameters
tf = 365
t0 = 0
dt = 1

#Model parameters
beta = 0.1
gamma = 1/22

t,S,I,R = rk4(SIR_model, S_values, I_values, R_values, t0, tf, dt, beta, gamma, N)

plt.plot(t, S, "yellow")
plt.plot(t, I, "red")
plt.plot(t, R, "green")
plt.xlabel("time")
plt.ylabel("population")
plt.legend()
plt.show()


