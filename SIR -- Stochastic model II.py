# SIR model -- Stochastic model II

import numpy as np
import matplotlib.pyplot as plt

def simulate_epidemic(N, T, alpha, beta, gamma, initial_state=None):
    """
    Simulate epidemic spread using probabilistic model
    
    Parameters:
    N: number of agents
    T: time steps
    alpha: probability of S->I transition
    beta: probability of I->R transition
    gamma: probability of R->S transition
    initial_state: initial state distribution (optional)
    """
    
    # Define transition probability matrix
    P = np.array([
        [1-alpha, alpha, 0],    # S -> S, I, R
        [0, 1-beta, beta],      # I -> S, I, R
        [gamma, 0, 1-gamma]     # R -> S, I, R
    ])
    
    # Initialize states (0=S, 1=I, 2=R)
    states = np.zeros((N, T), dtype=int)
    
    # Set initial states randomly if not provided
    if initial_state is None:
        states[:, 0] = np.random.choice([0, 1, 2], size=N)
    else:
        states[:, 0] = initial_state
    
    # Simulate for each time step
    for t in range(1, T):
        for agent in range(N):
            current_state = states[agent, t-1]
            # Sample next state based on transition probabilities
            states[agent, t] = np.random.choice(3, p=P[current_state])
    
    # Calculate proportions for each time step
    S_prop = np.mean(states == 0, axis=0)
    I_prop = np.mean(states == 1, axis=0)
    R_prop = np.mean(states == 2, axis=0)
    
    return S_prop, I_prop, R_prop

# Set parameters
N = 1000  # number of agents
T = 500   # time steps
alpha = 0.05  # infection probability
beta = 0.01   # recovery probability
gamma = 0.001 # immunity loss probability

# Run simulation
S, I, R = simulate_epidemic(N, T, alpha, beta, gamma)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible', color='blue')
plt.plot(I, label='Infected', color='red')
plt.plot(R, label='Recovered', color='green')
plt.xlabel('Time')
plt.ylabel('Estimated Probability')
plt.title(f'Probabilistic SIR Model (alpha={alpha}, Beta={beta}, gamma={gamma}, N={N})')
plt.legend()
plt.grid(True)
plt.show()

# Calculate theoretical steady state values
denominator = (1/gamma + 1/beta + 1/alpha)
a = (1/alpha) / denominator  # Susceptible
b = (1/beta) / denominator   # Infected
c = (1/gamma) / denominator  # Recovered

print("\nTheoretical steady state values:")
print(f"Susceptible (a): {a:.3f}")
print(f"Infected (b): {b:.3f}")
print(f"Recovered (c): {c:.3f}")


#-----------------------
# Computing proportions ------------------------------------################
#-----------------------
'''
Pt+1(S) = gamma*Pt(R) + (1 - alpha)*Pt(S)
Pt+1(I) = alpha*Pt(S) + (1 - Beta)*Pt(I)
Pt+1(R) = beta*Pt(I) + (1 - gamma)*Pt(R)
Try interpreting these equations in terms of proportions instead of
probabilities
'''

def simulate_sir_proportions(N, T, alpha, beta, gamma, initial_proportions=None):
    """
    Simulate the SIR model in terms of proportions.

    Parameters:
    N: total population (not used directly in proportions calculation)
    T: number of time steps
    alpha: probability of S -> I transition
    beta: probability of I -> R transition
    gamma: probability of R -> S transition
    initial_proportions: initial proportions of S, I, R (optional)
    
    Returns:
    Proportions of S, I, R over time.
    """
    
    # Initialize proportions
    if initial_proportions is None:
        a = 0.99  # Initial proportion of Susceptible
        b = 0.01  # Initial proportion of Infected
        c = 0.0   # Initial proportion of Recovered
    else:
        a, b, c = initial_proportions
    
    # Arrays to store proportions over time
    S_proportions = np.zeros(T)
    I_proportions = np.zeros(T)
    R_proportions = np.zeros(T)
    
    # Set initial proportions
    S_proportions[0] = a
    I_proportions[0] = b
    R_proportions[0] = c
    
    # Simulate over T time steps
    for t in range(1, T):
        a = gamma * R_proportions[t-1] + (1 - alpha) * S_proportions[t-1]
        b = alpha * S_proportions[t-1] + (1 - beta) * I_proportions[t-1]
        c = beta * I_proportions[t-1] + (1 - gamma) * R_proportions[t-1]
        
        S_proportions[t] = a
        I_proportions[t] = b
        R_proportions[t] = c

    return S_proportions, I_proportions, R_proportions

# Set parameters
N = 1000  # Total population (not directly used)
T = 500   # Number of time steps
alpha = 0.05  # Infection probability
beta = 0.01   # Recovery probability
gamma = 0.001 # Immunity loss probability

# Run simulation
S, I, R = simulate_sir_proportions(N, T, alpha, beta, gamma)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible', color='blue')
plt.plot(I, label='Infected', color='red')
plt.plot(R, label='Recovered', color='green')
plt.xlabel('Time Steps')
plt.ylabel('Proportion of Population')
plt.title(f'Probabilistic SIR Model (alpha={alpha}, Beta={beta}, gamma={gamma})')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

