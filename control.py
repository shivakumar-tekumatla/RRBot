import scipy.integrate as integrate
import numpy as np
from numpy import sin, cos, pi, zeros_like

import matplotlib.pyplot as plt 

pi = np.pi 


def dynamics(state,t):
    #Initialize the robot 
    m1 = 1
    m2 = 1
    l1 = 1
    l2 = 1
    r1 = 0.45
    r2 = 0.45
    I1 = 0.084
    I2 = 0.084
    g  = 9.81
    #LQR gains - these gains will take the robot in upward direction 
    K  = np.array([[19.6705,13.5741,4.5331,7.2672],
                    [4.4649,7.2775,1.2178,2.3712]])  
    #K is lqr gains 
    dxdt = zeros_like(state)
    theta1 , theta2 , theta1_dot , theta2_dot = state 

    #wrap the angles 
    theta1 =  (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 =  (theta2 + np.pi) % (2 * np.pi) - np.pi 

    U =  -K @np.array([[theta1],
                        [theta2],
                        [theta1_dot],
                        [theta2_dot]])

    u1,u2 = U.flatten() # these are the inputs to the robot 
    dxdt[0] = theta1_dot; 
    dxdt[1] = theta2_dot; 
    dxdt[2] = (I2*u1 - I2*u2 + m2*(r2**2)*u1 - m2*r2**2*u2 + l1*m2**2*r2**3*theta1_dot**2*sin(theta2) + l1*m2**2*r2**3*theta2_dot**2*sin(theta2) + g*l1*m2**2*r2**2*sin(theta1) + I2*g*l1*m2*sin(theta1) + I2*g*m1*r1*sin(theta1) - l1*m2*r2*u2*cos(theta2) + 2*l1*m2**2*r2**3*theta1_dot*theta2_dot*sin(theta2) + l1**2*m2**2*r2**2*theta1_dot**2*cos(theta2)*sin(theta2) - g*l1*m2**2*r2**2*sin(theta1 + theta2)*cos(theta2) + I2*l1*m2*r2*theta1_dot**2*sin(theta2) + I2*l1*m2*r2*theta2_dot**2*sin(theta2) + g*m1*m2*r1*r2**2*sin(theta1) + 2*I2*l1*m2*r2*theta1_dot*theta2_dot*sin(theta2))/(- l1**2*m2**2*r2**2*cos(theta2)**2 + l1**2*m2**2*r2**2 + I2*l1**2*m2 + m1*m2*r1**2*r2**2 + I1*m2*r2**2 + I2*m1*r1**2 + I1*I2)
    dxdt[3] = -(I2*u1 - I1*u2 - I2*u2 - l1**2*m2*u2 - m1*r1**2*u2 + m2*r2**2*u1 - m2*r2**2*u2 + l1*m2**2*r2**3*theta1_dot**2*sin(theta2) + l1**3*m2**2*r2*theta1_dot**2*sin(theta2) + l1*m2**2*r2**3*theta2_dot**2*sin(theta2) - g*l1**2*m2**2*r2*sin(theta1 + theta2) - I1*g*m2*r2*sin(theta1 + theta2) + g*l1*m2**2*r2**2*sin(theta1) + I2*g*l1*m2*sin(theta1) + I2*g*m1*r1*sin(theta1) + l1*m2*r2*u1*cos(theta2) - 2*l1*m2*r2*u2*cos(theta2) + 2*l1*m2**2*r2**3*theta1_dot*theta2_dot*sin(theta2) + 2*l1**2*m2**2*r2**2*theta1_dot**2*cos(theta2)*sin(theta2) + l1**2*m2**2*r2**2*theta2_dot**2*cos(theta2)*sin(theta2) - g*l1*m2**2*r2**2*sin(theta1 + theta2)*cos(theta2) + g*l1**2*m2**2*r2*cos(theta2)*sin(theta1) - g*m1*m2*r1**2*r2*sin(theta1 + theta2) + I1*l1*m2*r2*theta1_dot**2*sin(theta2) + I2*l1*m2*r2*theta1_dot**2*sin(theta2) + I2*l1*m2*r2*theta2_dot**2*sin(theta2) + g*m1*m2*r1*r2**2*sin(theta1) + 2*l1**2*m2**2*r2**2*theta1_dot*theta2_dot*cos(theta2)*sin(theta2) + l1*m1*m2*r1**2*r2*theta1_dot**2*sin(theta2) + 2*I2*l1*m2*r2*theta1_dot*theta2_dot*sin(theta2) + g*l1*m1*m2*r1*r2*cos(theta2)*sin(theta1))/(- l1**2*m2**2*r2**2*cos(theta2)**2 + l1**2*m2**2*r2**2 + I2*l1**2*m2 + m1*m2*r1**2*r2**2 + I1*m2*r2**2 + I2*m1*r1**2 + I1*I2);

    return dxdt

def main():

    K  = np.array([[19.6705,13.5741,4.5331,7.2672],
                    [4.4649,7.2775,1.2178,2.3712]])  

    
    dt_pbot = 0.010
    t = np.arange( 0.0, 10., dt_pbot )
    state = np.array( [ pi/4., 0., 0., 0. ] ) #initial state 

    states = integrate.odeint( dynamics, state, t )
    inputs = []
    for state in states:
        theta1 , theta2 , theta1_dot , theta2_dot = state 
        U =  -K @np.array([[theta1],
                        [theta2],
                        [theta1_dot],
                        [theta2_dot]])
        u1,u2 = U.flatten()
        inputs.append([u1,u2])

    print(states)

    plt.plot(t,states)
    plt.legend(["Theta-1","Theta-2","Theta-1-dot","Theta-2-dot"])
    plt.show()
    plt.plot(t,inputs)
    plt.legend(["u-1","u-2"])
    plt.show()



if __name__ == "__main__":
    main()


