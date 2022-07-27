"""
Author: Samuel M B Winter

Curve Fitting to obtain resistance of a superconducting joint, and the settling time constant.
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Function Definitions
def Ramping(x, a, y0):
    """
    A linear function to model the ramping curve
    """
    return a * x + y0

def Exponential(x, A, k):
    """
    A standard exponential function of the form:
        A exp(kx)
    """
    return A * np.exp(k * x)

def SecondOrderDecay(x, A, k):
    """
    Solution to dV/dx = kV^{2}, where t_0 = 0, and V(0) = A
    """
    return A / (1 - A * k * x)

def Biexponential(x, A, k1, B, k2):
    """
    The sum of two exponentials:
        A exp(k_{1} x) + B exp(k_{2} x)
    The settling is modelled as an exponenital decay, alongside the decay of the supercurrent.
    """
    return Exponential(x, A, k1) + Exponential(x, B, k2)

def Triexponential(x, A, k1, B, k2, C, k3):
    return Exponential(x, A, k1) + Exponential(x, B, k2) + Exponential(x, C, k3)

def CombinationDecay(x, A, k1, B, k2):
    """
    the sum of first and senind order decay.
    """
    return SecondOrderDecay(x, A, k1) + Exponential(x, B, k2)


def FullDataCurve(t, t0, a, t1, b, t2, A, k1, B, k2):
    """
    The full decay curve, including ramping.
    """
    V0 = Ramping(t1 - t0, a, 0)
    return (
        (t < t0)                * 0 +
        ((t >= t0) & (t < t1))  * Ramping(t-t0, a, 0) + 
        ((t >= t1) & (t < t2))  * Ramping(t-t1, b, V0) + 
        (t >= t2)               * Biexponential(t-t2, A, k1, B, k2)
    )


def main():

    df = pd.read_csv("data/Measurement 23 (2212 Gen2 v1).txt",sep='\t',header=(0))

    # defing the time and hall voltage arrays
    ts = df["Steps"][1475:124269]#[1271:144895]
    ts = (ts - np.min(ts))# / (60**2 *24)            # Sets time to start at zero
    Vs = df["Hall Voltage"][1475:124269] #[1271:144895]

    # Inital guesses, biexponential fits are sensitive, so resonable guesses are required. 
    inits = (0.06, -10**(-4), 2.5, -10**(-1), 0.5, -10**(-2))
    popt, pcov = curve_fit(Triexponential, ts, Vs, p0=inits, maxfev=10000)

    # Printing the Data: TODO: Formatting - use a conditional to seperate the pairs.
    print(popt)

    # Plotting the data
    plt.plot(ts, Vs, label="Data")
    plt.plot(ts, Triexponential(ts, *popt), label="Fit")
    plt.xlabel("Time / days")
    plt.ylabel("Hall Voltage / mV")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
