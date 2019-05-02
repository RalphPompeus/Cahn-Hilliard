# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:30:18 2019

@author: Ralph Pompeus
"""
import numpy as np
import sys
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from CH_class import PDE


#Define time and space increments.
dt, dx = 2,1

#Create instance of PDE class and add random noise.
#PDE input arguments: dt, dx, dimension, phi
dimension = int(input("Please enter a lattice dimension: "))
#dimension = dimension.lower()

phi = float(input("Please enter a value for phi: "))
#phi = phi.lower()


task = input("Please enter task: viz or data: ")
task = task.lower()



A = PDE(dt,dx,dimension,phi)
A.Noise()

#If user requires visual simulation
if task=='viz':
    
    #Update function

    def UpdatePlot(*args):
        image = ax.imshow(A.order_array)
        for i in range(50):
            A.Sweep()
        return image,
    
    #Create animation
    fig,ax = plt.subplots()
    image = ax.imshow(A.lattice)
    ani = FuncAnimation(fig,UpdatePlot,blit=True)
    plt.show()

#If user requires free energy data
elif task=='data':
    
    #Lists for plotting
    FE_list = []
    time_list = []
    
    #Simulates the system for a number of iterations
    for i in range(100000):
        A.Sweep()
        if i>=500 and i%500==0:
            #Measures free energy at regular intervals
            FE_list.append(A.Free_Energy())
            time_list.append(i)
            print(i)

    #Plots free energy vs timestep
    plt.plot(time_list,FE_list)
    plt.xlabel("Timestep")
    plt.ylabel("Free Energy")
    plt.show()
