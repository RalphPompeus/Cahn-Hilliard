import numpy as np
import matplotlib.pylab as plt
import random


#PDE class containing methods to solve the Cahn Hilliard equation using the Euler algorithm based on finite difference methods
class PDE(object):
    
    #Init method. Takes time and space intervals, the dimension of the system and initial value of phi
    def __init__(self,dt,dx,N,phi):
        self.dt, self.dx = dt, dx
        self.N = N
        self.lattice = np.full((self.N,self.N),phi)
        self.chemPotential = np.zeros((self.N,self.N))
        self.a, self.k, self.M = float(0.1),float(0.1),float(0.1)
    
    
    
###############################################################################
######################## Model Definition and Update rules ####################
###############################################################################
    
    #Instance method which imposes periodic boundary conditions on a specific index
    def PBC(self,index):
        return index%(self.N)
    
    
    
    #Adds a different random amount of noise to each index of the order array
    def Noise(self):
        for i in range(self.N):
            for j in range(self.N):
                self.lattice[i,j] = self.lattice[i,j] + random.uniform(-0.1,0.1)



    #Method to calculate the value of a given index of the chemical potential array according to the order array
    def Calc_Chem_Index(self,i,j):
        self.chemPotential[i,j] = -self.a*(self.lattice[i,j]) + self.a*(self.lattice[i,j])**3. 
        - (self.k/self.dx**2.)*(self.lattice[self.PBC(i+1),j]+self.lattice[self.PBC(i-1),j]
        +self.lattice[i,self.PBC(j+1)]+self.lattice[i,self.PBC(j-1)]-4*self.lattice[i,j])



    #Method to calculate the chemical potential of the whole array
    def Set_Chem_Array(self):
        for i in range(self.N):
            for j in range(self.N):
                self.Calc_Chem_Index(i,j)



    #Euler algorithm to update the order parameter array
    def Update_Order_Array(self):
        const = (self.dt*self.M)/(self.dx**2.)
        for i in range(self.N):
            for j in range(self.N):
                self.lattice[i,j] = self.lattice[i,j] + const*(self.chemPotential[self.PBC(i+1),j]+self.chemPotential[self.PBC(i-1),j]
                +self.chemPotential[i,self.PBC(j+1)]+self.chemPotential[i,self.PBC(j-1)] - 4*self.chemPotential[i,j])



    #Sweep method which calculates the chemical potential of the whole array and uses it to update the order parameter array
    def Sweep(self):
        self.Set_Chem_Array()
        self.Update_Order_Array()


###############################################################################
############################### Calculation Parts #############################
###############################################################################


    #Method to calculate the 2D Laplacian of an array index using finite difference methods. Returns a list with the vector components
    def Laplacian_2D(self,array,i,j):
        wrtx = (array[self.PBC(i+1),j] + array[self.PBC(i-1),j] -2*array[i,j])/(self.dx**2)
        wrty = (array[i,self.PBC(j+1)] + array[i,self.PBC(j-1)] -2*array[i,j])/(self.dx**2)
        return [wrtx,wrty]



    #Method to calculate the square of the grad of an array index
    def Grad_Sq(self,array,i,j):
        wrtx = (array[self.PBC(i+1),j] - array[self.PBC(i-1),j])/(2*self.dx)
        wrty = (array[i,self.PBC(j+1)] - array[i,self.PBC(j-1)])/(2*self.dx)
        dot_prod = PDE.Dot_Product([wrtx,wrty],[wrtx,wrty])
        return dot_prod



    #Algorithm to compute the free energy of a certain index
    def free_density(self,i,j):
        f = (self.a/-2)*(self.lattice[i,j])**2 + (self.a/4)*(self.lattice[i,j])**4 + (self.k/2)*(self.Grad_Sq(self.lattice,i,j))
        return f


    #Method to calculate the free energy of the entire lattice
    def Free_Energy(self):
        flist = []
        for i in range(self.N):
            for j in range(self.N):
                flist.append(self.free_density(i,j))
        return sum(flist)



    #Method to return the dot product of two lists
    def Dot_Product(list1,list2):
        dotlist = []
        for i in range(len(list1)):
            dotlist.append(list1[i]*list2[i])
            return sum(dotlist)
