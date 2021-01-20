#Carlos J. Buxo Vazquez
#CMSE821 Homework 3
#March 1, 2019
#
#
#This program implements a number of methods to solve two differential equations numerically.
#
#The first problem is solving the steady state heat equation u''(x)=f(x) in one dimension subject to
#the Neumann boundary conditions u'(0)=u(1)=0 for which the exact solution is given by u(x)=(x(1-x))^2.01.
#For this problem we implement three types of finite difference methods: one in which we approximate
#the boundary condition u'(0)=0 using a first order approximation D_+u(0), one where we approximate u'(0)
#using the method of ghost points, and one method in which we approximate u'(0) using a second order approximaiton
#of the derivative using the method of undetermined coefficients.
#
#The second problem is solving Poisson's equation in a unit square domain subject to the boundary conditions
#u(x,0)=u(x,1)= 0 for 0 <= x <= 1 and u(0,y)=u(1,y)=0 for 0 <= y <= 1 with the exact solution given by
#u(x,y)=sin(pi*x)sin(y*pi). For this differential equation we implement four different iterative methods:
#Jordan iterative method, Gauss-Seidel iterative method, Successive-Overrelaxation (SOR) method, and Steepest Descent method

from scipy import special
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

############################################################################################################
#                   FUNCTIONS FOR SOLVING THE STEADY STATE HEAT EQUATION                                   #
############################################################################################################


#Function that calculates the second derivative of the exact solution u(x)=[x(1-x)]^2.01
#Arguments: point where the second derivative will be calculated x
def powerFunc(x):
    return (-2*2.01*((x*(1-x))**1.01))+(1.01*2.01*((1-2*x)**2)*((x*(1-x))**0.01))

#Function that calculates the exact solution u(x)=[x(1-x)]^2.01
#Arguments: point where the function will be calculated x
def powerExactSol(x):
    return (x*(1-x))**2.01

#Function that calculates the coefficient matrix A in the finite difference
#method (1/h^2)AU=F.
#Arguments: Number of rows and columns m, string that determines if using Ghost Points or Undetermined Coefficients method
def coeffMatrix(m, method):
    #Initialize matrix as the zero mxm matrix
    A = np.zeros((m+2,m+2))
    
    h = 1.0/(m+1)
    
    A[0][0] = 1.0/h
    A[0][1] = -A[0][0]
    
    #If using Undetermined Coefficients, initialize the first row accordingly
    if(method == "UC"):
        A[0][0] = (3.0/2.0)*(1.0/h)
        A[0][1] = -2.0*(1.0/h)
        A[0][2] = 0.5*(1.0/h)
    
    A[1][0] = 1.0/(h**2)
    A[m][m+1] = A[1][0]
    A[m+1][m+1] = 1.0
    
    #loop through rows of A
    for i in range(1,m+1):
        #loop through columns of A
        for j in range(1,m+1):
            #create the tridiagonal shape of A
            if(i == j-1 or i == j+1):
                A[i][j] = 1.0/(h**2)
            elif(i == j):
                A[i][j] = -2.0/(h**2)
    return A

#Function that calculates the function vector F in the finite difference
#method (1/h^2)AU=F.
#Arguments: Length of the vector m, function f to evaluate entries of F, string that determines if using Ghost Points or Undetermined Coefficients method
def functionVector(m,f,method):
    #Define equidistant spacing between grid points
    h = 1.0/(1.0+m)
    
    F = np.zeros(m+2)
    
    #If using Ghost Points, set the appropriate value of the first entry of the vector
    if(method == "GP"):
        F[0] = (h/2.0)*f(0)
    
    for i in range(0, m):
        F[i+1] = f(h*(i+1))
    
    return F

#Function that implements the Gauss-Jordan elimination method to solve matrix equations.
#This function is used to solve the finite difference method AU=(h^2)F
#Arguments: Coefficient matrix A, function vector F
def gaussJordanElim(A,F):
    #loop through the rows of A
    for i in range(0,len(A)):
        
        #Set diagonal entries of A equal to 1 by dividing ith row of A
        #by A[i][i]. Also divide ith entry of F by A[i][i]
        if(A[i][i] != 1):
            F[i] /= A[i][i]
            A[i, :] /= A[i][i]
        
        #loop through columns of A
        for j in range(0,len(A)):
            #If the other jth entries on the same column as A[i][i] are non-zero
            #then substract them. Also substract the jth entry of F by A[j][i]F[i]
            #in accordance to Gauss-Jordan elimination
            if(j == i or A[j][i] == 0):
                continue
            F[j] -= A[j][i]*F[i]
            A[j, :] -= A[j][i]*A[i, :]

    #Returns U=(h^2)(A^-1)F
    return F

#Function that solves the finite difference method for a given function f, calculates the global error
#of the method, and plots the different norms as functions of the step size h
#Arguments: function f to evaluate function vector F in the finite difference method (1/h^2)AU=F, exact solution of the heat equation sol,
#string that determines if using Ghost Points or Undetermined Coefficients method
def printAndPlotErrors(f, sol, method):
    #Array containing different values for the size of the matrix A
    m_values = [9, 19, 99, 199, 999]
    
    #Array to contain the different global error norms for different values of h
    E_inf=[]
    E_l1 = []
    E_l2 = []
    
    for m in m_values:
        #Solve the finite difference method with h = 1.0/(1.0+m)
        A = coeffMatrix(m, method)
        F = functionVector(m,f, method)
        U = gaussJordanElim(A,F)
        
        #Array to store the different entries of the error vector
        E = []
        
        #Calculating the error vector entries
        for i in range(0,m+2):
            E.append(abs(U[i]-sol(i/(1.0+m))))
    
        #Calculating the different global error vector norms
        E_inf.append(max(E))
        
        e_l1 = 0
        e_l2 = 0
        for i in range(0,m+2):
            e_l1 += (1.0/(1.0+m))*E[i]
            e_l2 += (1.0/(1.0+m))*(E[i]**2)

        E_l1.append(e_l1)
        E_l2.append(np.sqrt(e_l2))

    h_values = [0.1, 0.05, 0.01, 0.005, 0.001]

    #Printing the values of the global error vector norms and the l2 norm for the coefficient matrix for a given value of h
    for i in range(0,5):
        print('h=', h_values[i], ', ||E||_inf=', E_inf[i], ', ||E||_l1=', E_l1[i], ', ||E||_l2=', E_l2[i])


    print('Slopes')
    for i in range(4, 0, -1):
        print('log(E_inf(',h_values[i-1],'))-log(E_inf(',h_values[i],')/(log(',h_values[i-1], ')-log(', h_values[i], '))=' , (np.log(E_inf[i-1])-np.log(E_inf[i]))/(np.log(h_values[i-1])-np.log(h_values[i])))
    
        print('log(E_l1(',h_values[i-1],'))-log(E_l1(',h_values[i],')/(log(',h_values[i-1], ')-log(', h_values[i], '))=' , (np.log(E_l1[i-1])-np.log(E_l1[i]))/(np.log(h_values[i-1])-np.log(h_values[i])))
    
        print('log(E_l2(',h_values[i-1],'))-log(E_l2(',h_values[i],')/(log(',h_values[i-1], ')-log(', h_values[i], '))=' , (np.log(E_l2[i-1])-np.log(E_l2[i]))/(np.log(h_values[i-1])-np.log(h_values[i])))

    
    #Plotting the global error vector norms
    plt.plot(h_values, E_inf, 'r+-', label='||E||_inf')
    plt.plot(h_values, E_l1, 'b+-', label='||E||_l1')
    plt.plot(h_values, E_l2, 'g+-', label='||E||_l2')
    plt.legend(loc = 'best')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('h')
    plt.ylabel('||E(h)||')
    plt.show()

############################################################################################################
#                   FUNCTIONS FOR SOLVING POISSON'S EQUATION IN UNIT SQUARE                                #
############################################################################################################


#Function that calculates the Laplacian of the exact solution u(x,y)=sin(pi*x)sin(pi*y)
#Arguments: points where the Laplacian will be calculated x, y
def sineExactSol(x,y):
    return -2.0*(np.pi**2)*(np.sin(x*np.pi)*np.sin(y*np.pi))

#Function that calculates the initial guess solution that will be used in the iterative methods
#Arguments: points where the guess solution will be calculated x, y
def guessSolution(x,y):
    #return np.sin(x*np.pi)*np.sin(y*np.pi)
    return 0.0


#Function that implements the Jacobi, Gauss-Seidel, SOR, and Steepest Descent methods used for solving
#Poisson's equation in a unit square domain subject to the boundary conditions u(x,0)=u(x,1)=0 for 0 <= x <= 1
#and u(0,y)=u(1,y) for 0 <= y <= 1 that will be implemented on an uniform grid of size (m+1)^2
#Arguments: number of iterations that will be performed itr, size of grid m, function that contains initial guess solution u,
#function that contains Laplacian of exact solution f, string that determines which iterative method to use method
def iterativeMethods(itr, m, u, f, method):

    #Defining the grid point separation
    h = 1.0/(1.0+m)

    #Generating coordinates for the uniform grid
    x = np.arange(0.0,1.0,h)
    y = np.arange(0.0,1.0,h)

    #Initializing matrix that will contain the numerical solution to Poisson's equation on the uniform grid
    #The shape of the matrix is designed to hold an (m^2)*1 column vector as a matrix such that usol[,:j] = [u(x[0],y[j]),...,u(x[m],y[j])]
    usol = np.zeros((m+1,m+1))
    #Initializing matrix that will contain the values of the Laplacian on the uniform grid
    #The shape of the matrix is designed to hold an (m^2)*1 column vector as a matrix such that fvec[,:j] = [f(x[0].y[j]),...,f(x[0],y[j])]
    fvec = np.zeros((m+1,m+1))
    
    #Loop that stores the values of the guess solution and Laplacian of the exact solution
    #in the matrices usol and fvec respectively
    for i in range(0,m+1):
        for j in range(0,m+1):
            usol[i][j] = u(x[i],y[j])
            fvec[i][j] = f(x[i],y[j])

    #Initializing array to store the l_2 error norm after each iteration
    Error =[]

    #Start if method is Jacobi method
    if(method == "J"):
        
        #Start loop to iterate numerical solution
        for k in range(0, itr):
            
            #Initialize value of l_2 error norm for the kth iteration
            e_l2 = 0
            
            #Initialize matrix that will contain the numerical solution in the kth iteration
            unew = np.zeros((m+1,m+1))
            
            #Start loop to move across the m+1*m+1 grid
            #We loop between the values 0 and m since we don't want to change the boundary condition values
            for i in range(1, m):
                for j in range(1, m):
                    
                    #Calculate the numerical solution at u(x[i],j[i]) for the kth iteration in accordance to the Jacobi iterative method
                    unew[i][j] = 0.25*( usol[i-1][j] + usol[i+1][j] + usol[i][j-1] + usol[i][j+1] - (h**2)*fvec[i][j] )
                    
                    #Actualize the l_2 error norm
                    e_l2 += h*((unew[i][j] - (np.sin(np.pi*x[i])*np.sin(np.pi*y[j])) )**2)
        
            #Actualize usol after the kth iteration
            usol = unew
            
            #Store the l_2 error norm for the kth iteration
            Error.append(np.sqrt(e_l2))
    #End if method is Jacobi method
    #Start if method is Successive-Overrelaxation method or Gauss-Seidel method
    elif(method == "SOR" or method == "GS"):
        
        #Initialize value for omega used in SOR
        omega = 1
        
        #If using SOR, use optimal omega
        if(method == "SOR"):
            print("Using SOR")
            omega = 2.0/(1.0+np.sin(h*np.pi))

        #Start loop to iterate numerical solution
        for k in range(0, itr):
            
            #Initialize value of l_2 error norm for the kth iteration
            e_l2 = 0
            
            #Start loop to move across the m+1*m+1 grid
            #We loop between the values 0 and m since we don't want to change the boundary condition values
            #Note that the value of u(x[i],y[j]) in the kth iteration has the general form of the SOR method
            #but if we are using the Gauss-Seidel method since omega = 1 the expression reduces to the one used
            #for this method
            for i in range(1, m):
                for j in range(1, m):
                    
                    #Actualize the numerical solution at u(x[i],j[i]) for the kth iteration in accordance to the SOR or Gauss-Seidel iterative method
                    usol[i][j] = 0.25*omega*( usol[i-1][j] + usol[i+1][j] + usol[i][j-1] + usol[i][j+1] - (h**2)*fvec[i][j] ) + (1-omega)*usol[i][j]
                    
                    #Actualize the l_2 error norm
                    e_l2 += h*((usol[i][j] - (np.sin(np.pi*x[i])*np.sin(np.pi*y[j])) )**2)

            #Store the l_2 error norm for the kth iteration
            Error.append(np.sqrt(e_l2))
    #End if method is Successive-Overrelaxation method or Gauss-Seidel method
    #Start if method is Steepest Descent method
    elif(method == "SD"):
        
        #Initialize the (m+1)*(m+1) coefficient matrix needed to actualize the numerical solution and the residue vector after each iteration
        T = generateCoeffMatrix(m)
    
        #Initialize matrix that contains the residue vector
        r_0 = np.zeros((m+1,m+1))
        
        omega_k = np.zeros((m+1,m+1))
    
        #Calculate the initial value of the residue vector given by r = f - Au where A is the (m+2^2)*(m+2^2) matrix with shape
        #
        #            [T I 0 0 . . .   0]
        #            [I T I 0 . . .   0]
        #            [0 I T I         0]
        # A =(1/h^2) [.       .       .]
        #            [.         .     .]
        #            [.          .    0]
        #            [.           I T I]
        #            [0   . . .   0 I T]
        #
        #where I and 0 are the (m+1)*(m+1) identity and zero matrices, and T is the (m+1)*(m+1) coefficient matrix generated above
        
        r_0[:,0] = fvec[:,0] - (1.0/(h**2))*((T@usol[:,0]) + usol[:,1])
        r_0[:,m] = fvec[:,m] - (1.0/(h**2))*(usol[:,m-1] + (T@usol[:,m]))
        for j in range(1,m):
            r_0[:,j] = fvec[:,j] - (1.0/(h**2))*(usol[:,j] + (T@usol[:,j]) + usol[:,j])

        #Start loop to iterate numerical solution
        for k in range(0, itr):
            
            #Calculate first and last entry of A*r_0 to iterate residue vector
            omega_k[:,0] = (1.0/(h**2))*((T@r_0[:,0]) + r_0[:,1])
            omega_k[:,m] = (1.0/(h**2))*(r_0[:,m-1] + (T@r_0[:,m]))
            
            #Initialize values of numerator and denominator for iteration constant alpha = r^T*r/(r^T*A*r)
            alpha_num = r_0[:,0]@r_0[:,0] + r_0[:,m]@r_0[:,m]
            alpha_den = r_0[:,0]@omega_k[:,0] + r_0[:,m]@omega_k[:,m]
            
            #Calculate other entries of residue vector and actualize iteration constant alpha
            for j in range(1, m):
                omega_k[:,j] = (1.0/(h**2))*(r_0[:,j] + (T@r_0[:,j]) + r_0[:,j])
                alpha_num += r_0[:,j]@r_0[:,j]
                alpha_den += r_0[:,j]@omega_k[:,j]

            alpha_k = alpha_num/alpha_den

            #Actualize numerical solution
            usol = usol + alpha_k*r_0
            
            #Actualize residue vector
            r_0 = r_0 - alpha_k*omega_k

            #Initialize l_2 error norm
            e_l2 = 0
            
            #Calculate l_2 error norm
            for i in range(1, m):
                for j in range(1, m):
                    e_l2 += h*((usol[i][j] - (np.sin(np.pi*x[i])*np.sin(np.pi*y[j])) )**2)
            Error.append(np.sqrt(e_l2))
    #End if method is Steepest Descent

                
    k_values = np.linspace(1, itr, itr, endpoint = True)
    plt.plot(k_values, Error, '+', label='||E||_l2')
    plt.legend(loc = 'best')
    plt.xlabel('k')
    plt.ylabel('||E||_l2')
    plt.show()

    x, y = np.meshgrid(x,y)
                        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,usol,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    plt.show()
                        
    #print(usol)
    return usol

#Function that generates the coefficient matrix used to iterate the numerical solution and residue vector for the
#Steepest Descent iterative method
#Arguments: Size of the matrix m
def generateCoeffMatrix(m):
    A = np.zeros((m+1,m+1))
    for i in range(0, m+1):
        for j in range(0, m+1):
            if(i == j):
                A[i][j]=-4
            elif(i == j-1 or i == j+1):
                A[i][j] = 1
    return A



printAndPlotErrors(powerFunc, powerExactSol, "UC")
printAndPlotErrors(powerFunc, powerExactSol, "GP")
printAndPlotErrors(powerFunc, powerExactSol, "SO")

iterativeMethods(10, 999, guessSolution, sineExactSol, "J")
iterativeMethods(10, 999, guessSolution, sineExactSol, "GS")
iterativeMethods(10, 999, guessSolution, sineExactSol, "SOR")
iterativeMethods(10, 999, guessSolution, sineExactSol, "SD")

