#Carlos J. Buxo Vazquez
#CMSE821 Final Project - Problem 3
#May 4, 2019
#
#
#This program implements several numerical finite difference methods to solve the Stokes-Navier
#equation for an incompressible fluid contained in a two dimensional square container with sides
#of length 1m. The container has a lid on top that's moving horizontally from left to right with
#velocity U=1m/s. The program solves for the vorticity and the stream-function of the fluid and
#plots these quantities in addition to the fluid's vertical and horizontal velocity components, and
#the streamlines of the fluid.

from scipy import special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

################################################################################
#                                                                              #
#  Functions for solving the non-linear portion of the Stokes-Navier Equation  #
#                                                                              #
################################################################################


#Function that generates the diagonal elements of the matrix that
#represents the non-linear operator in the finite difference scheme
#for solving the Stokes-Navier equation.
#Arguments: Number of interior grid points: m, vector of horizontal velocity components at fixed j: u_j
#constants that are multiplied by the matrix: weight, boolean to determine if we want nonzero diagonal entries: diagonal
def GenerateDiagonalNonLinear(m, u_j, weight, diagonal):

    #Generate a (m+2*m+2) placeholder matrix
    A=np.zeros((m+2,m+2))
    
    #Start loop through the rows of the matrix
    for i in range(1,m+1):
        #Set the values of the subdiagonal entries
        A[i][i+1] = -weight*u_j[i]
        A[i][i-1] = -A[i][i+1]
    
    #Start if we want nonzero diagonal entries
    if(diagonal == True):
        A = A + np.identity(m+2)
        
        #Setting the first and last entries of the diagonal to zero in order
        #to not change the boundary values of the vorticity
        A[0][0]=0
        A[m+1][m+1] = 0

    return A


#Function that generates the offdiagonal elements of the matrix that
#represents the non-linear operator in the finite difference scheme
#for solving the Stokes-Navier equation.
#Arguments: Number of interior grid points: m, vector of vertical velocity components at fixed j: v_j
#constants that are multiplied by the matrix: weight
def GenerateOffDiagonalNonLinear(m, v_j, weight):
    
    #Generate a (m+2*m+2) placeholder matrix
    A = np.zeros((m+2,m+2))

    #Start loop through the rows of the matrix
    for i in range(1,m+1):
        #Set the values along the diagonal. The first and last entries of the diagonal are zero
        #in order to not change the boundary values of the vorticity
        A[i][i] = -weight*v_j[i]

    return A


#Function that applies the Forward-Euler method to solve for the vorticity at a time step n+1 from the non-linear component
#of the Stokes-Navier equation. The function is used only when generating the values of omega(t_1=k).
#Arguments: Time step: k, Number of interior grid points: m, vector containing vorticity at previous time step n: omega_n,
#vector of horizontal velocity components at time step n: u_n, vector of vertical velocity componenets at time step n: v_n
def ForwardEuler(k, m, omega_n, u_n, v_n):

    #Defining the grid spacing
    h = 1.0/(1.0+m)
    
    #Placeholder for the vorticity at time step n+1
    omega_new = np.zeros((m+2,m+2))
    
    #Constant that gets multiplied by the matrices when using Forward-Euler
    weight_a = (k/(2.0*h))

    #Start loop through componetns of omega_n with fixed j
    for j in range(1, m+1):
        
        #Generating the matrices that will be used to calculate the jth component of omega_new
        A_D = GenerateDiagonalNonLinear(m, u_n[j], weight_a, True)
        A_O = GenerateOffDiagonalNonLinear(m, v_n[j], weight_a)

        #Calculate the vorticity at time step n+1, omega(i*h,j*h,(n+1)*k) for fixed j
        omega_j = (-A_O@omega_n[j-1] + A_D@omega_n[j] + A_O@omega_n[j+1])

        omega_new[j]=omega_j

    return omega_new


#Function that applies the Adams-Bashforth two step method to solve for the vorticity at a time step n+1
#from the non-linear component of the Stokes-Navier equation.
#Arguments: Time step: k, Number of interior grid points: m, vector containing vorticity at time step n: omega_n,
#vector containing vorticity at time step n-2: omega_n_previous, vector of horizontal velocity components at time step n: u_n,
#vector of vertical velocity componenets at time step n: v_n, vector of horizontal velocity components at time step n-2: u_n_previous,
#vector of vertical velocity components at time step n-2: v_n_previous
def vorticityNonLinear(k, m, omega_n, omega_n_previous, u_n, v_n, u_n_previous, v_n_previous):
    
    #Defining the grid spacing
    h = 1.0/(1.0+m)
    
    #Placeholder for the vorticity at time step n+1
    omega_new = np.zeros((m+2,m+2))
    
    #Constants that are multiplied by the matrices when using Adams-Bashforth 2-step
    weight_a = (3.0*k/(4.0*h))
    weight_b = -(k/(4.0*h))
    
    #Start loop through components of omega_n with fixed j
    for j in range(1, m+1):
    
        #Generating the matrices that will be used to calculate the jth component of omega_new
        A_D = GenerateDiagonalNonLinear(m, u_n[j], weight_a, True)
        A_O = GenerateOffDiagonalNonLinear(m, v_n[j], weight_a)
        
        B_D = GenerateDiagonalNonLinear( m, u_n_previous[j], weight_b, False)
        B_O = GenerateOffDiagonalNonLinear(m, v_n_previous[j], weight_b)
        
        #Calculate the vorticity at time step n+1, omega(i*h,j*h,(n+1)*k) for fixed j
        omega_j = (-A_O@omega_n[j-1] + A_D@omega_n[j] + A_O@omega_n[j+1]) +(-B_O@omega_n_previous[j-1] + B_D@omega_n_previous[j] + B_O@omega_n_previous[j+1])
        
        omega_new[j]=omega_j
    

    return omega_new


################################################################################
#                                                                              #
#         Functions for solving the Stream-function Poisson Equation           #
#                                                                              #
################################################################################

#Function that applies the Successive Over Relaxation (SOR) method for solving the Poisson equation of the stream-function at time step n.
#Arguments: Number of interior grid points: m, vector containing vorticity at time step n: omega_n
def PoissonSORSolver(m, omega_n):

    #Defining the grid spacing
    h = 1.0/(1.0+m)
    
    #Defining the scalar parameter for the SOR method
    omega = 2.0/(1.0+np.sin(h*np.pi))
    
    #Placeholder for the stream-function at time step n
    psi = np.zeros((m+2,m+2))
    
    #Defining maximum number of iterations for the SOR method
    itr = int(np.floor(m*np.log(m)))
    
    #Start loop through iteration steps
    for k in range(0, itr):
        
        #Start loop through horizontal values of the stream-function
        for i in range(1, m+1):
            
            #Start loop through vertical values of the stream-function
            for j in range(1, m+1):
                
                #Calculate the stream-function psi(i*h,j*h) at iteration step k
                psi[j][i] = 0.25*omega*( psi[j-1][i] + psi[j+1][i] + psi[j][i-1] + psi[j][i+1] + (h**2)*omega_n[j][i] ) + (1-omega)*psi[j][i]

    return psi


################################################################################
#                                                                              #
#    Functions for solving the linear portion of the Stokes-Navier Equation    #
#                                                                              #
################################################################################


#Function that generates the diagonal elements of the matrix that
#represents the linear operator in the finite difference scheme
#for solving the Stokes-Navier equation.
#Arguments: Number of interior grid points: m, constants that are multiplied by the matrix: weight,
#boolean to determine if we want to add the identity matrix along the diagonal: diagonal
def GenerateDiagonalLinear(m, weight, diagonal):

    #Generate a (m+2*m+2) placeholder matrix
    A = np.zeros((m+2,m+2))
    
    #Start loop through the rows of the matrix
    for i in range (1, m+1):
        #Set values along the diagonal
        A[i][i]=-4.0*weight
        #Set values along the subdiagonals
        A[i][i-1] = weight
        A[i][i+1] = weight
    
    #Start if we want to add the identity matrix along the diagonal
    if(diagonal == True):
        A = A + np.identity(m+2)
        #Set the first and last entries in the diagonal to zero to not modify the boundary values of vorticity
        A[0][0]=0
        A[m+1][m+1] = 0

    return A


#Function that generates the offdiagonal elements of the matrix that
#represents the linear operator in the finite difference scheme
#for solving the Stokes-Navier equation.
#Arguments: Number of interior grid points: m, constants that are multiplied by the matrix: weight
def GenerateOffDiagonalLinear(m, weight):

    #Generate a (m+2*m+2) placeholder matrix
    A = np.zeros((m+2,m+2))
    
    #Start loop through the rows of the matrix
    for i in range(1,m+1):
        A[i][i] = weight
    
    return A


#Function that applies the Trapezoidal method to solve for the vorticity at a time step n+1 from the linear component
#of the Stokes-Navier equation. The function is used only when generating the values of omega(t_1=k).
#Arguments: Time step: k, Number of interior grid points: m, vector containing vorticity at previous time step n: omega_n,
#vector containing the auxiliary vorticity function: omega_star
def Trapezoidal(k, m, omega_n, omega_star):
    
    #Defining the grid spacing
    h = 1.0/(1.0+m)
    
    #Defining the value of the kinetic viscosity
    nu = 0.01

    #Placeholder for the vorticity at time step n+1
    omega_new = np.zeros((m+2,m+2))
    
    #Defining constants that multiply the matrices when using the trapezoidal method
    weight_a = (k*nu)/(2.0*(h**2))
    weight_b = (k*nu)/(2.0*(h**2))
    
    #Generating the matrices that will be used to calculate the jth component of omega_new
    A_D = GenerateDiagonalLinear(m, weight_a, True)
    A_O = GenerateOffDiagonalLinear(m, weight_a)
    
    B_D = GenerateDiagonalLinear(m, weight_b, False)
    B_O = GenerateOffDiagonalLinear(m, weight_b)
    
    #Start loop through components of omega_n with fixed j
    for j in range(1, m+1):
        
        #Calculate the vorticity at time step n+1, omega(i*h,j*h,(n+1)*k) for fixed j
        omega_j = (A_O@omega_n[j-1] + A_D@omega_n[j] + A_O@omega_n[j+1]) + (B_O@omega_star[j-1] + B_D@omega_star[j] + B_O@omega_star[j+1])
        
        omega_new[j]=omega_j

    
    return omega_new


#Function that applies the Adams-Moulton two step method to solve for the vorticity at a time step n+1
#from the linear component of the Stokes-Navier equation.
#Arguments: Time step: k, Number of interior grid points: m, vector containing vorticity at time step n: omega_n,
#vector containing vorticity at time step n-2: omega_n_previous, vector containing the auxiliary vorticity function: omega_star
def vorticityLinear(k, m, omega_n, omega_n_previous, omega_star):

    #DEfining the grid spacing
    h = 1.0/(1.0+m)
    
    #Defining the kinetic viscosity
    nu = 0.01
    
    #Placeholder for the vorticity at time step n+1
    omega_new = np.zeros((m+2,m+2))
    
    #Defining constants that multiply the matrices when using the Adams-Moulton two step method
    weight_a = (2.0*k*nu)/(3.0*(h**2))
    weight_b = -(k*nu)/(12.0*(h**2))
    weight_c = (5.0*k*nu)/(12.0*(h**2))
    
    #Generating matrices that will be used to calculate the jth component of omega_new
    A_D = GenerateDiagonalLinear(m, weight_a, True)
    A_O = GenerateOffDiagonalLinear(m, weight_a)

    B_D = GenerateDiagonalLinear(m, weight_b, False)
    B_O = GenerateOffDiagonalLinear(m, weight_b)
    
    C_D = GenerateDiagonalLinear(m, weight_c, False)
    C_O = GenerateOffDiagonalLinear(m, weight_c)
    
    #Start loop through components of omega_n with fixed j
    for j in range(1, m+1):

        #Calculate the vorticity at time step n+1, omega(i*h,j*h,(n+1)*k) for fixed j
        omega_j = (A_O@omega_n[j-1] + A_D@omega_n[j] + A_O@omega_n[j+1]) + (B_O@omega_n_previous[j-1] + B_D@omega_n_previous[j] + B_O@omega_n_previous[j+1]) + (C_O@omega_star[j-1] + C_D@omega_star[j] + C_O@omega_star[j+1])
        
        omega_new[j]=omega_j

    return omega_new


################################################################################
#                                                                              #
#            Main function for solving the Stokes-Navier Equation              #
#                                                                              #
################################################################################

#Function that solves numerically for the vorticity and stream-function of a fluid using Stokes-Navier equation.
#Argumenets: Time step: k, Number of interior grid points: m
def StokesNavierSolver(k,m):

    #Defining the grid spacing
    h = 1.0/(1.0+m)
    
    #Initializing arrays that will contain quantities for each time step
    
    #Array for storing the horizontal velocity component
    velocity_x =[]
    
    #Array for storing the vertical velocity component
    velocity_y =[]
    
    #Array for storing the vorticity
    omega=[]
    
    #Array for storing the stream-function
    psi =[]
    
    #Initializing the vorticity, stream function and velocity components with the initial conditions
    omega_initial = np.zeros((m+2,m+2))
    psi_initial = np.zeros((m+2,m+2))
    velocity_x_initial = np.zeros((m+2,m+2))
    velocity_y_initial = np.zeros((m+2,m+2))
    
    omega.append(omega_initial)
    psi.append(psi_initial)
    velocity_x.append(velocity_x_initial)
    velocity_y.append(velocity_y_initial)
    
    #Calculating the auxiliary vorticity function by applying the Forward-Euler method to the initial value of vorticity
    omega_star = ForwardEuler(k, m, omega_initial, velocity_x_initial, velocity_y_initial)
    #Calculating the vorticity function at time step t=k by applying the Trapezoidal method to the auxiliary vorticity
    #function and the initial value of the vorticity
    omega_new = Trapezoidal(k,m,omega_initial,omega_star)
    #Calculating the stream-function at time step t=k from the vorticity function at time step t=k by applying SOR
    psi_new = PoissonSORSolver(m,omega_new)
    
    #Updating the vorticity boundary conditions using the stream-function
    omega_new[m+1]=-(2.0/(h**2))*psi_new[m]+(-2.0/h)*np.ones(m+2)
    omega_new[0] = -(2.0/(h**2))*psi_new[1]
    omega_new[:,0] = -(2.0/(h**2))*psi_new[:,1]
    omega_new[:,m+1] = -(2.0/(h**2))*psi_new[:,m]
    
    #Setting the vorticity values at the corners of the grid equal to the average of it's boundary neighbors
    omega_new[m+1][0] = 0.5*(omega_new[m+1][1]+omega_new[m][0])
    omega_new[m+1][m+1] = 0.5*(omega_new[m+1][m]+omega_new[m][m+1])
    omega_new[0][0] = 0.5*(omega_new[0][1]+omega_new[1][0])
    omega_new[0][m+1] = 0.5*(omega_new[0][m]+omega_new[1][m+1])
    
    omega.append(omega_new)
    psi.append(psi_new)
    
    velocity_x_new = np.zeros((m+2,m+2))
    velocity_y_new = np.zeros((m+2,m+2))
    
    #Calculate the velocity components at time step t=k from the stream function at time step t=k
    for j in range(1,m):
        velocity_x_new[j] = (1.0/(2.0*h))*(psi_new[j+1]-psi_new[j-1])
        velocity_y_new[:,j] = -(1.0/(2.0*h))*(psi_new[:,j+1]-psi_new[:,j-1])
    
    velocity_x.append(velocity_x_new)
    velocity_y.append(velocity_y_new)

    #Define the maximum number of time iterations to take
    t_step_max = int(2.0/k)

    #Counter that counts how many time iterations have been performed
    count = 0

    #Start loop through time steps
    for t in range(1, t_step_max):
        
        print('At time t=',t*k)
    
        #Calculate the auxiliary vorticity function by applying the Adams-Bashforth two step method to the vorticity at time steps
        #t=n*k and t=(n-1)*k
        omega_aux = vorticityNonLinear(k, m, omega[t], omega[t-1], velocity_x[t], velocity_y[t], velocity_x[t-1], velocity_y[t-1])
        
        #Calculate the vorticity function at time step t=(n+1)*k by applying the Adams-Bashforth two step method to the vorticity at
        #time steps t=n*k, t=(n-1)*k, and the auxiliary vorticity function
        omega_next = vorticityLinear(k, m, omega[t], omega[t-1], omega_aux)
    
        #Calculate the stream-function at time step t=(n+1)*k from the vorticity function at time step t=(n+1)*k by applying the SOR method
        psi_next = PoissonSORSolver(m,omega_next)
    
        #Update the boundary conditons of the vorticity at time step t=(n+1)*k by using the stream-function at time step t=(n+1)*k
        omega_next[m+1]=-(2.0/(h**2))*psi_next[m]+(-2.0/h)*np.ones(m+2)
        omega_next[0] = -(2.0/(h**2))*psi_next[1]
        omega_next[:,0] = -(2.0/(h**2))*psi_next[:,1]
        omega_next[:,m+1] = -(2.0/(h**2))*psi_next[:,m]
    
        #Setting the vorticity values at the corners of the grid equal to the average of it's boundary neighbors
        omega_next[m+1][0] = 0.5*(omega_next[m+1][1]+omega_next[m][0])
        omega_next[m+1][m+1] = 0.5*(omega_next[m+1][m]+omega_next[m][m+1])
        omega_next[0][0] = 0.5*(omega_next[0][1]+omega_next[1][0])
        omega_next[0][m+1] = 0.5*(omega_next[0][m]+omega_next[1][m+1])
    
        velocity_x_next = np.zeros((m+2,m+2))
        velocity_y_next = np.zeros((m+2,m+2))
    
        #Calculating the velocity components at time step t=(n+1)*k using the stream -unction at time step t=(n+1)*k
        for j in range(1,m+1):
            velocity_x_next[j] = (1.0/(2.0*h))*(psi_next[j+1]-psi_next[j-1])
            velocity_y_next[:,j] = -(1.0/(2.0*h))*(psi_next[:,j+1]-psi_next[:,j-1])
    
        velocity_x.append(velocity_x_next)
        velocity_y.append(velocity_y_next)
    
        omega.append(omega_next)
        psi.append(psi_next)
        velocity_x.append(velocity_x_next)
        velocity_y.append(velocity_y_next)

        
        #Variable to hold the difference between the vorticity at time step t=(n+1)*k and t=n*k
        error1 = 0.0
        #Variable to hold the difference between the stream-function at time step t=(n+1)*k and t=n*k
        error2 = 0.0

        #Calculating the difference of the vorticity and stream-functions at the current and previous time step
        for m in range(0,m+1):
            for n in range(0,m+1):
                error1 += (omega[t+1][m][n]-omega[t][m][n])**2
                error2 += (psi[t+1][m][n]-psi[t][m][n])**2
        
        #Updating the counter of how many time iterations have been performed
        count +=1
        
        #If the difference of the vorticity and stream-function is below a tolerance value then
        #we stop the time iterations since the solution has reached steady state
        if(np.sqrt(error1) < 0.01 and np.sqrt(error2) < 0.01):
            print('Solution has reached steady state')
            break


    #Setting grid for plotting
    x = np.arange(0.0,1.0,1.0/(m+2.0))
    y = np.arange(0.0,1.0,1.0/(m+2.0))
    x, y = np.meshgrid(x,y)
    
    #Plotting vorticity
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,omega[count],cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('vorticity[1/s]')
    plt.show()
    
    
    plt.contour(x, y, omega[count], 20, cmap='RdGy')
    plt.show()
    
    #Plotting stream-function
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,psi[count],cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('streamfunction[m^2/s]')
    plt.show()
    

    
    plt.contour(x, y, psi[count], 20, cmap='RdGy')
    plt.show()

    #Plotting horizontal velocity component
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,velocity_x[count],cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('horizontal velocity [m/s]')
    plt.show()
    
    #Plotting vertical velocity component
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,velocity_y[count],cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('vertical velocity [m/s]')
    plt.show()
    
    #Plotting streamlines
    plt.streamplot(x,y, velocity_x[count], velocity_y[count], density=3)
    plt.show()

#StokesNavierSolver(0.0001, 50)
StokesNavierSolver(0.0005, 25)
#StokesNavierSolver(0.001, 15)
#StokesNavierSolver(0.01, 10)



