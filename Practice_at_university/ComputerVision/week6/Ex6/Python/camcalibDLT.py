import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    A=[]
    for i in range(len(x_im)):
        x,y,z = x_world[i,0], x_world[i,1], x_world[i,2]
        u,v = x_im[i,0], x_im[i,1]
        A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
        
    A = np.array(A)
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    U,S,V=np.linalg.svd(np.dot(A.T,A))
    L=V[-1,:]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    #P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    P = np.reshape(L, (3, 4))  # remove this and uncomment the line above
    
    return P
