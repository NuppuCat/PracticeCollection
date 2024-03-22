import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
   
    
    a1,b1,_=x1
    a2,b2,_=x2
    x1x=[]
    x1x.append([0,-1,b1])
    x1x.append([1,0,-a1])
    x1x.append([-b1,a1,0])
    x1x=np.array(x1x)
    x2x=[]
    x2x.append([0,-1,b2])
    x2x.append([1,0,-a2])
    x2x.append([-b2,a2,0])
    x2x=np.array(x2x)
    L=np.dot(x1x,P1)
    R=np.dot(x2x,P2)
    A= np.vstack((L,R))
 
    
    U,S,V=np.linalg.svd(np.dot(A.T,A))
    N=V[-1,:]
   
   
   
    X = N
     ##-your-code-ends-here-##
    return X
