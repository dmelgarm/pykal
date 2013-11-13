'''
D Melgar Nov-2013

Kalman filters for seismology
'''

def kalmand(td,d,ta,a,dtd,dta,q,r):
    '''
    The classic Kalmanfitler formulation from Bock et al (2011, BSSA)
    
    Usage:
        dk,vk,tk = kalmand(d,a,dtd,dta)
    '''
    import numpy as np
    #Initalize outputs
    tk=np.zeros(ta.shape)
    vk=np.zeros(ta.shape)
    dk=np.zeros(ta.shape)
    #Initalize system states
    x=np.zeros((2,1))
    #Initalize covariance
    P=np.eye(2)
    #Define measurement matrix
    H=np.array([1,0])[None]
    #Define state transition matrix
    A=np.array([[1,dta],[0,1]])
    B=np.array([[0.5*dta**2],[dta]])
    kd=0
    for ka in range(len(a)):
        #Time update
        Q=np.array([[(1/3)*q[ka]*dta**3,0.5*a[ka]*dta**2],[0.5*a[ka]*dta**2,q[ka]*dta]])
        #Predict state
        x=np.dot(A,x)+np.dot(B,a[ka])
        #Predict covariance
        P=np.dot(A,np.dot(P,A.T))+Q
        #Measurememt update
        if np.allclose(ta[ka],td[kd]):  #GPS available
            R=r[kd]/dtd
            #Compute filter gain
            K=np.dot(P,H.transpose())/(np.dot(H,np.dot(P,H.transpose()))+R)
            #Update state
            x=x+np.dot(K,d[kd]-np.dot(H,x))
            #Update covariance
            P=np.dot(np.eye(2)-np.dot(K,H),P)
            kd=kd+1
            if kd>=d.shape[0]:
                kd=kd-1
        #Update output
        tk[ka]=ta[ka]
        dk[ka]=x[0]
        vk[ka]=x[1]
    return tk,dk,vk
        
        

def kaldaz(td,d,ta,a,dtd,dta):
    '''
    Use GPS and acceleroemer as measurements. This approach is in contrast
    to the original one in Bock et al (2011,BSSA) where we treated acceleration
    as a system input isntead of a measurememnt. The benefit is that now you 
    don't need to remove the DC offset before filtering, rather this code will 
    estimate the DC offset itself.
    
    Usage:
        dout,vout,aout,DC = kaldaz(d,a,dtd,dta)
        
    '''
    
    import numpy as np
    
    #Initalize system states
    x=np.zeros((4,))
    #Initalize covariance
    P=np.eye(2)
    #Define measurement matrix
    Ha=np.array([[1,0,0,0],[0,0,1,1]])
    #Define state transition matrix
    A=np.array([[1,dta,0.5*dta**2,0],[0,1,dta,0],[0,0,1,0],[0,0,0,1]])
    
    
def xyz2neu(x,y,z,lat,lon):
    '''
    Convert Earth centered Earth fixed coordiantes to local North,East,Up
    
    x,y,z - Vectors of XYZ coordinates
    lat,lon - Local station coordiantes in degrees
    
    Returns n,e,u vectors of local cartesian coordinates
    '''
    from numpy import array,zeros,dot,cos,sin,deg2rad
    
    lat=deg2rad(lat)
    lon=deg2rad(lon)
    n=zeros(x.shape)
    e=zeros(x.shape)
    u=zeros(x.shape)
    R=array([[-sin(lat)*cos(lon),-sin(lat)*sin(lon),cos(lat)],[-sin(lon),cos(lon),0],[cos(lon)*cos(lat),cos(lat)*sin(lon),sin(lat)]])
    for k in range(len(x)):
        xyz=array([[x[k]-x[0]],[y[k]-y[0]],[z[k]-z[0]]])
        neu=dot(R,xyz)
        n[k]=neu[0]
        e[k]=neu[1]
        u[k]=neu[2]
    return n,e,u
        
    