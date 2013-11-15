'''
D Melgar Nov-2013

Kalman filters for seismology

There are several routines in this module

    kalmand() is the forward Kalman filter that uses the accelerations as control
        inputs and the GPS as system measurements. This requires teh accelerometer
        data to be zeroth order baseline corrected, i.e. pre-event mean must be 
        removed.
    kalmans() same routine as kalmand() but with a backwards smoother over the 
        entire filtering interval
        
If you use these codes or require further information please read and reference BOTH 
these papers:
    
     * D. Melgar, Bock, Y., Sanchez, D. & Crowell, B.W. (2013), On Robust and Reliable 
     Automated Baseline Corrections for Strong Motion Seismology, J. Geophys. Res, 
     119, 1-11, doi:10.1002/jgrb.50135.
     * Y. Bock, Melgar, D. & Crowell, B.W., (2011), Real-Time Strong-Motion Broadband 
     Displacements from Collocated GPS and Accelerometers, Bull. Seism. Soc. Am., 
     101(5) 2904-2925.
'''

def kalmand(td,d,ta,a,dtd,dta,q,r):
    '''
    The classic forward Kalman filter formulation from Bock et al (2011, BSSA)
    
    Usage:
        dk,vk,tk = kalmand(d,a,dtd,dta,q,r)
        
        d - displacement time series (m)
        dtd - displacement sampling interval (s)
        a - acceleration tiem series (m/s^2)
        dta - accelerationsampling interval (s)
        q - time series of system variances
        r - time series of observation variances
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
    B=np.array([[0.5*(dta**2)],[dta]])
    kd=0
    for ka in range(len(a)):
        #Time update
        Q=np.array([[(1/3)*q[ka]*dta**3,0.5*q[ka]*dta**2],[0.5*q[ka]*dta**2,q[ka]*dta]])
        #Predict state
        x=np.dot(A,x)+np.dot(B,a[ka])
        #Predict covariance
        P=np.dot(A,np.dot(P,A.T))+Q
        #Measurememt update
        if np.allclose(ta[ka]-td[kd],0):  #GPS available
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
    print "--> "+str(kd)+" measurement updates in Kalman filter"
    return tk,dk,vk
        
        
def kalmans(td,d,ta,a,dtd,dta,q,r):
    '''
    The classic Kalman filter formulation with an RTS backwards smoother from 
    Bock et al (2011, BSSA)
    
    Usage:
        dk,vk,tk = kalmans(d,a,dtd,dta,q,r)
        
        d - displacement time series (m)
        dtd - displacement sampling interval (s)
        a - acceleration tiem series (m/s^2)
        dta - accelerationsampling interval (s)
        q - time series of system variances
        r - time series of observation variances
    '''
    import numpy as np
    #Initalize outputs
    tk=np.zeros(ta.shape)
    vk=np.zeros(ta.shape)
    dk=np.zeros(ta.shape)
    #Initalize ndarrays for covariances in smoother
    Pplus=np.zeros((2,2,len(ta)))
    Pminus=np.zeros((2,2,len(ta)))
    #Initalize system statesre
    x=np.zeros((2,1))
    xplus=np.zeros((2,len(ta)))
    xminus=np.zeros((2,len(ta)))
    #Initalize covariance
    P=np.eye(2)
    #Define measurement matrix
    H=np.array([1,0])[None]
    #Define state transition matrix
    A=np.array([[1,dta],[0,1]])
    B=np.array([[0.5*(dta**2)],[dta]])
    kd=0
    for ka in range(len(a)):
        #Time update
        Q=np.array([[(1/3)*q[ka]*dta**3,0.5*q[ka]*dta**2],[0.5*q[ka]*dta**2,q[ka]*dta]])
        #Predict state
        x=np.dot(A,x)+np.dot(B,a[ka])
        #Predict covariance
        P=np.dot(A,np.dot(P,A.T))+Q
        #Save estiamtes for smoother
        xminus[:,ka]=x.T
        Pminus[:,:,ka]=P
        #Measurememt update
        if np.allclose(ta[ka]-td[kd],0):  #GPS available
            R=r[kd]/dtd
            #Compute filter gain
            K=np.dot(P,H.transpose())/(np.dot(H,np.dot(P,H.transpose()))+R)
            #Update state
            x=x+np.dot(K,d[kd]-np.dot(H,x))
            #Update covariance
            P=np.dot(np.eye(2)-np.dot(K,H),P)
            #Save updates for smoother
            Pplus[:,:,ka]=P
            xplus[:,ka]=x.T
            #Update measurmemnt counter
            kd=kd+1
            if kd>=d.shape[0]: #No measurements left
                kd=kd-1
        else: #No measurement was necessary
            #Save updates for smoother
            Pplus[:,:,ka]=P
            xplus[:,ka]=x.T
        #Update output
        tk[ka]=ta[ka]
        dk[ka]=x[0]
        vk[ka]=x[1]
    print "--> "+str(kd)+" measurement updates in Kalman filter"
    print "--> Running RTS smoother"
    #Go to smoothigns tage
    s=np.zeros((2,len(ta)))
    #Initalize smoothed time series
    s[0,:]=dk
    s[1,:]=vk
    #Backwards smooth
    k=len(ta)-1
    while k>0:
        k=k-1
        #Covariances for gain
        P1=Pplus[:,:,k]
        P2=Pminus[:,:,k+1]
        #Smoother gain
        F=np.dot(P1,np.dot(A.T,np.linalg.inv(P2)))
        #Update state
        xp=xplus[:,k][None].T
        xm=xminus[:,k+1][None].T
        s[:,k]=(xp+np.dot(F,s[:,k+1][None].T-xm)).T
    #Update outputs
    dk=s[0,:]
    vk=s[1,:]
    return tk,dk,vk  
    
    
def kaldau(td,d,ta,a,dtd,dta,qa,qomega,r):
    '''
    A reformulation of the classic filter. Still Use GPS as a measurements. But
    Introduce a new state variable, omega, the DC offset in the acceleration 
    time series and estimate it epoch by epoch. This is better or real-time 
    because you do not need to remove pre-event means
    
    Usage:
        tk,dk,vk,Ok = kaldau(d,a,dtd,dta,qa,qomega,r)
        
        td - Time vector for displacements
        d - Dispalcement time series
        ta - Time vector for accelerations
        a - Acceleration tiem series
        dtd - Displacement sampling rate
        dta - Accelerometer sampling rate
        qa - Acceleration system noise
        qomega - DC offset system noise
        r - Displacement measurement noise
        
        Returns tk,dk,Ok
        
        tk - Kalman fitlered tiem vector
        dk - Filtered dispalcement
        vk - Filtered velocity
        Ok - Filtered DC offset
        
    '''  
    import numpy as np
    #Initalize outputs
    tk=np.zeros(ta.shape)
    vk=np.zeros(ta.shape)
    dk=np.zeros(ta.shape)
    Ok=np.zeros(ta.shape)
    #Initalize system states
    x=np.zeros((3,1))
    x[0]=d[0]
    x[2]=a[0]
    #Initalize covariance
    P=np.zeros(3)
    #Define measurement matrix and state transition matrices
    H=np.array([1,0,0])[None]
    A=np.array([[1,dta,-dta**2/2],[0,1,-dta],[0,0,1]])
    B=np.array([[0.5*(dta**2)],[dta],[0]])
    kd=0
    for ka in range(len(a)):
        #Time update
        Q=np.array([[(1/3)*qa[ka]*dta**3,0.5*qa[ka]*dta**2,0],
            [0.5*qa[ka]*dta**2,qa[ka]*dta+qomega[ka]*dta**3/3,-qomega[ka]*dta**2/2],
            [0,-qomega[ka]*dta**2/2,qomega[ka]*dta]])
        #Predict state
        x=np.dot(A,x)+np.dot(B,a[ka])
        #Predict covariance
        P=np.dot(A,np.dot(P,A.T))+Q
        #Measurememt update
        if np.allclose(ta[ka]-td[kd],0):  #GPS available
            R=r[kd]/dtd
            #Compute filter gain
            K=np.dot(np.dot(P,H.transpose()),np.linalg.inv(np.dot(H,np.dot(P,H.transpose()))+R))
            #Update state
            x=x+np.dot(K,d[kd]-np.dot(H,x))
            #Update covariance
            P=np.dot(np.eye(3)-np.dot(K,H),P)
            kd=kd+1
            if kd>=d.shape[0]:
                kd=kd-1
        #Update output
        tk[ka]=ta[ka]
        dk[ka]=x[0]
        vk[ka]=x[1]
        Ok[ka]=x[2]
    print "--> "+str(kd)+" measurement updates in Kalman filter"
    return tk,dk,vk,Ok
    

def kaldaz(td,d,ta,a,dtd,dta,qa,qomega,rd,ra):
    '''
    Use GPS and acceleroemer as measurements. This approach is in contrast
    to the original one in Bock et al (2011,BSSA) where we treated acceleration
    as a system input isntead of a measurememnt. The benefit is that now you 
    don't need to remove the DC offset before filtering, rather this code will 
    estimate the DC offset itself.
    
    Usage:
        dout,vout,aout,DC = kaldaz(d,a,dtd,dta,qomega,qa,rd,ra)
        
        td - Time vector for displacements
        d - Dispalcement time series
        ta - Time vector for accelerations
        a - Acceleration tiem series
        dtd - Displacement sampling rate
        dta - Accelerometer sampling rate
        qa - Acceleration system noise
        qomega - DC offset system noise
        rd - Displacement measurement noise
        ra - Acceleration measurement noise
        
        Returns tk,dk,vk,ak,Ok
        
        tk - Kalman fitlered tiem vector
        dk - Filtered dispalcement
        vk - Filtered velocity
        ak - Filtered acceleration
        Ok - Filtered DC offset
        
    '''
    
    import numpy as np
    
    tk=np.zeros(ta.shape)
    vk=np.zeros(ta.shape)
    dk=np.zeros(ta.shape)
    ak=np.zeros(ta.shape)
    Ok=np.zeros(ta.shape)
    #Initalize system states
    x=np.zeros((4,))[None].T
    x[0]=d[0]
    x[3]=a[0]
    #Initalize covariance
    P=np.eye(4)
    #Define measurement matrices
    Had=np.array([[1,0,0,0],[0,0,1,1]])
    Ha=np.array([0,0,1,1])[None]
    #Define state transition matrix
    A=np.array([[1,dta,0.5*dta**2,0],[0,1,dta,0],[0,0,1,0],[0,0,0,1]])
    kd=0
    for ka in range(len(a)):
        #Time update
        #Q=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,(1+dta)*q[ka]*dta]])
        Q=np.array([[0,0,0,0],[0,0,0,0],[0.25*(dta**4*qa[ka]**2),0.5*(dta**3*qa[ka]**2),dta**2*qa[ka]**2+dta*qa[ka],0],[0,0,0,(1+dta)*qomega[ka]*dta]])
        #Predict state
        x=np.dot(A,x)
        #Predict covarian
        P=np.dot(A,np.dot(P,A.T))+Q
        #Measurememt update (GPS+accel)
        if np.allclose(ta[ka]-td[kd],0):  #GPS available
            R=np.array([[rd[kd]/dtd,0],[0,ra[ka]/dta]])
            #Compute filter gain
            K=np.dot(np.dot(P,Had.transpose()),np.linalg.inv(np.dot(Had,np.dot(P,Had.transpose()))+R))
            #Update state
            z=np.array([[d[kd]],[a[ka]]])
            x=x+np.dot(K,z-np.dot(Had,x))
            #Update covariance
            P=np.dot(np.eye(4)-np.dot(K,Had),P)
            kd=kd+1
            if kd>=d.shape[0]:
                kd=kd-1
        else: #Only accel measurement
            R=np.array(ra[ka]/dta)
            #Gain
            K=np.dot(P,Ha.T)/(np.dot(Ha,np.dot(P,Ha.T))+R)
            #Update state
            z=np.array(a[ka])
            x=x+np.dot(K,z-np.dot(Ha,x))
            #Update covariance
            P=np.dot(np.eye(4)-K*Ha,P)
        #Update output
        tk[ka]=ta[ka]
        dk[ka]=x[0]
        vk[ka]=x[1]
        ak[ka]=x[2]
        Ok[ka]=x[3]
    print "--> "+str(kd)+" measurement updates in Kalman filter"
    return tk,dk,vk,ak,Ok
    
    
#--------------------        SUPPORTING TOOLS          -------------------------
    
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
    R=array([[-sin(lat)*cos(lon),-sin(lat)*sin(lon),cos(lat)],[-sin(lon),cos(lon),0],
        [cos(lon)*cos(lat),cos(lat)*sin(lon),sin(lat)]])
    for k in range(len(x)):
        xyz=array([[x[k]-x[0]],[y[k]-y[0]],[z[k]-z[0]]])
        neu=dot(R,xyz)
        n[k]=neu[0]
        e[k]=neu[1]
        u[k]=neu[2]
    return n,e,u
        
    