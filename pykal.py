'''
D Melgar Nov-2013

Kalman filters for seismology
'''

def kaldam(d,a,dtd,dta):
    '''
    Use GPS and acceleroemer as measurements. This approach is in contrast
    to the original one in Bock et al (2011,BSSA) where we treated acceleration
    as a system input isntead of a measurememnt. The benefit is that now you 
    don't need to remove the DC offset before filtering, rather this code will 
    estimate the DC offset itself.
    
    Usage:
        dout,vout,aout,DC = kalda(d,a,dtd,dta)
        
    '''
    
    import numpy as np
    
    #Initalize system states
    x=np.zeros((4,))
    #Initalize covariance
    P=np.eye(2)
    #Define acceleration only measurement matrix
    Ha=np.array([[0,0,0,0],[0,0,0,0],[0,0,1,-1],[0,0,0,1]])
    #Define acceleration+gps measurement matrix
    Ha=np.array([[1,0,0,0],[0,0,0,0],[0,0,1,-1],[0,0,0,1]])
    #Define state transition matrix
    A=np.array([[1,dta,0.5*dta**2,0],[0,1,dta,0],[0,0,1,0],[0,0,0,1]])
    