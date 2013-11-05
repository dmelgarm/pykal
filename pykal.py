'''
D Melgar Nov-2013

Kalman filters for seismology
'''

def kalda(d,a,dtd,dta):
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
    x=np.zeros((4,))