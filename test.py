import pykal
import numpy as np
import matplotlib.pyplot as pl



#Accelerometer CI WES HNN 2010/04/04,22:40:32.849 1270420832.849     0.010000
anfile='/Users/dmelgarm/Research/Events/14607652/14607652.CI.WES.HNN.ascii'
aefile='/Users/dmelgarm/Research/Events/14607652/14607652.CI.WES.HNE.ascii'
azfile='/Users/dmelgarm/Research/Events/14607652/14607652.CI.WES.HNZ.ascii'
an=np.loadtxt(anfile)
an=an/100
ae=np.loadtxt(aefile)
ae=ae/100
az=np.loadtxt(azfile)
az=az/100
a=an
ta=np.arange(len(an))*0.01+(22*3600)+(40*60)+32.85+15
dta=0.01

#What station?
s='P494'
lon=-115.73206681
lat=32.75965554
z=39.48
#Files
gpsfile='/Users/dmelgarm/Research/Events/GPS/EMC_GNPS_fix_2010_04_04.txt'
ftype=np.dtype([('station','S4'),('x',float),('y',float),('z',float),('yr',int),
    ('month',int),('day',int),('hr',int),('min',int),('sec',int)])
#Load data
gps=np.loadtxt(gpsfile,dtype=ftype,delimiter=' ')
sta=gps['station']
#Find GPS station
i=np.nonzero(sta==s)
i=i[0]
#Load XYZ GPS data
x=gps['x'][i]
y=gps['y'][i]
z=gps['z'][i]
hr=gps['hr'][i]
mi=gps['min'][i]
se=gps['sec'][i]
#Make time vector
td=hr*3600+mi*60+se
#Rotate to latlon
n,e,u=pykal.xyz2neu(x,y,z,lat,lon)
#Get channel variance
r=np.var(n[0:500])

#Now trim them
i1=np.nonzero(td>ta[0])[0]
i2=np.nonzero(td<ta[-1])[0]
i=np.intersect1d(i1,i2)
td=td[i]
dtd=1
n=n[i]
e=e[i]
u=u[i]

#Run filter
Kq=1
q=np.var(a[0:1000])*Kq
q=np.ones(ta.shape)*q
r=np.ones(td.shape)*r
qomega=1e-10
qomega=np.ones(ta.shape)*qomega
qa=6e-1
qa=np.ones(ta.shape)*qa
#tk,dk,vk=pykal.kalmans(td,n,ta,a-np.mean(a[0:500]),dtd,dta,q,r)
#tk,dk,vk,ak,Ok=pykal.kaldaz(td,n,ta,a,dtd,dta,qa,qomega,rd=r,ra=q)
print q[0]
print qomega[0]
print r[0]
tk,dk,vk,Ok=pykal.kaldau(td,n,ta,a,dtd,dta,q,qomega,r)
ak=a-Ok

tmp=np.zeros((n.shape[0],2))
tmp[:,0]=td
tmp[:,1]=n
np.savetxt('wes_p494.disp',tmp)
tmp=np.zeros((a.shape[0],2))
tmp[:,0]=ta
tmp[:,1]=a
np.savetxt('wes_p494.acc',tmp)
tmp=np.zeros((dk.shape[0],4))
tmp[:,0]=tk
tmp[:,1]=dk
tmp[:,2]=vk
tmp[:,3]=Ok
np.savetxt('wes_p494.kal',tmp)

pl.figure()
pl.subplot(211)
pl.scatter(td,n,color='blue')
pl.plot(tk,dk)
pl.grid()
pl.subplot(212)
pl.plot(tk,ak,tk,Ok)
#pl.legend(['GPS','Kalman'])
pl.grid()
pl.show()
