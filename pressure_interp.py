import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def sech(x):
    return 1/np.cosh(x)

def tanh_matcher(x, y2, x2, y1, x1, alpha, amin, amax, diagnostic_flag):
    """
    x: x-axis variable. Must be >= x2
    x1: x-value of first point to match to
    x2: x-value of second point to match to (x2 must be > x1)
    y1: y-value of first point to match to
    y2: y-value of second point to match to
    alpha: location of tanh curvature switch relative to x2 (must be posotive)
    amin: guess for minimum slope
    amax: guess for maximum slope
    diagnostic_flag: True will show plots for finding solution, False will hide plots
    """
    if x[-1] < x2:
        print('Error: using x value below tanh fitted function range')
    elif x2 < x1:
        print('Error: x2 must be greater than x1')
    elif y2 > y1:
        print('Error: this function only works for negative slope')
    
    slope = (y2 - y1)/(x2-x1)
    value = y2
    print(slope)
    alpha2 = alpha-x2 # shift to center on x2 instead of zero
    print('alpha2', alpha2)
    a_array = np.linspace(amin, amax, int(1e6))
    d1 = value/(1-np.tanh(a_array*x2 - alpha2))
    d2 = -slope/(a_array*sech(alpha2 - a_array*x2)**2)
    # Solve for intersection
    imin = np.where(np.abs(d2 - d1) == min(np.abs(d2 - d1)))
    if diagnostic_flag:
        ax3.plot(a_array, d1)
        ax3.plot(a_array, d2, 'r--')
        ax3.scatter(a_array[imin], d2[imin], marker='*', color='red')
    
    d = d2[imin]
    a = a_array[imin]
    
    return -d*np.tanh(a*x - alpha2) + d
 
def tanh_matcher_2_0(x, y2, x2, y1, x1, R, n, d, b, hmin, hmax, diagnostic_flag):

    """

    x: x-axis variable. Must be >= x2

    x1: x-value of first point to match to

    x2: x-value of second point to match to (x2 must be > x1)

    y1: y-value of first point to match to

    y2: y-value of second point to match to

    alpha: location of tanh curvature switch relative to x2 (must be posotive)

    amin: guess for minimum slope

    amax: guess for maximum slope

    diagnostic_flag: True will show plots for finding solution, False will hide plots

    """

    if x[-1] < x2:

        print('Error: using x value below tanh fitted function range')

    elif x2 < x1:

        print('Error: x2 must be greater than x1')

    elif y2 > y1:

        print('Error: this function only works for negative slope')

    S = (y2 - y1)/(x2-x1)

    P = y2

    print(S)

    R2 = R+x2 # shift to center on x2 instead of zero

    print('R2', R2)

    h_array = np.linspace(hmin, hmax, int(1e6))

    m1 = ((P - b) - (h_array/2)*(np.tanh((R2 - x2)/d)+1))/(R2 - x2 -d)**n

    m2 = -(S + h_array/(2*d)*(sech((R2-x2)/d))**2) / (n*(R2- x2 - d)**(n-1))

    # Solve for intersection

    imin = np.where(np.abs(m2 - m1) == min(np.abs(m2 - m1)))

    if diagnostic_flag:

        plt.plot(h_array, m1)

        plt.plot(h_array, m2, 'r--')

        plt.scatter(h_array[imin], m2[imin], marker='*', color='red')
        
    m = m2[imin]

    h = h_array[imin]
    
    return b + (h/2)*(np.tanh((R2-x)/d) + 1) + m* np.abs(R2 - x -d )**n * np.heaviside(R2-x-d, 0)
   

a = 1.2 #[m]

#core stuff
transport = np.load("/home/diab/NTARCtransport/transport_iter1b_new.npy",allow_pickle = True).item()
pcore = transport['pe'][-1]+transport['pi'][-1]
pcore = pcore*1e3*1.6e-19*1e20
rhocore = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
rhopcore = np.array([0.2,0.4,0.6,0.8])
psicore = rhocore**2
psipcore = np.array([(psicore[i+1]+psicore[i])/2 for i in range(len(psicore)-1)])
ppcore_trinity = -1/2/rhopcore*1.6e4*(transport['aLpe'][-1]/a*np.interp(psipcore,psicore,transport['pe'][-1])+transport['aLpi'][-1]/a*np.interp(psipcore,psicore,transport['pi'][-1]))
ncore = transport['n'][-1]
Tecore = transport['pe'][-1]/ncore
Ticore = transport['pi'][-1]/ncore
ppcore = np.array([(pcore[i+1]-pcore[i])/(psicore[i+1]-psicore[i]) for i in range(len(pcore)-1)]) #dp/d(psinorm)
psipcore = np.insert(psipcore,0,0)
ppcore = np.insert(ppcore,0,0)
ppcore_trinity = np.insert(ppcore_trinity,0,0)
psicore = np.insert(psicore,[0,2,len(psicore)],[0,0.16,1])
pcore = np.insert(pcore,[0,2,len(pcore)],[pcore[0]+1e3,(pcore[1]+pcore[2])*0.495,0])

#ppcore = ppcore_trinity





fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=[12, 6.5])
# Plot input pressure gradient
Pprime_in = root['PRO_create']['INPUTS']['gEQDSK']['PPRIME'].copy()
#Pprime_in /= Pprime_in[0]
psi_norm = np.linspace(0,1,len(Pprime_in))#np.concatenate([np.linspace(0,0.01,10),np.linspace(0.01,0.09,100),np.linspace(0.09,1,300)])#len(Pprime_in))
#ax1.plot(psi_norm,Pprime_in)
#psi_norm = psi_norm[20:np.argmin(np.abs(psi_norm-0.9**2))]
Pprime_new = np.zeros(len(psi_norm))
psia = psicore[0]
psib = psicore[1]
ppa = 0
k = 1
n = 0
pa = pcore[0]
pb = pcore[1]
psin = psi_norm[n]

while n<len(psi_norm):
    print("entering")
    c = (pb - pa - ppa*(psib - psia))/(0.5*(psib**2 - psia**2) - psia*(psib-psia))
    while psia<=psin<=psib:
        Pprime_new[n] = ppa+c*(psin - psia)
        n += 1
        if n >=len(psi_norm): break
        psin = psi_norm[n]
        #print(k,psin)
    psia = psib
    k+= 1
    if k >= len(psicore): break
    psib = psicore[k]
    #print(psia,psin,psib)
    ppa = Pprime_new[n-1]
    pa = pb
    pb = pcore[k]
#print(psicore,pcore)
#print(Pprime_new)
# Define the pressure gradient
#Pprime_new = np.interp(psi_norm,psipcore,ppcore)
#141.55020165, 163.19442333, 167.5882915 , 195.09494517
# Integrate to get pressure (correct for integrating backward)
pressure = cumtrapz(Pprime_new,psi_norm,initial = 0)
pressure += pcore[0]# pressure[-1]
#ax1.plot(psi_norm,Pprime_new,'x')



deg = 5
paxis = pressure[0]
#pcore = np.insert(pcore,[0,1],[paxis,paxis])
#psicore = np.insert(psicore,[0,1],[0,0.001])
psi_norm2 = psi_norm[:np.argmin(np.abs(psi_norm-0.7**2))]
psi_norm3 = psi_norm[np.argmin(np.abs(psi_norm-0.7**2)):]
rho3 = np.sqrt(psi_norm3)
rho2 = np.sqrt(psi_norm2)

f = interp1d(psicore,pcore,'cubic')
pressure2 = f(psi_norm2)


alpha = 50
hmin = 1e3
hmax = 1e6
diagnostic_flag = True
R = 0.29
n = 1.65
d = 0.02
b = 1.2e3


### Interpolating the pedestal region
pped = tanh_matcher_2_0(rho3, pressure2[-1], rho2[-1], pressure2[-2], rho2[-2], R, n, d, b, hmin, hmax, diagnostic_flag)


pressure2 = np.concatenate([pressure2,pped])
psi_norm2 = np.concatenate([psi_norm2, psi_norm3])

#pressure2 -= pressure2[-1]
# Write pressure to tree for use with PRO_create
root['PRO_create']['SETTINGS']['PHYSICS']['Ptot_in'] = pressure2
#ax2.plot(psi_norm,pressure,'o')
#ax2.plot(np.sqrt(psi_norm2[np.argmin(np.abs(psi_norm-0.7**2)):np.argmin(np.abs(psi_norm-0.95**2))]),pressure2[np.argmin(np.abs(psi_norm-0.7**2)):np.argmin(np.abs(psi_norm-0.95**2))],'o')
ax2.plot(psicore,pcore,'o')
#ax2.plot(psipcore,ppcore,'o')
#ax2.plot(np.insert(rhopcore,0,0)**2,ppcore_trinity,'o')
ax2.plot(psi_norm2,pressure2,'o')
print(np.sqrt(psi_norm2),len(psi_norm2))
print(pressure2[np.argmin(np.abs(psi_norm2 - 0.9**2))])
print(pcore)
#print(pressure2)
deriv = np.gradient(pressure2)
ax1.plot(psi_norm,deriv)
#print((Tecore+Ticore)/2)
#print(ncore)
print(pressure2)
np.save("/home/diab/NTARCtransport/pressure.npy",pressure2)
np.save("/home/diab/NTARCtransport/psi_norm.npy",psi_norm2)
ax2.set_xlabel(r"$\Psi_n$")
ax1.set_xlabel(r"$\Psi_n$")
ax1.set_ylabel("pressure gradient (Pa)")
ax2.set_ylabel("pressure (Pa)")
print(ncore)
print(pressure2)

