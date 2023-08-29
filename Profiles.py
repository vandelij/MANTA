import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

'''
This class is called within the Trinity-Engine library

It handles general profiles (n, p, F, gamma, Q, etc)
with options to evaluate half-steps and gradients when initializing new Profile objects
'''
class Profile():
    def __init__(self, arr, grid):

        # take a 1D array to be density, for example
        self.profile = np.array(arr) 
        self.length  = len(arr)

        self.axis = grid.rho_axis 

    def plot(self,show=False,new_fig=False,label=''):

        if (new_fig):
            plt.figure(figsize=(4,4))

        #ax = np.linspace(0,1,self.length)
        #plt.plot(ax,self.profile,'.-')

        if (label):
            plt.plot(self.axis,self.profile,'.-',label=label)
        else:
            plt.plot(self.axis,self.profile,'.-')

        if (show):
            plt.show()

    __array_ufunc__ = None

    # operator overloads that automatically dereference the profiles
    def __add__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile + B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile + B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__add__")

    def __radd__(A,B):
        return A.__add__(B)

    def __sub__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile - B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile - B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__sub__")

    def __rsub__(A,B):
        return -1*(A.__sub__(B))

    def __mul__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile * B.profile, A.grid)
        elif (hasattr(B, 'size') and B.size == 1) or (isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile)) or not hasattr(B, '__len__'):
            return A.__class__(A.profile * B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__mul__")

    def __rmul__(A,B):
        return A.__mul__(B)

    def __truediv__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile / B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile / B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")

    def __rtruediv__(A,B):
        if isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(B / A.profile, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")

    def __neg__(A):
        return -1*A

    def __pow__(A, b):
        return A.__class__(A.profile**b, A.grid)
  
    def __eq__(A, B):
        if isinstance(B, A.__class__):
            return (A.profile == B.profile).any()
        else:
            raise Exception("Type mismatch in Profile.__eq__")
        
    def __getitem__(A, i):
        return A.profile[i]

    def __setitem__(A, i, val):
        A.profile[i] = val

    def __len__(A):
        return len(A.profile)

    def __repr__(A):
        return A.profile.__repr__()

    def __set__(A, B):
        if isinstance(B, A.__class__):
            A = B
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            A.profile = B
        else:
            raise Exception("Type mismatch in Profile.__set__")
        

class GridProfile(Profile):

    '''
    Profile class with values on the N_radial rho_axis grid points
    '''
    
    def __init__(self, arr, grid):

        self.grid = grid
        self.axis = grid.rho_axis
        self.length = len(self.axis)
        if hasattr(arr, '__len__'):
            self.profile = arr
        else:
            self.profile = np.ones(self.length)*arr

        self.isGridProfile = True
        self.isFluxProfile = False

    def gradient(self):
        grad_f = np.zeros(self.length)
        f = self.profile
        dr = self.grid.drho
        N = self.length
   
        # inner boundary
        grad_f[0] = (2.*f[3] - 9.*f[2] + 18.*f[1] - 11.*f[0])/(6.*dr)
        grad_f[1] = (-f[3] + 6.*f[2] - 3.*f[1] -2.*f[0])/(6.*dr)
        
        # interior (4 pt centered)
        for ix in np.arange(2, N-2):
            grad_f[ix] = (f[ix-2] - 8.*f[ix-1] + 8.*f[ix+1] - f[ix+2]) / (12.*dr)

        # outer boundary
        grad_f[N-2] = (f[N-4] - 6.*f[N-3] + 3.*f[N-2] + 2.*f[N-1])/(6.*dr)
        grad_f[N-1] = (-2.*f[N-4] + 9.*f[N-3] - 18.*f[N-2] + 11.*f[N-1])/(6.*dr)

        return GridProfile(grad_f, self.grid)

    def log_gradient(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.gradient()/self

    def gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with gradient values evaluated at midpoints
        '''

        flux_length = self.length - 1
        grad_prof = np.zeros(flux_length)
        prof = self.profile
        dr = self.grid.drho

        # inner boundary (uncentered)
        ix = 0
        grad_prof[ix] = (-prof[ix+3] + 3.*prof[ix+2] + 21.*prof[ix+1] - 23.*prof[ix]) / (24.*dr)

        # outer boundary (uncentered)
        ix = flux_length - 1
        grad_prof[ix] = (prof[ix-2] - 3*prof[ix-1] - 21.*prof[ix] + 23.*prof[ix+1]) / (24.*dr)

        # interior (centered)
        for ix in np.arange(1, flux_length-1):
            grad_prof[ix] = (prof[ix-1] - 27.*prof[ix] + 27.*prof[ix+1] - prof[ix+2]) / (24.*dr)

        return FluxProfile(grad_prof, self.grid)

    def log_gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with log gradient values
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.gradient_as_FluxProfile()/self.toFluxProfile()

    def toFluxProfile(self):
 
        flux_length = self.length - 1
        flux_prof = np.zeros(flux_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        flux_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix+1] - prof[ix+2])

        # outer boundary (third order)
        ix = flux_length-1
        flux_prof[ix] = 0.125*(3.0*prof[ix+1] + 6.0*prof[ix] - prof[ix-1])

        # interior (fourth order)
        for ix in np.arange(1, flux_length-1):
            flux_prof[ix] = 0.0625*(-prof[ix-1] + 9.0*prof[ix] + 9.0*prof[ix+1] - prof[ix+2])

        # enforce positivity if profile is everywhere positive
        if np.all(prof > 0):
            flux_prof[flux_prof < 0] = 1e-16

        return FluxProfile(flux_prof, self.grid)

    # evaluate at j+1/2
    def plus(self):
        return self.toFluxProfile().plus()

    # evaluate at j-1/2
    def minus(self):
        return self.toFluxProfile().minus()

    def plus1(self):
        arr = np.roll(self.profile, -1)
        arr[-1] = 0.0
        return GridProfile(arr, self.grid)

    def minus1(self):
        arr = np.roll(self.profile, 1)
        arr[0] = 0.0
        return GridProfile(arr, self.grid)

class FluxProfile(Profile):

    '''
    Profile class with values on the N_radial-1 mid_axis midpoints 
    '''

    def __init__(self, arr, grid):

        self.grid = grid
        self.axis = grid.mid_axis
        self.length = len(self.axis)
        if hasattr(arr, '__len__'):
            self.profile = arr
        else:
            self.profile = np.ones(self.length)*arr

        self.isGridProfile = False
        self.isFluxProfile = True

    def toGridProfile(self, axis_val = 0.0):

        grid_length = self.length + 1
        grid_prof = np.zeros(grid_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        grid_prof[ix] = 0.125*(3.0*axis_val + 6.0*prof[ix] - prof[ix+1])

        ix = 1
        grid_prof[ix] = 0.0625*(-axis_val + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix])

        # outer boundary (third order)
        ix = grid_length-2
        grid_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix-1] - prof[ix-2])

        # this is only second order accurate, but these values are never really needed (outer boundary is fixed)
        ix = grid_length-1
        grid_prof[ix] = 2.0*prof[ix-1]-prof[ix-2]

        # interior (fourth order)
        for ix in np.arange(2, grid_length-2):
            grid_prof[ix] = 0.0625*(-prof[ix-2] + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix+1])

        # enforce positivity if profile is everywhere positive
        if np.all(prof > 0):
            grid_prof[grid_prof < 0] = 1e-16

        return GridProfile(grid_prof, self.grid)

    # FluxProfile already evaluated at j+/-1/2 grid, so just need to shift for minus
    def plus(self):
        return self

    def minus(self):
        arr = np.roll(self.profile, 1)
        arr[0] = 1e-16
        return FluxProfile(arr, self.grid)
    
    #def plus(self):
    #    return self.toFluxProfile()

    #def minus(self):
    #    return self.toFluxProfile().minus1()

    #def plus1(self):
    #    arr = np.roll(self.profile, -1)
    #    arr[-1] = 0.0
    #    return GridProfile(arr, self.grid)

    #def minus1(self):
    #    arr = np.roll(self.profile, 1)
    #    arr[0] = 0.0
    #    return GridProfile(arr, self.grid)
