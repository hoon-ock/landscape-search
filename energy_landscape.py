"""Propagating 2D dynamics on the muller potential using OpenMM.
Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import openmm as mm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np


class DoubleWell(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    def __init__(self, a=10, b=60, c=10, d=10):
        # start with a harmonic restraint on the Z coordinate
        self.aa = a 
        self.bb = b 
        self.cc = c 
        self.dd = d 
        expression = f'1000.0 * z^2 + {a}*x^4 - {b}*x^2 + {c}*x + {d}*y^2'
        super(DoubleWell, self).__init__(expression)
    
    # @classmethod   
    def potential(self, x, y):
        "Compute the potential at a given point x,y"
        value = self.aa*x**4 - self.bb*x**2 + self.cc*x + self.dd*y**2/2
        return value

    # @classmethod
    def plot(self, ax=None, minx=-3, maxx=3, miny=-5.0, maxy=5.0, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = self.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        CS = ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)
        cbar = plt.colorbar(CS)
        cbar.ax.set_ylabel('Potential Energy')


class MullerForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    def __init__(self, 
                 a=[-1, -1, -6.5, 0.7], 
                 b=[0, 0, 11, 0.6], 
                 c=[-10, -10, -6.5, 0.7], 
                 A=[-200, -100, -170, 15], 
                 X=[1, 0, -0.5, -1], 
                 Y=[0, 0.5, 1.5, 1]):
        # start with a harmonic restraint on the Z coordinate
        self.aa = a 
        self.bb = b 
        self.cc = c 
        self.AA = A 
        self.XX = X 
        self.YY = Y 
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=a[j], bb=b[j], cc=c[j], AA=A[j], XX=X[j], YY=Y[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX}) 
                               * (y - {YY}) + {cc} * (y - {YY})^2)'''.format(**fmt)
        super(MullerForce, self).__init__(expression)
    
    # @classmethod   
    def potential(self, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            value += self.AA[j] * np.exp(self.aa[j] * (x - self.XX[j])**2 + \
                self.bb[j] * (x - self.XX[j]) * (y - self.YY[j]) + self.cc[j] * (y - self.YY[j])**2)
        return value

    # @classmethod
    def plot(self, ax=None, minx=-3.0, maxx=2.0, miny=-1.0, maxy=3.0, fontsize=12, colorbar=True, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        #ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = self.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        CS = ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)
        if colorbar:
            cbar = plt.colorbar(CS)
            cbar.ax.set_ylabel('Potential Energy', fontsize=fontsize)  # Adjust label fontsize here
            cbar.ax.tick_params(labelsize=fontsize-2)


class ModifiedMullerForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    def __init__(self, a2=-0.1, c2=-0.1, A2=500, X2=-0.5582, Y2=1.4417):
        # start with a harmonic restraint on the Z coordinate
        self.aa = [-1, -1, -6.5, 0.7] 
        self.bb = [0, 0, 11, 0.6] 
        self.cc = [-10, -10, -6.5, 0.7] 
        self.AA = [-200, -100, -170, 15] 
        self.XX = [1, 0, -0.5, -1] 
        self.YY = [0, 0.5, 1.5, 1] 
        
        self.aa2 = a2
        self.cc2 = c2
        self.AA2 = A2
        self.XX2 = X2 
        self.YY2 = Y2
        
        expression = f'1000.0 * z^2 + {A2}*sin(x*y)*exp({a2}*(x-{X2})^2 + {c2}*(y-{Y2})^2)'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], 
                       AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX}) 
                               * (y - {YY}) + {cc} * (y - {YY})^2)'''.format(**fmt)
        super(ModifiedMullerForce, self).__init__(expression)
    
    # @classmethod   
    def potential(self, x, y):
        "Compute the potential at a given point x,y"
        value = self.AA2*np.sin(x*y)*np.exp(self.aa2*(x-self.XX2)**2 + self.cc2*(y-self.YY2)**2)
        for j in range(4):
            value += self.AA[j] * np.exp(self.aa[j] * (x - self.XX[j])**2 + \
                self.bb[j] * (x - self.XX[j]) * (y - self.YY[j]) + self.cc[j] * (y - self.YY[j])**2)
        return value

    def plot(self, ax=None, minx=-3.5, maxx=2.0, miny=-1.0, maxy=3.5, fontsize=12, colorbar=True, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        #ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = self.potential(xx, yy)
        if ax is None:
            ax = plt
        CS = ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)
        if colorbar:
            cbar = plt.colorbar(CS)
            cbar.ax.set_ylabel('Potential Energy', fontsize=fontsize)  # Adjust label fontsize here
            cbar.ax.tick_params(labelsize=fontsize-2)  # Adjust tick label fontsize here

def landscape(potential):
    if potential == 'Muller':
        pes = MullerForce()
    elif potential == 'Adjusted1':
        pes = MullerForce(a=[-1, -1, -6.5, 0.7],
                          b=[0, 0, 11, 0.6], 
                          c=[-10, -10, -6.5, 0.7], 
                          A=[-200, -100, -350, 15], 
                          X=[1, 0, -2.5, -1], 
                          Y=[0, 1.5, -0.5, 1])
    elif potential == 'Adjusted2':
        pes = MullerForce(a=[-1, -1, -6.5, 0.7],
                          b=[0, 0, 10, 0.6], 
                          c=[-10, -10, -6.5, 0.7], 
                          A=[-260, -100, -200, 15], 
                          X=[1, 0, -1.7, -1], 
                          Y=[-0.2, 1.5, 2, 1])
    elif potential == 'Adjusted3':
        pes = MullerForce(a=[-1, -1, -6.5, 0.7],
                          b=[0, 0, 4, 0.6], 
                          c=[-10, -10, -6.5, 0.7], 
                          A=[-200, -150, -200, 15], 
                          X=[1, -0.5, -1.7, -1], 
                          Y=[0.2, -0.5, 2.5, 1])
    elif potential == 'Modified_Muller':
        pes = ModifiedMullerForce()
    elif potential == 'DoubleWell':
        pes = DoubleWell()
    return pes