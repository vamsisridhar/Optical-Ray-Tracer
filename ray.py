# -*- coding: utf-8 -*-
"""
This modules deals with the initialisation of Ray objects and RayBundles.
"""
import numpy as np
import vector as vec
import matplotlib.pyplot as plt



class Ray:
    """
    A Ray object represents a singular ray of light in ray optics
    """
    
    def __str__(self):
        return f"Ray at {self.p()} with direction {self.k()}."
    
    def __init__(self, p = [0.0,0.0,0.0], k = [0.0,0.0,0.0], color=[1,0,0]):
        
        """
        Input Parameters
        ----------
        p : list, initial position = [0.0,0.0,0.0].
        k : list, initial direction = [0.0,0.0,0.0].
        color : list, color of the ray on the plot = [1,0,0].
        
        Hidden Variables
        --------
        __vertices : list, stores propagated points of the ray

        Initialisation of the Ray
        """
        
        self.__color = color
        # __vertices stores the list of points the ray has propagated through
        self.__vertices = [] 
        # appends the first point in the ray path
        self.append(p, vec.normalise(k))
        
    
    def append(self, p, k):
        
        """
        Parameters
        ----------
        p : list, initial position of ray
        k : list, initial direction of ray
        
        Sets p and k to be the current __p and __k and appends p to __vertices
        """
        
        # validation if p and k are 3D only
        
        # sets current __p and __k to the new p and k
        if len(p) == 3:
            # _p is the current position of the ray
            self.__p = np.array(p)
        else:
            raise Exception('position needs to be a array of length 3')
        
        if len(k) == 3:
            #_k is the current direction of the ray
            self.__k = np.array(k)
        else:
            raise Exception('direction needs to be a array of length 3')
        
        # appends new p to vertices array
        self.vertices().append(self.p())
    
    #getters
    
    def p(self): return self.__p
    def k(self): return self.__k
    def vertices(self): return self.__vertices
    def color(self): return self.__color
    

class RayBundle:
    """
    A bundle of rays represent a beam of light made of multiple light rays uniformly distributed
    """
    def __str__(self):
        return f"RayBundle with radius {self.radius()} with direction {self.k()}."
    def __init__(self, radius,rings,  k = [0,0,1],centre = [0.0,0.0,0.0],color = [1.0, 0.0, 0.0], ring1_ray_count = 6,empty = False):
        
        """
        Input Parameters
        ----------
        radius : float, radius of the bundle = required
        rings : int, number of concentric rings of rays in bundle = required
        k : list, direction vector of the bundle = [0,0,1]
        centre : list, centre of the bundle = [0.0,0.0,0.0]
        color : list, unified color for all rays in bundle = [1.0, 0.0, 0.0]
        ring1_ray_count : int, number of rays in the first ring = 6
        empty : bool, creates an empty bundle with all params set to None
                allows user to populate bundle with custom rays using make_bundle
                this ignores any value set for radius and rings during initialisation
                = False
        
        Hidden Variables
        ----------
        __beam_pos : list, stores positions of the rays
        __beam : list, stores the rays
        __uniform : bool, (True) bundle is unformly distributed
                    or (False) is filled with arbitrary rays
                
        Initialises a RayBundle
        """
        
        if empty:
            self.__radius = None
            self.__rings = None
            self.__k = None
            self.__centre = None
            self.__ring1_ray_count = None
            self.__color = None
            self.__beam_pos = None
            self.__beam = None
            self.__uniform = False
        else:
            if not isinstance(rings, int):
                raise Exception("ring number must be an integer!")
            self.__radius = radius
            self.__rings = rings
            self.__ring1_ray_count = ring1_ray_count
            self.__k = k
            self.__centre = np.array(centre)
            self.__color = color

            radial_spacing = radius/rings # calculates the radial spacing between the concentric rings of rays
            # first position is always the centre so it appended into the beam_pos
            # Ray is created at that position and appened to beam
    
            self.__beam_pos = [self.__centre] # stores ray positions for convinience
            self.__beam = [Ray(self.__centre, k)] # stores rays

            #for loop for individual rays
            for i in range(self.rings()):
                for j in range(self.ring1_ray_count()*(i+1)):
                    
                    """
                    The calculation for the concentric ring positions are done in the complex plane.
                    angle_spacing: the angular spacing between each successive point in each ring
                    position_complex: the positioning of the points in the complex plane, centred at 0
                    position: convert the complex number into a real vector and translate it by the centre            
                    """
                    
                    angle_spacing = (2*np.pi/(self.ring1_ray_count()*(i+1)))
                    
                    position_complex = (radial_spacing * (i+1))*np.exp((j+1)*angle_spacing*(1j)) #plotting an uniform distribution of points in the complex plane
                    position = self.centre() + np.array([position_complex.real, position_complex.imag, 0])
                    self.ray_positions().append(position) # beam_pos is used to create a x,y plot of the points
                    self.bundle().append(Ray(position, k, color = self.color()))
                    
            self.__is_uniform = True
            
    #getters
    
    def radius(self): return self.__radius
    def rings(self): return self.__rings
    def ring1_ray_count(self): return self.__ring1_ray_count
    def k(self): return self.__k
    def centre(self): return self.__centre
    def ray_positions(self): return self.__beam_pos
    def bundle(self): return self.__beam
    def color(self): return self.__color
    def is_uniform(self): return self.__is_uniform()
    
    #setters
    
    def set_bundle(self, ray_list): self.__beam = ray_list
    def set_ray_pos(self, ray_pos_list): self.__beam_pos = ray_pos_list
    
    def plot(self, ax = None):
        
        """
        Parameters
        ----------
        ax : matplotlib.axes.Axes, used to populate externally created plots = None

        Returns
        -------
        if ax = None
            fig : matplotlib.figure.Figure, outputs the figure created within the function
        else:
            populates the input axes which is caried to the parent figure

        Plots the initial spread of the beam, i.e. the input plane
        """

        if ax == None:
            fig = plt.figure(figsize=(10,10))
            for ray in self.bundle():
                
                plt.plot(ray.vertices()[0][0], ray.vertices()[0][1], "o", color = ray.color())
                plt.title("Input Bundle")
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
            return fig
        else:
            for ray in self.bundle():
                ax.plot(ray.vertices()[0][0], ray.vertices()[0][1], "o", color = ray.color())
                ax.set_title("Input Bundle (b)")
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
    
    
    def make_bundle(ray_list):
        
        """
        Parameters
        ----------
        ray_list : list, a list of arbitrary rays
        
        Returns
        -------
        bundle : RayBundle, a new bundle populated with the input rays

        Makes a bundle using a list of arbitrary rays.
        """
        
        bundle = RayBundle(None, None, empty = True)
        bundle.set_bundle(ray_list)
        
        ray_pos = []
        
        for ray in ray_list:
            ray_pos.append(ray.p())
        
        bundle.set_ray_pos(ray_pos)
        return bundle
            
        
    

                
            
            

    