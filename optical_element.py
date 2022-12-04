# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:11:56 2021

@author: vamsi
"""
    

import numpy as np
from ray import Ray
import vector as vec
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from optical_element import *


class OpticalElement:
    """
    An element that can propagate a ray through it under geometric optics
    
    """
    def propagate_ray(self, ray):
        "propogate a ray through the optical element"
        raise NotImplementedError()
        
    
    def snell(i,  norm, n1, n2):
        """
        Parameters
        ----------
        i : incident ray 3d unit vector
        norm : surface normal 3d unit vector
        n1 : refractive index (from)
        n2 : refractive index (to)

        Returns
        -------
        if sin_i (= ||(n x i)||) > n2/n1:
            the total internal reflected ray, r normalised
        else:
            the refracted ray, r normalised
        
        > Uses a vector form of Snell's law

        """
        mu = n1/n2 # ratio between n1 and n2
        crx = np.cross(norm, i) # cross product of normal and incident
        dot = np.dot(norm, i) # dot product of normal and incident
        sin_i = np.sqrt(crx.dot(crx)) # sin(i) calculated via magnitude of cross product
    
        # condition to check for total internal reflection
        if sin_i > 1/mu:
            # TIR
            r = i - 2*dot*norm
            return r/np.sqrt(r.dot(r))
        else:
            # Snells Law for refraction in vector form
            r = -np.sqrt(1 - (mu**2)*(1-dot**2)) * norm + mu*(i - (dot * norm))
            #return normalised refracted ray
            return r/np.sqrt(r.dot(r))
                    
class SphericalRefractor(OpticalElement):
    
    """
    A spherical refractor, formed as an hemisphere acts as an interface between two refractive indexes n1 and n2.
    
    """
    
    def __str__(self):
        return f"z0: {self.z0()}\ncurvature: {self.curvature()}\nn1: {self.n1()}\nn2: {self.n2()}\napperture: {self.apper()}"
    
    def __init__(self, z0, curvature, n1, n2, r_aper):
        
        """
        Input Parameters
        ----------
        z0 :  float, intercept of the surface with the z-axis
        curvature : float, curvature of the surface
                    mag: 1/(radius of curvature:z)
                    sign: z > z0 +ve
                          z < z0 -ve
        n1 : float, refractive index to the left of lens
        n2 : float, refractive index to the right of lens
        r_aper : float, aperture radius
        
        Local Variables
        ---------
        __radius: float, radius of curvature |1/curvature|
        __centre: float, the centre position on the z-axis calculated by calc_centre
        
        Initialises a Spherical Refractor

        """
        
        self.__z0 = z0
        self.__curvature = curvature
        self.__n1 = n1
        self.__n2 = n2
        self.__r_aper = r_aper
        
        # check if planar
        if self.curvature() != 0:
            # radius is the inverse of curvature
            self.__radius = 1/np.abs(self.curvature())
            # radius is added/subtracted depending on the curvature's polarity
            self.__centre = self.z0() + SphericalRefractor.calc_centre(self.__curvature)
            
        else:
            # radius and centre do not exist for planar
            self.__radius = None
            self.__centre = None
    
    def spherical_surface_focus(self):
        """
        
        Returns
        ------
        The focus based on the spherical surface focus formula

        """
        
        return self.n2()/(self.curvature()*(self.n2()-self.n1()))
    
    def calc_centre(curvature):
        
        """
        Parameters
        ----------
        curvature : float, curvature of refractor
        
        Calculates the centre of curvature

        """
        # the displacement of the centre from z0 varies with the sign of the curvature
        # the centre is alway radius (=1/|curvature|) away from z0
        return np.sign(curvature)*(1/np.abs(curvature))
    
    # getters
    
    def z0(self): return self.__z0
    def curvature(self): return self.__curvature
    def n1(self): return self.__n1
    def n2(self): return self.__n2
    def apper(self): return self.__r_aper
    def radius(self): return self.__radius
    def centre(self): return self.__centre   

    def intercept(self, ray):
        
        """
        Input Parameters
        ----------
        ray : Ray, the ray being intercepted with the refractor
        
        Calculates the intercept between the ray and the refractor
        
        """
        
        if isinstance(ray, Ray):

            """
            O is the centre of curvature
            P is the current position of ray
            Q is the intersection point of the ray and the optical element
            - variable names defined in vector notation 
            
            """
            
            P = ray.p()
            
            # check planar
            if self.curvature() == 0:
                # calculate intersection with plane
                k = vec.normalise(ray.k())
                l = (self.z0() - P[2])/k[2]
                
                # p_z + λ*k_z = z
                
                return P + l*k
            
            O = np.array([0,0,self.centre()])
            # the vector from the ray position to the centre of curvature of the lens
            OP = P-O 
            R = self.radius()
            # dot product of the vector r and the direction vector of the ray
            r_dot_k = np.dot(OP, vec.normalise(ray.k())) 
           
            # the two intersection points for line and sphere intersection        
            # there will always be two intersections if the ray passes through the hypothetical sphere of the spherical lens.
            l1 = -r_dot_k + np.sqrt(r_dot_k**2 -(vec.magnitude(OP)**2 - R**2))
            l2 = -r_dot_k - np.sqrt(r_dot_k**2 -(vec.magnitude(OP)**2 - R**2))
            
            l = 0 # initialising the l variable
            
            # when curvature is greater than 0 then the lower value for the intersection is taken else the higher value.
            if self.curvature() > 0:
                if l1 > l2:
                    l = l2
                elif l1 < l2:
                    l = l1
                else:
                    return None
            elif self.curvature() < 0:
                if l1 > l2:
                    l = l1
                elif l1 < l2:
                    l = l2
                else:
                    return None
            
            Q = P + l*vec.normalise(ray.k()) # calculates the intersection point.
            return Q
        
    def plot(self, ax):
        
        """
        Input Parameters
        ----------
        ax : matplotlib.axes.Axes, external axes sent for population 
        
        Creates a patch matching curvature of the lens, so it can be displayed in the 2d plots
        
        """

        if self.curvature() == 0:
            #Planar lens, a straight line is created between the apperture limits
            patch = ptch.ConnectionPatch((self.z0(), -self.apper()/2), (self.z0(), self.apper()/2), "data", linewidth = 5)
            
        else:
            
            # The sign of the curvature influences the orientation of the Spherical lens
            if self.curvature() < 0:
                #curved surface faces right
                angle = 0
            elif self.curvature() > 0:
                #curved surface faces left
                angle = 180
            
            theta = np.arcsin((self.apper()*0.5)/self.radius()) * 180/np.pi
            #Arc patch created to represent the spherical lens in 2d
            patch = ptch.Arc(xy = (self.centre(), 0),
                         width = self.radius()*2,
                         height = self.radius()*2,
                         angle = angle,
                         theta1 = -theta,
                         theta2 = theta,
                         linewidth = 5)
            
        ax.add_patch(patch) # patch added to axes
        
    def propagate_ray(self, ray):
        
        """
        Parameters
        ----------
        ray : Ray, ray object being propagated through the optical element

        """
        Q = self.intercept(ray) 
        # check if the intersection point exists and is within the apperature range
        if not isinstance(Q, np.ndarray) or np.abs(Q[1]) > self.apper()/2 or np.abs(Q[0]) > self.apper()/2:
            return
        
        # incident vector to intersection
        incident = Q - ray.p()
        
        if self.curvature() == 0:
            # planar case
            norm = np.array([0,0,-1]) # plane normal
            # snells
            ref_dir = OpticalElement.snell(vec.normalise(incident), vec.normalise(norm), self.n1(), self.n2())
            
        else:
            # spherical case
            norm =  Q - [0, 0, self.centre()] # spherical normal
            #snells
            ref_dir = OpticalElement.snell(vec.normalise(incident), np.sign(self.curvature())* vec.normalise(norm), self.n1(), self.n2())
        
        # append the new ray
        if isinstance(ref_dir, type(None)):
            return
        ray.append(Q, ref_dir)



class BiSphericalRefractor(OpticalElement):
    """
    Biconvex Lens Setup, a positive curvature + a negative curvature (can be 0 curvature)

    """
    
    def __str__(self):
        return f"z: {self.z0()}\nthickness: {self.thickness()}\ncurvature: {self.curvature()}\nn1: {self.n1()}\nn2: {self.n2()}\napperture: {self.apper()}"
    
    def __init__(self, z0, thickness, n1, n2, curvature1, curvature2):
        
        """
        Input Parameters
        ----------
        z0 : float, the point where the left most part of the lens crosses the z-axis
        thickness : thickness between the two z0 values of the refractors
        n1 : refractive index outside the refractors
        n2 : refractive index between the refractors
        curvature1 : curvature of the left refractor (positive)
        curvature2 : curvature of the right refractor (negative)
        
        BiConvex Lens Setup 


        """
        self.__z0 = z0
        if thickness <= 0:
            raise Exception("Thickness must be positive!")

        self.__thickness = thickness
        self.__n1 = n1
        self.__n2 = n2
        
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        
        
        """
        If one of the curvature is 0:
            First check if the plane intersects the spherical hemisphere of the other curvature.
            If it does: use pythagoras to determine apperture
            Else: Set apperture to the diameter of non-0 curvature
        Else:
            d - the distance between the two centre of curvatures
            if d is greater than the sum of the two radius of curvatures:
                set apperture to the smaller diameter of the two curvatures
            else:
                d1 - the horizontal distance from the center of curvature 2 to the intersection point of the two hemispheres
                if d1 is greater than the radius of curvature 2:
                    set apperture to the smaller diameter of the two curvatures
                else: use pythagoras to determine apperture
        """
        
        if self.__curvature1 == 0:
            self.__apper = 2*np.sqrt((1/np.abs(self.__curvature2))**2 - ((1/np.abs(self.__curvature2)) - self.__thickness)**2)
            # apper will be nan if the planar surface does not intersect with the curved surface
            if np.isnan(self.__apper):
                self.__apper = 2/self.__curvature1
            
        elif self.__curvature2 == 0:
            self.__apper = 2*np.sqrt((1/np.abs(self.__curvature1))**2 - ((1/np.abs(self.__curvature1)) - self.__thickness)**2)
            if np.isnan(self.__apper):
                self.__apper = 2/self.__curvature1

        else:
            d = SphericalRefractor.calc_centre(self.__curvature1) - (SphericalRefractor.calc_centre(self.__curvature2) + self.__thickness)

            if d >= (1/np.abs(self.__curvature2) + 1/np.abs(self.__curvature1)):
                self.__apper = np.min([2/np.abs(self.__curvature1), 2/np.abs(self.__curvature2)])

            else:

                d1 = (d**2 - (1/np.abs(self.__curvature1))**2 + (1/np.abs(self.__curvature2))**2)/(2*d)
                if (1/np.abs(self.__curvature2) < d1):
                    self.__apper = np.min([2/np.abs(self.__curvature1), 2/np.abs(self.__curvature2)])

                else:
                    self.__apper = 2*np.sqrt((1/self.__curvature2)**2 - (d1)**2)    


        #initialisation of the lens
        #the first lens with postive curvature
        #the second lens with negative curvature
        self.__lens1 = SphericalRefractor(self.__z0, self.__curvature1, self.__n1,  self.__n2, self.__apper)
        self.__lens2 = SphericalRefractor(self.__z0+self.__thickness, self.__curvature2,  self.__n2, self.__n1, self.__apper)
        
    def thin_lens_focus(self):
        """
        Returns
        ------
        The focus based on the thin lens formula

        """
        return 1/(((self.n2()/self.n1()) - 1)*(self.curvature1() - self.curvature2()))
    #getters
    
    def z0(self): return self.__z0
    def thickness(self): return self.__thickness
    def n1(self): return self.__n1
    def n2(self): return self.__n2
    def curvature1(self): return self.__curvature1
    def curvature2(self): return self.__curvature2
    def apper(self): return self.__apper
    def lens1(self): return self.__lens1
    def lens2(self): return self.__lens2
    
    def propagate_ray(self, ray):
        """
        Parameters
        ----------
        ray : Ray, ray object being propagated through the optical element
        
        propagates the ray through two lens

        """
        self.lens1().propagate_ray(ray)
        self.lens2().propagate_ray(ray)
    
    def plot(self, ax):
        
        """
        Input Parameters
        ----------
        ax : matplotlib.axes.Axes, external axes sent for population 
        
        Calls the plot functions for both lens to create the respective patches
        
        """
        
        self.lens1().plot(ax)
        self.lens2().plot(ax)# -*- coding: utf-8 -*-

class PlanoConvex(BiSphericalRefractor):
    """
    Plano-Convex Lens Setup, a 0 curvature and a non-0 curvature

    """
    def __init__(self, z0, thickness, n1, n2, curvature, plane_second = True):
        """
        

        Parameters
        ----------
        z0 : float, the point where the left most part of the lens crosses the z-axis
        thickness : thickness between the two z0 values of the refractors
        n1 : refractive index outside the refractors
        n2 : refractive index between the refractors
        curvature : float, the curvature of the curved surface
        plane_second : bool, (True) the planar surface is treated as the second curvature
                        or (False) the planar surface is treated as the first curvature
                        this also affects the sign convention of curvature based of the BiSphericalRefractor

        """
        if plane_second:
            super().__init__(z0, thickness, n1, n2, curvature, 0)
        else:
            super().__init__(z0, thickness, n1, n2, 0, curvature)# -*- coding: utf-8 -*-


class OutputPlane(OpticalElement):
    """
    The Output Plane propagates the end progations of the rays and finishes the optical system
    """
    
    def __init__(self, z0):
        
        """
        Input Parameters
        ----------
        z0 : float, the point where the left most part of the lens crosses the z-axis
        
        Hidden Variables
        ----------
        __output_plane_rays : list, stores a list of rays that are propagated to the output plane
        
        Initialises the Output Plane
        
        """
        self.__z0 = z0
        self.__output_plane_rays = []
        
    def intercept(self, ray):
        
        """
        Parameters
        ----------
        ray : Ray, ray being propagated to the Output Plane
        
        Calculates the intercept to the Output Plane

        """
        if isinstance(ray, Ray):
            
            P = ray.p()
            k = vec.normalise(ray.k())
            
            # First check if the ray is moving towards output plane
            if k[2] > 0:
                #propagate to output plane
                l = (self.__z0 - P[2])/k[2]
                #if ray hits the output plane, append it to the output plane rays
                self.output_plane_rays().append(ray)
            elif k[2] < 0:
                #If the ray is going towards the input plane
                #Propagate it to z=0
                l = (0 - P[2])/k[2]
            else:
                #If ray direction is perpendicular to z-axis, terminate it
                l = 0
            # p_z + λ*k_z = z
            return P + l*k
        
    # getters
    
    def z0(self): return self.__z0
    def output_plane_rays(self): return self.__output_plane_rays
        
    def propagate_ray(self, ray):
        """
        Parameters
        ----------
        ray : Ray, ray object being propagated through the optical element
        
        propagates the ray to Output Plane

        """
        Q = self.intercept(ray)
        ray.append(Q, ray.k())
        
    
    def calculate_RMS_spot_radius(self):
        """
        

        Returns
        -------
        RMS : float, the root mean square of the radii of the ray points in the x,y plane from the origin

        Calculates the RMS spot radius of the rays at the output plane

        """
        if len(self.output_plane_rays()) != 0:
            radius_list = [] #stores radius of the ray points in the x,y plane from the origin

            for ray in self.output_plane_rays():
                #calculate radius
                radius = vec.magnitude(ray.p()[0:2])
                radius_list.append(radius)
            
            radius_list = np.array(radius_list) - np.array(radius_list).mean()
            #calculate radius ^ 2
            radius_list_sqr = np.array([r**2 for r in radius_list ])
            #calculate RMS
            RMS = np.sqrt(np.array(radius_list_sqr).mean())
            #print(RMS)
            return RMS
    
    def plot_output_plane(self, ax=None):
        """
        

        Parameters
        ----------
        ax : matplotlib.axes.Axes, external axes for population

        Returns
        -------
        fig : matplotlib.figure.Figure, figure containing the spot diagram of the points in the output plane

        """
        ray_list = np.array(self.output_plane_rays())

        if ax == None:
            fig = plt.figure(figsize = (10,10))
            plt.title("Output Plane")
            plt.set_xlabel("x (mm)")
            plt.set_ylabel("y (mm)")
            for ray in ray_list:
                plt.plot(ray.p()[0], ray.p()[1], 'o', color = ray.color())
            
            return fig
        else:
            ax.set_title("Output Plane (c)")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            for ray in ray_list:
                ax.plot(ray.p()[0], ray.p()[1], 'o', color = ray.color())
        # -*- coding: utf-8 -*-



if __name__ == '__main__':
    
    # Check Intercept Spherical:
    sp1 = SphericalRefractor(12+5-(5*np.sqrt(2)), 1/(5*np.sqrt(2)), 1, 1.5, 10)
    r1 = Ray(k = [5, 0, 12])
    print(np.round(vec.magnitude(sp1.intercept(r1))))
    if np.round(vec.magnitude(sp1.intercept(r1))) == 13:
        print("Spherical Ray is intercepting correctly")    
   
 

    
    
    
   
        
__all__ = ['OpticalElement']


        