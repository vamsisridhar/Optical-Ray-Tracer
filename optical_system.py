import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import optical_element as oe
from ray import Ray, RayBundle
from scipy.optimize import fmin_tnc
import matplotlib as mpl
class OpticalSystem:
    def __init__(self, optical_elements, bundle):
        
        if not isinstance(optical_elements[-1], oe.OutputPlane):
            raise Exception(f"{oe.OutputPlane} not found at the end of optical_elements list, instead found {type(optical_elements[-1])}")
        self.__optical_elements = optical_elements
        self.__rays = bundle.bundle()
        self.__bundle = bundle
        self.__propagated = False
    
    
    def optical_elements(self): return self.__optical_elements
    def rays(self): return self.__rays
    def bundle(self): return self.__bundle
    def propagated(self): return self.__propagated
    
    def set_propagated(self, state): self.__propagated = state
    
    def propagate(self):
        #print(self.optical_elements())
        for elem in self.optical_elements():
            for ray in self.rays():

                elem.propagate_ray(ray)
        
        self.set_propagated(True)
        
    def output_plane_to_focus(self, guess_focus = 10000, paraxial_range = 0.1):
        def optimize_focal_point(focus):
            ray_list = []
            # a set of rays within the paraxial range
            for i in np.linspace(-paraxial_range/2, paraxial_range/2, 10):
                ray_list.append(Ray([i,0,0], [0,0,1]))
                
            bundle = RayBundle.make_bundle(ray_list)
            optical_elements = self.optical_elements()[:-1]
            op = oe.OutputPlane(focus)
            optical_elements.append(op)
            os = OpticalSystem(optical_elements, bundle)
            os.propagate()
            return op.calculate_RMS_spot_radius()
            #calculate the average focal point for the rays
        opt = fmin_tnc(optimize_focal_point, guess_focus, bounds=((self.optical_elements()[-2].z0(), np.inf),), approx_grad=True, disp = 0)
        
        focus = opt[0][0]
        self.optical_elements()[-1] = oe.OutputPlane(focus)
        return focus - self.optical_elements()[0].z0()

    def plot_imaging_system(self, title, plane_plots = True, switch_axes = False, zoom =False, ylimit = None, figure_size = None, font_size = 15):
        mpl.rcParams.update({'font.size': font_size})
        if not self.propagated():
            raise Exception("Implement propagate() for the system!")
        
        y_axis = 0
        if switch_axes:
            y_axis = 1
        
        if plane_plots:
            fig = plt.figure(figsize = (12,15))

            gs = gridspec.GridSpec(3, 2, hspace = 0.3, wspace = 0.4, top = 0.94, bottom = 0.08)
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            ax2 = fig.add_subplot(gs[4])
            ax3 = fig.add_subplot(gs[5])
            
            for ray in self.rays():
                ax1.plot(np.array(ray.vertices())[:, 2], np.array(ray.vertices())[:, y_axis], color = ray.color())
                ax1.plot(np.array(ray.vertices())[:, 2], np.array(ray.vertices())[:, y_axis],'x',  color = ray.color())
            
            
            ylim = np.max([np.abs(np.max(np.array(ray.vertices())[:, y_axis])), np.abs(np.min(np.array(ray.vertices())[:, 2]))]) * 1.5
            
            ax1.set_title(title)
            ax1.set_xlabel("z (mm)")
            if switch_axes:
                ax1.set_ylabel("y (mm)")
            else:
                ax1.set_ylabel("x (mm)")
            if zoom:
                ax1.set_ylim([-ylim, +ylim])
            elif ylimit != None:
                ax1.set_ylim([ylimit[0], ylimit[1]])
            ax1.set_xlim([0,self.optical_elements()[-1].z0()])
            ax1.grid()
            
            for elem in self.optical_elements():
                if not isinstance(elem, oe.OutputPlane):
                    elem.plot(ax1)

            
            self.bundle().plot(ax2)
            self.optical_elements()[-1].plot_output_plane(ax3)
            
            
            return fig, (ax1,ax2,ax3)
            
        else:
            if figure_size == None:
                fig, ax1 = plt.subplots(figsize=(10,10))
            else:
                fig, ax1 = plt.subplots(figsize=figure_size)

            for ray in self.rays():
                ax1.plot(np.array(ray.vertices())[:, 2], np.array(ray.vertices())[:, y_axis], color = ray.color())
                ax1.plot(np.array(ray.vertices())[:, 2], np.array(ray.vertices())[:, y_axis],'x',  color = ray.color())
            ax1.set_title(title)
            ax1.set_xlabel("z (mm)")
            if switch_axes:
                ax1.set_ylabel("y (mm)")
            else:
                ax1.set_ylabel("x (mm)")
            ylim = np.max([np.abs(np.max(np.array(ray.vertices())[:, y_axis])), np.abs(np.min(np.array(ray.vertices())[:, 2]))]) * 1.5
            if zoom:
                ax1.set_ylim([-ylim, +ylim])
            elif ylimit != None:
                ax1.set_ylim([ylimit[0], ylimit[1]])
            
            for elem in self.optical_elements():
                if not isinstance(elem, oe.OutputPlane):
                    elem.plot(ax1)
        
            return fig, ax1 

    def plot_imaging_system_3d(self, title):
                
        fig = plt.figure(figsize = (10,10))
        ax1 = fig.add_subplot(projection='3d')
        
        for ray in self.rays():
            X = np.array(ray.vertices())[:, 0]
            Y = np.array(ray.vertices())[:, 1]
            Z = np.array(ray.vertices())[:, 2]
            
            ax1.plot3D(X, Y, Z)# Data for three-dimensional scattered points
            ax1.scatter3D(X, Y, Z, c=Z, cmap='Blues')
        ax1.set_title(title)
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        ax1.set_zlabel("z (mm)")
        return fig, ax1