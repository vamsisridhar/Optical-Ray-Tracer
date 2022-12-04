import ray as r
import optical_element as oe
from optical_system import OpticalSystem


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as transf
# disables unnessary warnings
import warnings
warnings.filterwarnings("ignore")
import sys
import pandas as pd
"""
This python file investigates the working of a single spherical refractor
"""

#Params for the Spherical Refractor
apper_size = 2 * (1/0.03)
out_z0 = 250
lens_z0 = 100
lens_curv = 0.03
n1 = 1
n2 = 1.5


#Initialising the Spherical Refractor
sr1 = oe.SphericalRefractor(lens_z0, lens_curv, n1, n2, apper_size)

theoretical_focus = sr1.spherical_surface_focus()
print(f"Theoretical focus of the Spherical Surface: {theoretical_focus} mm")

"""
------------------------------------------------------------------------------

TASK 9
"""
print("TASK 9")
output_plane = oe.OutputPlane(out_z0) # output plane
optical_elements = [sr1, output_plane] # optical_elements


ray_list = []

# A list of interesting ray cases for the Task 9 Plot
paraxial_colour = [1,0,0]   #Paraxial Ray
abberated_colour = [0,0,1]  #Spherically Aberrated Ray
tangent_colour = [0,1,0]    #Tangential Ray
ofb_colour = [1,0.5,0]      #Out Of Bounds Ray
normal_colour = [0,1,1]     #Normal Ray
colour_labels = ['paraxial', 'spherically abberated', 'tangential', 'out of bounds', 'normal']
fake_lines = [
    plt.Line2D([0,0],[0,1], color=[1,0,0], linestyle='-'),
    plt.Line2D([0,0],[0,1], color=[0,0,1], linestyle='-'),
    plt.Line2D([0,0],[0,1], color=[0,1,0], linestyle='-'),
    plt.Line2D([0,0],[0,1], color=[1,0.5,0], linestyle='-'),
    plt.Line2D([0,0],[0,1], color=[0,1,1], linestyle='-')]
r1 = r.Ray([0,0,0], [0,0,1], color =normal_colour)
r2 = r.Ray([0,0,0], [1,0,1], color=ofb_colour)
r3 = r.Ray([0,0,0], [-1,0,1],color=ofb_colour)
r4 = r.Ray([10,0,0], [0,0,1],color=paraxial_colour)
r5 = r.Ray([-10,0,0], [0,0,1], color=paraxial_colour)
r6 = r.Ray([apper_size/2,0,0], [0,0,1], color=tangent_colour)
r7 = r.Ray([-apper_size/2,0,0], [0,0,1], color=tangent_colour)
r8 = r.Ray([30,0,0], [0,0,1], color=abberated_colour)
r9 = r.Ray([-30,0,0], [0,0,1], color=abberated_colour)
r10 = r.Ray([5,0,0], [0,0,1],color=paraxial_colour)
r11 = r.Ray([-5,0,0], [0,0,1], color=paraxial_colour)
ray_list = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11]

# makes a bundle object from a list of rays
bundle = r.RayBundle.make_bundle(ray_list)
# create OpticalSystem object using the optical_elements and the bundle
os1 = OpticalSystem(optical_elements,bundle)
os1.propagate() # propagate
fig, ax = os1.plot_imaging_system(f"Task 9 Plot - curvature: {lens_curv}$mm^{-1}$, z0: {lens_z0}$mm$",
                                  plane_plots = False,
                                  ylimit = [-1/0.03 * 1.5, 1/0.03 * 1.5],
                                  figure_size = (20,10),
                                  font_size = 25) #creates plots

ax.legend(handles = fake_lines, labels = colour_labels, loc = 4)
ax.text(0, 40,"$n_1 = 1$")
ax.text(out_z0 - 20, 40,"$n_2 = 1.5$")
fig.savefig("plots/Task_9_1.png")
fig.show() #show plots
print("-------------------------------------------------")

print("Investigating Paraxial Rays through the Spherical Refractor")
op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

optical_elements = [ sr1 , op]
#uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
bundle = r.RayBundle(0.1,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)

# Optical System object propagated to create plots
os1 = OpticalSystem(optical_elements, bundle)
focus = os1.output_plane_to_focus()
print(f"Paraxial focus from simulation: {focus} mm")
os1.propagate()
rms = os1.optical_elements()[-1].calculate_RMS_spot_radius()
print(f"RMS of the bundle spread at the output plane: {rms:.4g} mm")

fig, ax = os1.plot_imaging_system(f"Single Spherical Refractor - curvature: {lens_curv}$mm^{{-1}}$, z0: {lens_z0}$mm$ (a)", zoom = True)

fig.savefig("plots/Task_10_1.png")
fig.show()

fig, ax = os1.plot_imaging_system_3d(f"Single Spherical Refractor - curvature: {lens_curv}$mm^{{-1}}$, z0: {lens_z0}$mm$")
fig.savefig("plots/Task_10_2.png")
fig.show()

print("-------------------------------------------------")

print("Investigating Large Bundles through the Spherical Refractor")
op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

optical_elements = [ sr1 , op]
#uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
bundle = r.RayBundle(10.0,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)

# Optical System object propagated to create plots
os1 = OpticalSystem(optical_elements, bundle)
os1.output_plane_to_focus()
os1.propagate()
rms = os1.optical_elements()[-1].calculate_RMS_spot_radius()
print(f"RMS of the bundle spread at the output plane: {rms:.4g} mm")

fig, ax = os1.plot_imaging_system(f"Single Spherical Refractor - curvature: {lens_curv}$mm^{{-1}}$, z0: {lens_z0}$mm$ (a)", zoom = True)
fig.savefig("plots/Task_11_1.png")
fig.show()




img  = mpimg.imread('lord_blackett.jpg')
#lowers the resolution of the image
img = transf.rescale(img, 0.20, anti_aliasing=False)
#plt.imshow(img)


output_plane = oe.OutputPlane(lens_z0 + (2*focus))
optical_elements = [sr1, output_plane]


ray_list = []
width = len(img[0])
height = len(img)
for i in range(len(img)):
    for j in range(len(img[i])):
        ray = r.Ray([((-width/2) + j) * 0.01,((height) - i) * 0.01, 0], [0,0,1],color=[0,img[i,j], 0])
        ray_list.append(ray)

bundle = r.RayBundle.make_bundle(ray_list)

os1 = OpticalSystem(optical_elements,bundle)
os1.propagate()
print("Please wait a few moments while the plot is being processed...")
print("The plot is quite graphic intensive...")
#print(sp_elem_1.hits())
fig, ax = os1.plot_imaging_system(f"Propagating an Image of Lord Blackett through the lens at 2f\n curvature: {lens_curv}$mm^{{-1}}$, z0: {lens_z0}$mm$ (a)", switch_axes = True, ylimit = (-5, 5))
fig.savefig("plots/Task_11_2.png")
fig.show()


img  = mpimg.imread('lord_blackett.jpg')
#lowers the resolution of the image
img = transf.rescale(img, 0.20, anti_aliasing=False)
#plt.imshow(img)


output_plane = oe.OutputPlane(lens_z0 + focus)
optical_elements = [sr1, output_plane]


ray_list = []
width = len(img[0])
height = len(img)
for i in range(len(img)):
    for j in range(len(img[i])):
        ray = r.Ray([((-width/2) + j) * 0.01,((height) - i) * 0.01, 0], [0,0,1],color=[0,img[i,j], 0])
        ray_list.append(ray)

bundle = r.RayBundle.make_bundle(ray_list)

os1 = OpticalSystem(optical_elements,bundle)
os1.propagate()
print("Please wait a few moments while the plot is being processed...")
print("The plot is quite graphic intensive...")
fig, ax = os1.plot_imaging_system(f"Propagating an Image of Lord Blackett through the lens at f\n curvature: {lens_curv}$mm^{{-1}}$, z0: {lens_z0}$mm$ (a)", switch_axes = True, ylimit = (-5, 5))
fig.savefig("plots/Task_11_3.png")
fig.show()