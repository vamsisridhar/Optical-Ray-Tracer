import ray as r
import optical_element as oe
from optical_system import OpticalSystem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib.image as mpimg
import skimage.transform as transf
# disables unnessary warnings
import warnings
import scipy.optimize as sop
warnings.filterwarnings("ignore")
import sys
curvature = 0.02
thickness = 5
n = 1.5168
z0 = 100


print("CONVEX-PLANO ORIENTATION")

pc1 = oe.PlanoConvex(z0, thickness, 1, n, curvature)

op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

optical_elements = [ pc1 , op]
#uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
bundle = r.RayBundle(10,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)

# Optical System object propagated to create plots
os1 = OpticalSystem(optical_elements, bundle)
focus = os1.output_plane_to_focus()
print(f"Paraxial focus from simulation: {focus} mm")
os1.propagate()
rms = os1.optical_elements()[-1].calculate_RMS_spot_radius()
print(f"RMS of the bundle spread at the output plane: {rms:.4g} mm")

fig, ax = os1.plot_imaging_system(f"Convex-Plano Lens - curvature: 0.02mm$^{{-1}}$, thickness: 5mm, $n_{{glass}}$:1.5168", ylimit=(-30,30))

fig.savefig("plots/Task_15_1.png")
fig.show()


print("PLANO-CONVEX ORIENTATION")


pc2 = oe.PlanoConvex(z0, thickness, 1, n, -curvature, False)

op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

optical_elements = [ pc2 , op]
#uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
bundle = r.RayBundle(10,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)

# Optical System object propagated to create plots
os1 = OpticalSystem(optical_elements, bundle)
focus = os1.output_plane_to_focus()
print(f"Paraxial focus from simulation: {focus} mm")
os1.propagate()
rms = os1.optical_elements()[-1].calculate_RMS_spot_radius()
print(f"RMS of the bundle spread at the output plane: {rms:.4g} mm")

fig, ax = os1.plot_imaging_system(f"Plano-Convex Lens - curvature: 0.02mm$^{{-1}}$, thickness: 5mm, $n_{{glass}}$:1.5168", ylimit=(-30,30))

fig.savefig("plots/Task_15_2.png")
fig.show()

rad_list = np.linspace(0.1, 10, 100)
rms1 = []
rms2 = []


print("Calculating RMS values for Convex-Plano orientation for bundle size 10mm")
for rad in rad_list:
    op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

    optical_elements = [ pc1 , op]
    #uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
    bundle = r.RayBundle(rad,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)
    
    # Optical System object propagated to create plots
    os1 = OpticalSystem(optical_elements, bundle)
    focus = os1.output_plane_to_focus()
    #print(f"Paraxial focus from simulation: {focus} mm")
    os1.propagate()
    rms1.append(os1.optical_elements()[-1].calculate_RMS_spot_radius())
    sys.stdout.write('.'); sys.stdout.flush()
print("\n")


print("Calculating RMS values for Plano-Convex orientation for bundle size 10mm")
for rad in rad_list:
    op = oe.OutputPlane(200) # position the output plane on the focal point of sr1

    optical_elements = [ pc2 , op]
    #uniformly distributed bundle of radius 10, 2 rings, in the +z dir, centred in the xy plane
    bundle = r.RayBundle(rad,5, k = [0,0,1], centre = [0,0,0], ring1_ray_count=6)
    
    # Optical System object propagated to create plots
    os1 = OpticalSystem(optical_elements, bundle)
    focus = os1.output_plane_to_focus()
    #print(f"Paraxial focus from simulation: {focus} mm")
    os1.propagate()
    rms2.append(os1.optical_elements()[-1].calculate_RMS_spot_radius())
    sys.stdout.write('.'); sys.stdout.flush()
print("\n")    


fig, ax = plt.subplots(figsize = (10,10))
ax.set_title("RMS of beam spread at output of Convex + Plano combinations \nfor varying beam radius")
ax.set_xlabel("Beam Radius (mm)")
ax.set_ylabel("RMS of beam spread (mm)")
ax.plot(rad_list, np.array(rms1), label = "Convex-Plano")
ax.plot(rad_list, np.array(rms2), label = "Plano-Convex")
ax.legend()
fig.savefig("plots/Task_15_3.png")
fig.show()

print("PROPAGATING LORD BLACKETT THROUGH CONVEX PLANAR")
img  = mpimg.imread('lord_blackett.jpg')
#lowers the resolution of the image
img = transf.rescale(img, 0.20, anti_aliasing=False)
#plt.imshow(img)

output_plane = oe.OutputPlane(300)
optical_elements = [pc1, output_plane]

ray_list = []
width = len(img[0])
height = len(img)
for i in range(len(img)):
    for j in range(len(img[i])):
        ray = r.Ray([((-width/2) + j) * 0.1,((height) - i) * 0.1, 0], [0,0,1],color=[0,img[i,j], 0])
        ray_list.append(ray)

bundle = r.RayBundle.make_bundle(ray_list)

os1 = OpticalSystem(optical_elements,bundle)
os1.propagate()
print("Please wait a few moments while the plot is being processed...")
print("The plot is quite graphic intensive...")
#print(sp_elem_1.hits())
fig, ax = os1.plot_imaging_system(f"Propagating an Image of Lord Blackett through a convex-plano singlet\n curvature: 0.02mm$^{{-1}}$, thickness: 5mm, $n_{{glass}}$:1.5168 (a)", switch_axes = True,ylimit=(-30,30))
fig.savefig("plots/Task_15_4.png")
fig.show()


print("PROPAGATING LORD BLACKETT THROUGH PLANAR CONVEX")

output_plane = oe.OutputPlane(300)
optical_elements = [pc2, output_plane]


ray_list = []
width = len(img[0])
height = len(img)
for i in range(len(img)):
    for j in range(len(img[i])):
        ray = r.Ray([((-width/2) + j) * 0.1,((height) - i) * 0.1, 0], [0,0,1],color=[0,img[i,j], 0])
        ray_list.append(ray)

bundle = r.RayBundle.make_bundle(ray_list)

os1 = OpticalSystem(optical_elements,bundle)
os1.propagate()
print("Please wait a few moments while the plot is being processed...")
print("The plot is quite graphic intensive...")
#print(sp_elem_1.hits())
fig, ax = os1.plot_imaging_system(f"Propagating an Image of Lord Blackett through a plano-convex singlet\n curvature: 0.02mm$^{{-1}}$, thickness: 5mm, $n_{{glass}}$:1.5168 (a)", switch_axes = True, ylimit=(-30,30))
fig.savefig("plots/Task_15_5.png")
fig.show()


def rms_of_curvatures(c, focal_length = 200, beam_radius = 10, beam_dir = [0,0,1], beam_centre= [0,0,0]):

    bsr= oe.BiSphericalRefractor(20, 10,1, 1.5182, c[0], c[1])
    
    op1 = oe.OutputPlane(focal_length) # position the output plane on the focal point of sr1
    optical_elements1 = [ bsr, op1 ]
    bundle1 = r.RayBundle(beam_radius,5, k = beam_dir, centre = beam_centre)

    #bundle.plot().show()
    
    
    #r.plot_imaging_system_3d(optical_elements, bundle.bundle(), "Ray Bundle Refraction at Spherical Lens").show()
    os1 = OpticalSystem(optical_elements1, bundle1)
    os1.propagate()
    #os1.plot_imaging_system("System").show()

    
    #optical_elements[-1].plot_output_plane().show()
    
    
    return op1.calculate_RMS_spot_radius()

def callback(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))

rms_opt = sop.fmin_tnc(rms_of_curvatures, np.array([0.02, -0.02]), args = (150, 10), approx_grad=True)

# NOTE: this was optimising correctly but for some random reason broke the a day before submission.

print(rms_opt)
bsr1= oe.BiSphericalRefractor(100, 5,1, 1.5182, rms_opt[0][0], rms_opt[0][1])

bundle = r.RayBundle(3, 5, k = [0,0,1], centre = [0,0,0])

op = oe.OutputPlane(200)
optical_elements = [bsr1, op]

os1 = OpticalSystem(optical_elements, bundle)
os1.propagate()

rms = os1.optical_elements()[-1].calculate_RMS_spot_radius()
print(f"RMS of the bundle spread at the output plane: {rms:.4g} mm")

fig, ax = os1.plot_imaging_system(f"Bi-Convex Lens with curvatures: {rms_opt[0][0]:.4g} mm$^{{-1}}$ and {rms_opt[0][1]:.4g} mm$^{{-1}}$")
fig.savefig("plots/TASK_A.png")

fig.show()

