__author__ = 'Kryosugarra'
import numpy
import enum
import misc

# class Shape(enum.Enum):
#     Sphere = 1
#     Cube = 2
#     Cyllinder = 3
#     Spheroid = 4
#
# class Scatterer:
#     def __init__(self):
#         self.dipoles = []
#         self.dipole_spacing = 0
#
#     @staticmethod
#     def load_dda_si_dipole_file(filename):
#         f = open(filename, 'r')
#         lines = f.readlines()
#         dipoles = []
#         for line in lines:
#             dipole = []
#             for d in line.split(','):
#                 dipole.append(float(d))
#             dipoles.append(dipole)
#
#         scat = Scatterer()
#         scat.dipoles = numpy.asarray(dipoles)
#         return scat
#
#     @staticmethod
#     def scatterer_from_shape(shape, dipoles_per_dimension=16, **kwargs):
#         if shape == Shape.Sphere:
#             return Scatterer.dipole_sphere(**kwargs)
#         else:
#             raise Exception("Unknown shape")
#
#
#     @staticmethod
#     def dipole_sphere(dipoles_per_dimension, radius):
#         pow2 = misc.power_function(2)
#         pow3 = misc.power_function(3)
#         pow1d3 = misc.power_function(1./3.)
#
#         scat = Scatterer()
#         dipoles = []
#         for x in numpy.linspace(-radius, radius, dipoles_per_dimension):
#             for y in numpy.linspace(-radius, radius, dipoles_per_dimension):
#                 for z in numpy.linspace(-radius, radius, dipoles_per_dimension):
#                     if numpy.sqrt(pow2(x) + pow2(y) + pow2(z)) <= radius:
#                         dipoles.append([x, y, z])
#         scat.dipoles = numpy.asarray(dipoles)
#         #scat.dipole_spacing = 2*radius / (dipoles_per_dimension-1)
#         scat.dipole_spacing = pow1d3(4 / 3 * numpy.pi / len(dipoles)) * radius
#         scat.dipoles *= scat.dipole_spacing
#
#         print(pow3(scat.dipole_spacing)*len(dipoles), (4/3)*numpy.pi*pow3(radius))
#
#         return scat
#
#     @staticmethod
#     def dipole_cube(dipoles_per_dimension, radius):
#         scat = Scatterer()
#         dipoles = []
#         for x in numpy.linspace(-radius, radius, dipoles_per_dimension):
#             for y in numpy.linspace(-radius, radius, dipoles_per_dimension):
#                 for z in numpy.linspace(-radius, radius, dipoles_per_dimension):
#                         dipoles.append([x, y, z])
#         scat.dipoles = numpy.asarray(dipoles)
#         return scat
#
# def rescale_scatterer(scat, a_eff):
#     pow2 = misc.power_function(2)
#     pow3 = misc.power_function(3)
#     pow1d3 = misc.power_function(1./3.)
#
#     scat2 = Scatterer()
#     scat2.dipoles = scat.dipoles.copy()
#
#     #scat2.dipoles /= scat2.dipole_spacing
#     scat2.dipole_spacing = pow1d3(4 / 3 * numpy.pi / scat2.dipoles.shape[0]) * a_eff
#     scat2.dipoles *= scat2.dipole_spacing
#
#     return scat2

def dipole_sphere(dipoles_per_dimension, radius):
    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)
    pow1d3 = misc.power_function(1./3.)

    dipoles = []
    for x in numpy.linspace(-radius, radius, dipoles_per_dimension):
        for y in numpy.linspace(-radius, radius, dipoles_per_dimension):
            for z in numpy.linspace(-radius, radius, dipoles_per_dimension):
                if numpy.sqrt(pow2(x) + pow2(y) + pow2(z)) <= radius:
                    dipoles.append([x, y, z])

    initial_spacing = numpy.average(numpy.diff(numpy.linspace(-radius, radius, dipoles_per_dimension)))

    dipole_spacing = pow1d3(4 / 3 * numpy.pi / len(dipoles)) * radius
    dipoles = numpy.asarray(dipoles)*(dipole_spacing/initial_spacing)

    #print(pow3(dipole_spacing)*(dipoles.shape[0]), (4/3)*numpy.pi*pow3(radius))

    return dipoles, dipoles.shape[0], dipole_spacing

def dipole_cube(dipoles_per_dimension, side):
    dipoles = []
    for x in numpy.linspace(-side/2, side/2, dipoles_per_dimension):
        for y in numpy.linspace(-side/2, side/2, dipoles_per_dimension):
            for z in numpy.linspace(-side/2, side/2, dipoles_per_dimension):
                dipoles.append([x, y, z])
    dipoles = numpy.asarray(dipoles)
    dipole_spacing = numpy.average(numpy.diff(numpy.linspace(-side/2, side/2, dipoles_per_dimension)))
    return dipoles, dipoles.shape[0], dipole_spacing

def dipole_cylinder(dipoles_per_min_dimension, radius, height):
    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)
    pow1d3 = misc.power_function(1./3.)

    dipoles = []
    if radius*2 < height:
        r_dim = dipoles_per_min_dimension
        h_dim = int(dipoles_per_min_dimension * height/(radius*2))
        dipole_spacing = numpy.average(numpy.diff(numpy.linspace(-radius, radius, r_dim)))
    else:
        r_dim = int(dipoles_per_min_dimension * height/(radius*2))
        h_dim = dipoles_per_min_dimension
        dipole_spacing = numpy.average(numpy.diff(numpy.linspace(0, height, h_dim)))

    for x in numpy.linspace(-radius, radius, r_dim):
        for y in numpy.linspace(-radius, radius, r_dim):
            for z in numpy.linspace(0, height, h_dim):
                if numpy.sqrt(pow2(x) + pow2(y)) <= radius:
                    dipoles.append([x, y, z])

    dipoles = numpy.asarray(dipoles)
    return dipoles, dipoles.shape[0], dipole_spacing
