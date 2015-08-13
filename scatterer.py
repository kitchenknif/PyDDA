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
        h_dim = numpy.rint(dipoles_per_min_dimension * height/(radius*2))
        dipole_spacing = 2*radius/(r_dim-1)
    else:
        r_dim = numpy.rint(dipoles_per_min_dimension * (radius*2)/height)
        h_dim = dipoles_per_min_dimension
        dipole_spacing = height/(h_dim-1)

    for x in numpy.linspace(-radius, radius, r_dim):
        for y in numpy.linspace(-radius, radius, r_dim):
            for z in numpy.linspace(-height/2, height/2, h_dim):
                if numpy.sqrt(pow2(x) + pow2(y)) <= radius:
                    dipoles.append([x, y, z])

    dipoles = numpy.asarray(dipoles)
    return dipoles, dipoles.shape[0], dipole_spacing


def dipole_cylinder_r(dipoles_per_min_dimension, radius, height):
    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)
    pow1d3 = misc.power_function(1./3.)

    dipoles = []
    r_dim = dipoles_per_min_dimension
    h_dim = numpy.rint(dipoles_per_min_dimension * height/(radius*2))
    dipole_spacing = 2*radius/(r_dim-1)

    for x in numpy.linspace(-radius, radius, r_dim):
        for y in numpy.linspace(-radius, radius, r_dim):
            for z in numpy.linspace(-height/2, height/2, h_dim):
                if numpy.sqrt(pow2(x) + pow2(y)) <= radius:
                    dipoles.append([x, y, z])

    dipoles = numpy.asarray(dipoles)
    return dipoles, dipoles.shape[0], dipole_spacing



def dipole_spheroid(dipoles_per_min_dimension, a, b):
    pow2 = misc.power_function(2)
    pow3 = misc.power_function(3)
    pow1d3 = misc.power_function(1./3.)

    dipoles = []

    if a < b:
        a_dim = dipoles_per_min_dimension
        b_dim = numpy.rint(dipoles_per_min_dimension * b/a)
        #assert not a_dim == b_dim
        dipole_spacing = 2*a/(a_dim-1)
    else:
        b_dim = dipoles_per_min_dimension
        a_dim = numpy.rint(dipoles_per_min_dimension * a/b)
        #assert not a_dim == b_dim
        dipole_spacing = 2*b/(b_dim-1)

    for x in numpy.linspace(-a, a, a_dim):
        for y in numpy.linspace(-a, a, a_dim):
            for z in numpy.linspace(-b, b, b_dim):
                if numpy.sqrt(pow2(x/a) + pow2(y/a) + pow2(z/b)) <= 1:
                    dipoles.append([x, y, z])

    #initial_spacing = dipole_spacing
    #dipole_spacing = pow1d3(4 / 3 * numpy.pi / len(dipoles) * a * a * b)
    #dipoles = numpy.asarray(dipoles)*(dipole_spacing/initial_spacing)

    dipoles = numpy.asarray(dipoles)
    return dipoles, dipoles.shape[0], dipole_spacing