__author__ = 'Kryosugarra'
import numpy

class Scatterer:


    @staticmethod
    def load_dda_si_dipole_file(filename):
        f = open(filename, 'r')
        lines = f.readlines()
        dipoles = []
        for line in lines:
            dipole = []
            for d in line.split(','):
                dipole.append(float(d))
            dipoles.append(dipole)

        scat = Scatterer()
        scat.dipole = numpy.asarray(dipoles)
        return scat