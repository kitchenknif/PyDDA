import numpy
__author__ = 'p.dmitriev'


def read_data(filename):
    try:
        f = open(filename)
        lines = f.readlines()
        n_rows = len(lines)
        n_cols = len(lines[0].split(','))

        dat = numpy.zeros([n_rows, n_cols], dtype=numpy.complex128)
        for i in range(n_rows):
            l = lines[i].split(',')
            for j in range(n_cols):
                dat[i, j] = complex(l[j])

        return dat

    except Exception as err:
        raise err


def write_data(filename, l, dat):
    assert len(l) == len(dat)
    try:
        f = open(filename, 'w')
        for i in range(len(l)):
            f.write('{}, {}\n'.format(l[i], dat[i]))

        f.close()
    except Exception as err:
        raise err
