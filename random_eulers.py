import numpy as np
import os, argparse

"""
Draw orientations from a random distribution on a sphere and convert to Euler 
angles, saving files corresponding both to the ZXZ' and ZYX conventions. 
"""

def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Generate a list of random euler angles.')
    parser.add_argument('-s','--savepath', help='Path for saving eulers_zxz.txt and eulers_zyx.txt', 
                        required=True)
    parser.add_argument('-n','--n_eulers', help='Number of Euler angles to generate', 
                        required=True, type=int)
    return vars(parser.parse_args())


def random_rmatrix():
    """
    Generate a random rotation matrix drawn from a uniform distribution of orientations on a sphere. 
    Algorithm courtesy: http://biorobots.cwru.edu/personnel/adh/stackoverflow/02/randrotmat.m.
    """
    rn = 2*np.random.rand(3) 
    th = np.pi*rn[0]
    ph = np.pi*rn[1]
    z = rn[2]

    r = np.sqrt(z)
    vx = r*np.sin(ph)
    vy = r*np.cos(ph)
    vz = np.sqrt(2-z)

    st = np.sin(th)
    ct = np.cos(th)
    sx = vx*ct-vy*st
    sy = vx*st+vy*ct

    R = np.array([[vx*sx-ct, vx*sy-st, vx*vz],
                  [vy*sx+st, vy*sy-ct, vy*vz],
                  [vz*sx, vz*sy, 1-z]])

    return R


def matrix2euler_zxz(rmatrix, as_deg=True):
    """
    Convert an R matrix to Euler angles for rotation around Z,X,Z' axes. That's the EMAN2 
    convention according to: https://groups.google.com/forum/#!topic/eman2/v79NudDgIDY.
    Formula for conversion is based on: https://www.geometrictools.com/Documentation/EulerAngles.pdf.
    Default is to convert to degrees. Seems to match EMAN2 rotation matrix stored in Transform object.

    To check, e.g.:
    from EMAN2 import *
    t = Transform()
    t.set_params({"type":"eman", "az":-97.7, "alt": 87.1, "phi":-16.8})
    t.get_matrix()
    """
    if np.abs(rmatrix[2,2]!=1):
        theta_x = np.arccos(rmatrix[2,2])
        theta_z0 = np.arctan2(rmatrix[0,2], -1*rmatrix[1,2])
        theta_z1 = np.arctan2(rmatrix[2,0], rmatrix[2,1])   
        
    elif rmatrix[2,2]==-1:
        theta_x = np.pi
        theta_z0 = -1*np.arctan2(-1*rmatrix[0,1], rmatrix[0,0])
        theta_z1 = 0

    else:
        theta_x = 0
        theta_z0 = np.arctan2(-1*rmatrix[0,1], rmatrix[0,0])
        theta_z1 = 0

    return np.rad2deg(theta_z0), np.rad2deg(theta_x), np.rad2deg(theta_z1)


def matrix2euler_zyx(rmatrix, as_deg=True):
    """
    Convert an R matrix to Euler angles for rotation around Z,Y,X axes. That's the order 
    expected by the IMOD rotatevol function. Formula for conversion is based on: 
    https://www.geometrictools.com/Documentation/EulerAngles.pdf. Default output in degrees. 
    """
    if rmatrix[2,0]!=1:
        if rmatrix[2,0]!=-1:
            theta_y = np.arcsin(-1*rmatrix[2,0])
            theta_z = np.arctan2(rmatrix[1,0], rmatrix[0,0])
            theta_x = np.arctan2(rmatrix[2,1], rmatrix[2,2])
        else:
            theta_y = np.pi/2.0
            theta_z = -1*np.arctan2(rmatrix[1,2], rmatrix[1,1])
            theta_x = 0
    else:
        theta_y = -1*np.pi/2.0
        theta_z = np.arctan2(-1*rmatrix[1,2], rmatrix[1,1])
        theta_x = 0

    return np.rad2deg(theta_z), np.rad2deg(theta_y), np.rad2deg(theta_x)


if __name__ == '__main__':

    args = parse_commandline()

    eulers_zxz = np.zeros((args['n_eulers'], 3))
    eulers_zyx = np.zeros((args['n_eulers'], 3))

    for i in range(args['n_eulers']):
        R = random_rmatrix()
        eulers_zxz[i] = matrix2euler_zxz(R)
        eulers_zyx[i] = matrix2euler_zyx(R)

    np.savetxt(os.path.join(args['savepath'], "eulers_zxz.txt"), eulers_zxz, fmt='%.2f', delimiter=' ')
    np.savetxt(os.path.join(args['savepath'], "eulers_zyx.txt"), eulers_zyx, fmt='%.2f', delimiter=' ')
