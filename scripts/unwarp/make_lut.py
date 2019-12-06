import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Make lookup table (LUT) for spherical-to-cylindrical conversion')
parser.add_argument('input_width',type=int,help='width of input (spherical) panorama image')
parser.add_argument('input_height',type=int,help='height of input (spherical) panorama image')
parser.add_argument('output_width',type=int,help='width of output (cylindrical) panorama image')
parser.add_argument('output_height',type=int,help='height of output (cylindrical) panorama image')
parser.add_argument('bottom',type=float,help='bottom height value of cylinder')
parser.add_argument('top',type=float,help='top height value of cylinder')
# output
parser.add_argument('--lut_target', default='data/lut.npy',
                                    help='output lookup table location')
parser.add_argument('--intrinsics_target', default='data/intrinsics.txt',
                                           help='intrinsics location')
args = parser.parse_args()

# allocate arrays for lookup table
# lutx[y,x] maps y,x in output image to x coordinate in input image
# similar for luty
lutx = np.zeros((args.output_height,args.output_width))
luty = np.zeros((args.output_height,args.output_width))

# make arrays of equally spaced angle and height values
thetas = np.linspace(-np.pi,np.pi,num=args.output_width,endpoint=False)
heights = np.linspace(args.bottom,args.top,num=args.output_height,endpoint=True)

for r in range(args.output_height):
    height = heights[r]
    for c in range(args.output_width):
        theta = thetas[c]

        # get XYZ point
        X = np.sin(theta)
        Y = height
        Z = np.cos(theta)

        # get vertical angle
        phi = np.arcsin(Y/np.sqrt(X*X+Y*Y+Z*Z))

        # project to spherical panorama image
        x = args.input_width*(0.5*theta/np.pi+0.5)
        y = args.input_height*(phi/np.pi+0.5)

        lutx[r,c] = x
        luty[r,c] = y

print(lutx)
print(luty)
lut = np.concatenate([np.expand_dims(lutx,axis=-1),np.expand_dims(luty,axis=-1)],axis=-1)
np.save(args.lut_target,lut)

# solve for theta intrinsics
x0 = 0
x1 = len(thetas)-1
theta0 = thetas[0]
theta1 = thetas[-1]
c_theta = (theta0*x1 - theta1*x0)/(theta0 - theta1)
f_theta = (x0 - x1)/(theta0 - theta1)

# solve for Z intrinsics
y0 = 0
y1 = len(heights)-1
Z0 = heights[0]
Z1 = heights[-1]
c_Z = (Z0*y1 - Z1*y0)/(Z0 - Z1)
f_Z = (y0 - y1)/(Z0 - Z1)

print('Intrinsics:')
print('  f_theta: {}'.format(f_theta))
print('  c_theta: {}'.format(c_theta))
print('  f_Z: {}'.format(f_Z))
print('  c_Z: {}'.format(c_Z))

# save intrinsics to file
with open(args.intrinsics_target,'w') as f:
    f.write('%.15f %.15f %.15f %.15f' % (f_theta,c_theta,f_Z,c_Z))

