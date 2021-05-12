
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4]), float(l[5]), float(l[6]), float(l[7]) ) for l in list if len(l)>1]
    return np.array(list)

def visualize_trajectory(tx, ty, tz):
    # Visualize path 
    idx = np.arange(len(tx))

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.plot3D(tx, ty, tz, 'gray')
    ax.scatter3D(tx, ty, tz, c=idx, s=1, cmap='hsv')

    plt.show()

def quaternion_to_Matrix(qx, qy, qz, qw):

    return np.array([[1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw], \
                     [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw], \
                     [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]])

def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/norm(p0), p1/norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def interp(p0, p1, t):
    return p0*(1-t) + p1*t

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2
        midval = dset[mid]
        if midval == x:
            return mid
        # elif (r - l) < 1:
        #     return r
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    
    if side == 'left':
        return l
    return r

# timestamp tx ty tz qx qy qz qw
gt = read_file_list('groundtruth_file.txt')
pred = read_file_list('predicted2.txt')

gt_ts, gt_tx, gt_ty, gt_tz, gt_qx, gt_qy, gt_qz, gt_qw = gt.transpose()
pd_ts, pd_tx, pd_ty, pd_tz, pd_qx, pd_qy, pd_qz, pd_qw = pred.transpose()

gt_end_idx = np.array([binary_search_h5_dset(gt_ts, t) for t in pd_ts])[:-1]
gt_bgn_idx = gt_end_idx-1

gt_bgn_ts = gt_ts[gt_bgn_idx]
gt_end_ts = gt_ts[gt_end_idx]
interp_ts = (pd_ts[:-1] - gt_bgn_ts) / (gt_end_ts - gt_bgn_ts)

q0 = np.dstack([gt_qx[gt_bgn_idx], gt_qy[gt_bgn_idx], gt_qz[gt_bgn_idx], gt_qw[gt_bgn_idx]]).squeeze()[:-1]
q1 = np.dstack([gt_qx[gt_end_idx], gt_qy[gt_end_idx], gt_qz[gt_end_idx], gt_qw[gt_end_idx]]).squeeze()[:-1]
t0 = np.dstack([gt_tx[gt_bgn_idx], gt_ty[gt_bgn_idx], gt_tz[gt_bgn_idx]]).squeeze()
t1 = np.dstack([gt_tx[gt_end_idx], gt_ty[gt_end_idx], gt_tz[gt_end_idx]]).squeeze()

tm = np.array([interp(p0, p1, t) for p0, p1, t, in zip(t0, t1, interp_ts)])
qm = np.array([slerp(p0, p1, t) for p0, p1, t, in zip(q0, q1, interp_ts)])
Rot = quaternion_to_Matrix(qm[:,0], qm[:,1], qm[:,2], qm[:,3])
pd_Rot = np.array([quaternion_to_Matrix(qx, qy, qz, qw) for qx, qy, qz, qw in zip(pd_qx, pd_qy, pd_qz, pd_qw)])[:-1]


tmp_s = None
for i in range(100):
    
    s = np.eye(4)
    s[0:3, 0:3] = Rot[:,:,i]
    s[0:3, 3] = tm[i]

    if i == 0:
        tmp_s = s
        continue

    p = np.eye(4)
    p[0:3, 0:3] = pd_Rot[i]
    p[0:3, 0] = pd_tx[i]
    p[0:3, 1] = pd_ty[i]
    p[0:3, 2] = pd_tz[i]

    relative_error = np.linalg.inv(tmp_s) @ s
    RE_trans = relative_error[0:3, 3]


    print(relative_error)
    print(norm(RE_trans))
    print()





raise


# Rot = quaternion_to_Matrix(qx, qy, qz, qw)
# Rot = Rot.transpose(2,0,1)

visualize_trajectory(tx, ty, tz)



