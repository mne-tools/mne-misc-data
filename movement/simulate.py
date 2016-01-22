# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.transforms import rot_to_quat
from mne.simulation import simulate_raw

data_path = mne.datasets.sample.data_path()

###############################################################################
# Simulate some head movement (typically recorded data could be used instead)

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
subjects_dir = data_path + '/subjects'
bem_fname = subjects_dir + '/sample/bem/sample-5120-bem-sol.fif'
src_fname = subjects_dir + '/sample/bem/sample-oct-6-src.fif'

# let's make rotation matrices about each axis, plus some compound rotations
phi = np.deg2rad(30)
x_rot = np.array([[1, 0, 0],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi), np.cos(phi)]])
y_rot = np.array([[np.cos(phi), 0, np.sin(phi)],
                  [0, 1, 0],
                  [-np.sin(phi), 0, np.cos(phi)]])
z_rot = np.array([[np.cos(phi), -np.sin(phi), 0],
                  [np.sin(phi), np.cos(phi), 0],
                  [0, 0, 1]])
xz_rot = np.dot(x_rot, z_rot)
xmz_rot = np.dot(x_rot, z_rot.T)
yz_rot = np.dot(y_rot, z_rot)
mymz_rot = np.dot(y_rot.T, z_rot.T)


# Create different head rotations, one per second
rots = [x_rot, y_rot, z_rot, xz_rot, xmz_rot, yz_rot, mymz_rot]
# The transpose of a rotation matrix is a rotation in the opposite direction
rots += [rot.T for rot in rots]

raw = mne.io.Raw(raw_fname).crop(0, len(rots))
raw.load_data().pick_types(meg=True, stim=True, copy=False)
raw.add_proj([], remove_existing=True)
center = (0., 0., 0.04)  # a bit lower than device origin
raw.info['dev_head_t']['trans'] = np.eye(4)
raw.info['dev_head_t']['trans'][:3, 3] = center
pos = np.zeros((len(rots), 10))
for ii in range(len(pos)):
    pos[ii] = np.concatenate([[ii], rot_to_quat(rots[ii]), center, [0] * 3])
pos[:, 0] += raw.first_samp / raw.info['sfreq']  # initial offset

# Let's activate a vertices bilateral auditory cortices
src = mne.read_source_spaces(src_fname)
labels = mne.read_labels_from_annot('sample', 'aparc.a2009s', 'both',
                                    regexp='G_temp_sup-Plan_tempo',
                                    subjects_dir=subjects_dir)
assert len(labels) == 2  # one left, one right
vertices = [np.intersect1d(l.vertices, s['vertno'])
            for l, s in zip(labels, src)]
data = np.zeros([sum(len(v) for v in vertices), int(raw.info['sfreq'])])
activation = np.hanning(int(raw.info['sfreq'] * 0.2)) * 1e-9  # nAm
t_offset = int(np.ceil(0.2 * raw.info['sfreq']))  # 200 ms in (after baseline)
data[:, t_offset:t_offset + len(activation)] = activation
stc = mne.SourceEstimate(data, vertices, tmin=-0.2,
                         tstep=1. / raw.info['sfreq'])

# Simulate the movement
raw = simulate_raw(raw, stc, trans_fname, src, bem_fname,
                   head_pos=pos, interp='zero', n_jobs=-1)
raw_stat = simulate_raw(raw, stc, trans_fname, src, bem_fname,
                        head_pos=None, n_jobs=-1)
# Save the results
raw.save('simulated_movement_raw.fif', buffer_size_sec=1.)
raw_stat.save('simulated_stationary_raw.fif', buffer_size_sec=1.)
mne.chpi.write_head_quats('simulated_quats.pos', pos)
stc.save('simulated_activation')
