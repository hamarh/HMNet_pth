# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of {{ project }} nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath_data', type=str, help='')
    parser.add_argument('fpath_gt', type=str, help='')
    parser.add_argument('dpath_out', type=str, help='')
    args = parser.parse_args()


import os
import h5py
import numpy as np
from PIL import Image
import traceback

def mkdir(dpath):
    if not os.path.isdir(dpath):
        try:
            os.mkdir(dpath)
        except FileExistsError:
            print('Already exists: %s' % dpath)
            pass
        except:
            print(traceback.format_exc())


mkdir(args.dpath_out)
dpath_out = args.dpath_out + '/data/'
mkdir(dpath_out)

target_label = "depth"
events_label = "events-betweenframes"
frame_label = "frames"

dpath_depth  = dpath_out + '/' + target_label
dpath_events = dpath_out + '/' + events_label
dpath_frame  = dpath_out + '/' + frame_label
mkdir(dpath_depth)
mkdir(dpath_events)
mkdir(dpath_frame)


data = h5py.File(args.fpath_data)
gt = h5py.File(args.fpath_gt)

depth = gt['davis']['left']['depth_image_raw']
for i in range(len(depth)):
    print(f'{i} / {len(depth)}')
    dpt = depth[i]
    np.save(dpath_depth + '/%05d.npy' % i, dpt)

frame = data['davis']['left']['image_raw']
for i in range(len(frame)):
    frm = frame[i]
    print(f'{i} / {len(frame)}, {frm.shape}')
    frm = Image.fromarray(frm)
    frm.save(dpath_frame + '/%05d.png' % i)



depth_timestamps = gt['davis']['left']['depth_image_raw_ts'][...]
indices = np.arange(len(depth_timestamps))
out = np.stack([indices, depth_timestamps], axis=1)
np.savetxt(dpath_depth + '/timestamps.txt', out)

valid = np.ones_like(depth_timestamps).astype(bool)
np.save(dpath_out + '/valid_depth_indices.npy', valid)


frame_timestamps = data['davis']['left']['image_raw_ts'][...]
indices = np.arange(len(frame_timestamps))
out = np.stack([indices, frame_timestamps], axis=1)
np.savetxt(dpath_frame + '/timestamps.txt', out)

frame_to_event_index = data['davis']['left']['image_raw_event_inds']
np.save(dpath_out + '/frame_to_event_index.npy', frame_to_event_index)

ts = data['davis']['left']['events'][:,2]
np.save(dpath_events + '/t.npy', ts[:,None])


dts = depth_timestamps.copy()
depth_to_event_index = []
for idx, t in enumerate(ts):
    if len(dts) == 0:
        break
    if t > dts[0]:
        i = max(idx-1, 0)
        depth_to_event_index.append(i)
        dts = dts[1:]
for _ in range(len(dts)):
    depth_to_event_index.append(idx-1)

dts = depth_timestamps.copy()
depth_to_frame_index = []
for idx, t in enumerate(frame_timestamps):
    if len(dts) == 0:
        break
    if t > dts[0]:
        i = max(idx-1, 0)
        depth_to_frame_index.append(i)
        dts = dts[1:]
for _ in range(len(dts)):
    depth_to_frame_index.append(idx-1)

np.save(dpath_out + '/depth_to_event_index.npy', np.array(depth_to_event_index))
np.save(dpath_out + '/depth_to_frame_index.npy', np.array(depth_to_frame_index))

p = ((data['davis']['left']['events'][:,3] + 1) * 0.5).astype(np.uint8)
np.save(dpath_events + '/p.npy', p[:,None])

xy = data['davis']['left']['events'][:,:2]
np.save(dpath_events + '/xy.npy', xy)



