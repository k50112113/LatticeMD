
#This is a subroutine that produce a Movie using plotly and ffmpeg
#Snapshot = MD snapshots; Frame = movie frame
#Usage
#   mpiexec -n <number of procs> python MatterMovieMaker.py /path/to/MDSequenceData <number of snapshots to average> <snapshot interval per frame> [options]
#[options]
#   -c:   nth channel of matter image (nth matter in the dataset) Default: 0
#   -s:   number of iso-surfaces to render                        Default: 10
#   -op:  opacity of the iso-surfaces                             Default: 0.1
#   -max: max iso-value                                           Default: max of first frame
#   -min: min iso-value                                           Default: min of first frame, then min = min + (max - min)*0.3
#   -fr:  movie framerate                                         Default: 10
#   -o:   output filename                                         Default: output-<number of snapshots to average>-<snapshot interval per frame>.mp4

#example: mpiexec -n 4 python MatterMovieMaker.py MDSD-raw.save 100 100 -s 5 -max 4.0 -min 1.0 -fr 30 -o ttt

try:
    from mpi4py import MPI
    USE_MPI = True
except:
    USE_MPI = False
import dill
import sys
import os
import torch
import numpy as np

if USE_MPI:
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
else:
    rank = 0
    nprocs = 1

data_path = sys.argv[1]
num_snapshot_avg = int(sys.argv[2])
snapshot_interval = int(sys.argv[3])

nth_matter    = int(sys.argv[sys.argv.index('-c')+1])     if '-c'   in sys.argv else 0
surface_count = int(sys.argv[sys.argv.index('-s')+1])     if '-s'   in sys.argv else 10
opacity       = float(sys.argv[sys.argv.index('-op')+1])  if '-op'  in sys.argv else 0.1
isomin        = float(sys.argv[sys.argv.index('-min')+1]) if '-min' in sys.argv else None
isomax        = float(sys.argv[sys.argv.index('-max')+1]) if '-max' in sys.argv else None 
framerate     = float(sys.argv[sys.argv.index('-fr')+1])  if '-fr'  in sys.argv else 10 
output_filename = sys.argv[sys.argv.index('-o')+1] if '-o' in sys.argv else "output-%d-%d.mp4"%(num_snapshot_avg, snapshot_interval)
if output_filename[-4:] != '.mp4': output_filename += '.mp4'

if rank == 0:
    fin = open(data_path,"rb")
    number_of_matters_tmp, matter_dim_tmp, system_dim_tmp, sequence_length_tmp, \
    matter_sequence_data_tmp, _, _, \
    _, _, _ = dill.load(fin).get_data()
    fin.close()

    matter_sequence_data_tmp = matter_sequence_data_tmp.view(len(matter_sequence_data_tmp),1,system_dim_tmp[0],system_dim_tmp[1],system_dim_tmp[2],number_of_matters_tmp,matter_dim_tmp,matter_dim_tmp,matter_dim_tmp)
    start_snapshot_array = list(range(0, len(matter_sequence_data_tmp) - num_snapshot_avg, snapshot_interval))

    n_frame_per_rank = len(start_snapshot_array)//nprocs
    print("Total number of MD snapshots = %d"%(len(matter_sequence_data_tmp)), end = "")
    print(", will make %d images"%(len(start_snapshot_array)), end = "")
    print(" with %d images per processor"%(n_frame_per_rank))

    for irank in range(nprocs):
        start_frame_index = n_frame_per_rank*irank
        if irank < nprocs - 1: end_frame_index = n_frame_per_rank*(irank+1) - 1
        else: end_frame_index = len(start_snapshot_array)-1
        first_start_snapshot = start_snapshot_array[start_frame_index]
        last_start_snapshot = start_snapshot_array[end_frame_index]
        ff = start_snapshot_array[start_frame_index:end_frame_index+1]
        if irank > 0:
            comm.send([matter_sequence_data_tmp[first_start_snapshot:last_start_snapshot+num_snapshot_avg], ff, snapshot_interval, num_snapshot_avg, isomax, isomin], dest=irank, tag=11)
        else:
            split_data = matter_sequence_data_tmp[:last_start_snapshot+num_snapshot_avg]
            split_start_snapshot_array = ff[:]

            vol = split_data[0:num_snapshot_avg,0,:,:,:,nth_matter,:,:,:]
            vol = vol.permute(0,1,4,2,5,3,6)
            vol = vol.flatten(start_dim = 5, end_dim=6).flatten(start_dim = 3, end_dim=4).flatten(start_dim = 1, end_dim=2)
            vol = vol.mean(dim = 0)

            if isomax is None:
                isomax = vol.max().item()
            if isomin is None:
                isomin = vol.min().item()
                isomin = isomin + (isomax - isomin)*0.3

        print("Distribute No. %d ~ %d-th snapshot (%d ~ %d-th frame) to %d-th processor"%(first_start_snapshot, last_start_snapshot+num_snapshot_avg, start_frame_index, end_frame_index, irank))

import plotly.graph_objects as go
if rank > 0:
    split_data, split_start_snapshot_array, snapshot_interval, num_snapshot_avg, isomax, isomin = comm.recv(source=0, tag=11)

print("%d-th processor got %d snapshots %d (%d ~ %d)"%(rank, len(split_data), len(split_start_snapshot_array), split_start_snapshot_array[0], split_start_snapshot_array[-1]))

for start_snapshot in split_start_snapshot_array:
    this_filename = "tmp-%d-%d-%05d.png"%(num_snapshot_avg, snapshot_interval, start_snapshot)
    vol = split_data[start_snapshot-split_start_snapshot_array[0]:start_snapshot-split_start_snapshot_array[0]+num_snapshot_avg,0,:,:,:,nth_matter,:,:,:]
    vol = vol.permute(0,1,4,2,5,3,6)
    vol = vol.flatten(start_dim = 5, end_dim=6).flatten(start_dim = 3, end_dim=4).flatten(start_dim = 1, end_dim=2)
    vol = vol.mean(dim = 0)

    vol_dim = vol.shape
    X, Y, Z = np.mgrid[:vol_dim[0], :vol_dim[1], :vol_dim[2]]
    vol = vol.numpy()

    fig = go.Figure(data=go.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=vol.flatten(), isomin=isomin, isomax=isomax, opacity=opacity, surface_count=surface_count, colorscale = 'jet'))
    fig.update_layout(scene_xaxis_showticklabels=False, scene_yaxis_showticklabels=False, scene_zaxis_showticklabels=False)
    fig.write_image(this_filename)

if rank > 0:
    comm.send(1, dest=0, tag=12)
    print("Processor %d done"%(rank))

if rank == 0:
    for irank in range(1, nprocs):
        d = comm.recv(source=irank, tag=12)
    print("All done")
    os.system('rm %s'%(output_filename))
    os.system('''ffmpeg -framerate %d -pattern_type glob -i 'tmp-%d-%d-*.png' -vf scale=-1:1080 -c:v mpeg4 -pix_fmt yuv420p %s'''%(framerate ,num_snapshot_avg, snapshot_interval, output_filename))
    os.system('rm tmp-%d-%d-*.png'%(num_snapshot_avg, snapshot_interval))
    
