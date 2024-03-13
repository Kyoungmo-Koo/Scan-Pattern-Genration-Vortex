#RasterScan

from vortex import Range
from vortex.scan import RasterWaypoints
import matplotlib.pyplot as plt

wpts = RasterWaypoints()
wpts.volume_extent = Range.symmetric(1)
wpts.bscan_extent = Range.symmetric(1)
wpts.samples_per_segment = 20
wpts.segments_per_volume = 10

xy = wpts.to_waypoints()

fig, ax = plt.subplots()

for segment in xy:
    ax.plot(segment[:, 0], segment[:, 1], 'x')

ax.set_title('Raster Scan Waypoints')
ax.set_xlabel('x (au)')
ax.set_ylabel('y (au)')
ax.axis('equal')

#RasterScan unidirectional/bidirectional#############################################################################

from vortex import Range
from vortex.scan import RasterScan, RasterScanConfig

cfg = RasterScanConfig()
cfg.volume_extent = Range.symmetric(1)
cfg.bscan_extent = Range.symmetric(1)
cfg.samples_per_segment = 20
cfg.segments_per_volume = 10
for limit in cfg.limits:
    limit.velocity *= 10
    limit.acceleration *= 40
cfg.loop = True

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))

cfgs = []
names = []
path_list = []

cfgs.append(cfg.copy())
names.append('Raster Scan - Unidirectional')

cfgs.append(cfg.copy())
cfgs[-1].bidirectional_segments = True
names.append('Raster Scan - Bidirectional')

for (name, cfg, ax) in zip(names, cfgs, axs):
    scan = RasterScan()
    scan.initialize(cfg)

    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'w-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)

#unidirectional needs more scan points because it moves longer distance
len(path_list[0])
#bidirectional needs less scan points because it moves shorter distance
len(path_list[1])

#Rasterscan Variety####################################################################################################

from math import pi, cos, sin

import numpy as np
from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RasterScan, RasterScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list = []

cfg = RasterScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.segments_per_volume = 10
cfg.samples_per_segment = 50
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

# change offset
names.append('Offset')
cfgs.append(cfg.copy())
cfgs[-1].offset = (1, 0)

# change extent
names.append('Extent')
cfgs.append(cfg.copy())
cfgs[-1].volume_extent = Range(2, -2)
cfgs[-1].segment_extent = Range(0, 1)

# change shape
names.append('Shape')
cfgs.append(cfg.copy())
cfgs[-1].segments_per_volume = 5

# change rotation
names.append('Angle')
cfgs.append(cfg.copy())
cfgs[-1].angle = pi / 6

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RasterScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'w-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)

#Radialscan Variety####################################################################################################

from math import pi, cos, sin

import numpy as np
from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RadialScan, RadialScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list= []

cfg = RadialScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.segments_per_volume = 10
cfg.samples_per_segment = 50
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

# change offset
names.append('Offset')
cfgs.append(cfg.copy())
cfgs[-1].offset = (1, 0)

# change extent
names.append('Extent')
cfgs.append(cfg.copy())
cfgs[-1].volume_extent = Range(0, 5)
cfgs[-1].segment_extent = Range(0.5, 1)

# change shape
names.append('Shape')
cfgs.append(cfg.copy())
cfgs[-1].segments_per_volume = 5

# change rotation
names.append('Angle')
cfgs.append(cfg.copy())
cfgs[-1].angle = pi / 6

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RadialScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'w-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)


x = np.abs(axs.flat[0].get_xlim()).max()
axs.flat[0].set_xlim(-x, x)

#Rasterscan Variety 2 ###############################################################################################

from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RasterScan, RasterScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list = []

cfg = RasterScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.volume_extent = Range.symmetric(2)
cfg.segments_per_volume = 10
cfg.samples_per_segment = 50
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

names.append('Default')
cfgs.append(cfg.copy())

names.append('Segments')
cfgs.append(cfg.copy())
cfgs[-1].bidirectional_segments = True

names.append('Volumes')
cfgs.append(cfg.copy())
cfgs[-1].bidirectional_volumes = True

names.append('Segments + Volumes')
cfgs.append(cfg.copy())
cfgs[-1].bidirectional_segments = True
cfgs[-1].bidirectional_volumes = True

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RasterScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)

#test code
print(len(path_list[0]))
print(len(path_list[1]))
print(len(path_list[2]))
print(len(path_list[3]))
with open("output.txt", "w") as file:
    for i in range(2940):
        file.write(f"{path_list[2][i]}\n")
      
#RepeatedRasterscan Variety#######################################################################################

from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RepeatedRasterScan, RepeatedRasterScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list = []

cfg = RepeatedRasterScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.volume_extent = Range.symmetric(2)
cfg.segments_per_volume = 6
cfg.samples_per_segment = 50
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

names.append('Standard')
cfgs.append(cfg.copy())
cfgs[-1].repeat_count = 1
cfgs[-1].repeat_period = 1

names.append('Repeat')
cfgs.append(cfg.copy())

names.append('Even Repeat Period + Bidirectional')
cfgs.append(cfg.copy())
cfgs[-1].repeat_period = 2
cfgs[-1].bidirectional_segments = True

names.append('Odd Repeat Period + Bidirectional')
cfgs.append(cfg.copy())
cfgs[-1].repeat_period = 3
cfgs[-1].bidirectional_segments = True

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RepeatedRasterScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)

#test code
print(len(path_list[0]))
print(len(path_list[1]))
print(len(path_list[2]))
print(len(path_list[3]))
with open("output.txt", "w") as file:
    for i in range(1576):
        file.write(f"{path_list[2][i]}\n")

#Rasterscan Inactive Policy

from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RasterScanConfig, RasterScan, inactive_policy
from vortex_tools.scan import plot_annotated_waveforms_space, partition_segments_by_activity

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list = []

cfg = RasterScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.volume_extent = Range.symmetric(2)
cfg.segments_per_volume = 6
cfg.samples_per_segment = 50
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

names.append('Minimum Dynamic Limited')
cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.MinimumDynamicLimited()

names.append('Fixed Dynamic Limited')
cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

names.append('Fixed Linear')
cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedLinear()

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RasterScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)
    
for ax in axs.flat[len(names):]:
    fig.delaxes(ax)

#test code

print(len(path_list[0]))
print(len(path_list[1]))
print(len(path_list[2]))
with open("output.txt", "w") as file:
    for i in range(1500):
        file.write(f"{path_list[1][i]}\n")

#RasterScan Marker
from vortex import Range
from vortex.scan import RepeatedRasterScan
from vortex_tools.scan import plot_annotated_waveforms_space

scan = RepeatedRasterScan()
cfg = scan.config
cfg.volume_extent = Range.symmetric(2)
cfg.samples_per_segment = 100
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True
scan.initialize(cfg)

_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Repeated Raster')

#test code
path_list = []
path_list.append(scan.scan_buffer())
path_list.append(scan.scan_markers())
print(len(path_list[0]))
print(len(path_list[1]))
with open("output2.txt", "w") as file:
    for i in range(6660):
        file.write(f"{path_list[0][i]}\n")

#RadialScan Marker

from vortex.scan import RadialScan
from vortex_tools.scan import plot_annotated_waveforms_space

scan = RadialScan()
cfg = scan.config
cfg.samples_per_segment = 100
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True
scan.initialize(cfg)

_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Radial')

#RepeatedRadialScan Marker

from vortex.scan import RepeatedRadialScan
from vortex_tools.scan import plot_annotated_waveforms_space

scan = RepeatedRadialScan()
cfg = scan.config
cfg.samples_per_segment = 100
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True
scan.initialize(cfg)

_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Repeated Radial')

#FreeformScan Marker

from math import pi

from vortex.scan import FreeformScanConfig, FreeformScan, SequentialPattern
from vortex_tools.scan import plot_annotated_waveforms_space

(r, theta) = np.meshgrid(
    [2, 3, 4],
    np.linspace(0, 2*pi, 200),
    indexing='ij'
)

x = (r + 0.1*np.sin(r + 20*theta)) * np.sin(theta)
y = (r + 0.1*np.sin(r + 20*theta)) * np.cos(theta)

waypoints = np.stack((x, y), axis=-1)
pattern = SequentialPattern().to_pattern(waypoints)

cfg = FreeformScanConfig()
cfg.pattern = pattern
for limit in cfg.limits:
    limit.velocity *= 5
    limit.acceleration *= 10
cfg.bypass_limits_check = True
cfg.loop = True

scan = FreeformScan()
scan.initialize(cfg)

_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Freeform Scan')
