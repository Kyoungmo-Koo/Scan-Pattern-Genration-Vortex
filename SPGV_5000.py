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
cfg.segments_per_volume = 10
cfg.samples_per_segment = 300
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)
names.append('Raster Scan - Unidirectional')

cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)
cfgs[-1].bidirectional_segments = True
names.append('Raster Scan - Bidirectional')

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

print(len(path_list[0]))
print(len(path_list[1]))

with open("Unidirectional_Raster_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path_list[0][i][0]} {path_list[0][i][1]}\n")

with open("Bidirectional_Raster_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path_list[1][i][0]} {path_list[1][i][1]}\n")


###############################################################################################################################################################################################

from math import pi, cos, sin

import numpy as np
from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RadialScan, RadialScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list= []

cfg = RadialScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.segments_per_volume = 10
cfg.samples_per_segment = 300
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

# change offset
names.append('Radial Scan - Standard')
cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

# change extent
names.append('Radial Scan - Small Extent')
cfgs.append(cfg.copy())
cfgs[-1].segment_extent = Range(0.5, 1)
cfgs[-1].volume_extent = Range(0, 5.6)
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

for (name, cfg, ax) in zip(names, cfgs, axs.flat):
    scan = RadialScan()
    scan.initialize(cfg)
    for segment in cfg.to_waypoints():
        ax.plot(segment[:, 0], segment[:, 1], 'x')

    path = scan.scan_buffer()
    path_list.append(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', lw=1, zorder=-1)

    ax.set_xlabel('x (au)')
    ax.set_ylabel('y (au)')
    ax.set_title(name)

x = np.abs(axs.flat[0].get_xlim()).max()
axs.flat[0].set_xlim(-x, x)

print(len(path_list[0]))
print(len(path_list[1]))

with open("Standard_Radial_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path_list[0][i][0]} {path_list[0][i][1]}\n")

with open("Small_Extent_Radial_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path_list[1][i][0]} {path_list[1][i][1]}\n")

############################################################################################################################################################

from matplotlib import pyplot as plt

from vortex import Range
from vortex.scan import RasterScan, RasterScanConfig
from vortex_tools.scan import plot_annotated_waveforms_space

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, subplot_kw=dict(adjustable='box', aspect='equal'))
cfgs = []
names = []
path_list = []

cfg = RasterScanConfig()
cfg.segment_extent = Range.symmetric(1)
cfg.volume_extent = Range.symmetric(2)
cfg.segments_per_volume = 10
cfg.samples_per_segment = 300
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

names.append('Segments')
cfgs.append(cfg.copy())
cfgs[-1].bidirectional_segments = True
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

names.append('Volumes')
cfgs.append(cfg.copy())
cfgs[-1].bidirectional_volumes = True
cfgs[-1].samples_per_segment = 150
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(100, 100)

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

print(len(path_list[0]))
print(len(path_list[1]))

with open("Volume_Bidirectional_Raster_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path_list[1][i][0]} {path_list[1][i][1]}\n")

###################################################################################################################################################

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
cfg.samples_per_segment = 100
for limit in cfg.limits:
    limit.acceleration *= 5
cfg.loop = True

names.append('Standard')
cfgs.append(cfg.copy())
cfgs[-1].repeat_count = 1
cfgs[-1].repeat_period = 1
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

names.append('Repeat')
cfgs.append(cfg.copy())
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

names.append('Even Repeat Period + Bidirectional')
cfgs.append(cfg.copy())
cfgs[-1].repeat_period = 2
cfgs[-1].bidirectional_segments = True
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

names.append('Odd Repeat Period + Bidirectional')
cfgs.append(cfg.copy())
cfgs[-1].repeat_period = 3
cfgs[-1].bidirectional_segments = True
cfgs[-1].inactive_policy = inactive_policy.FixedDynamicLimited(200, 200)

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

print(len(path_list[0]))
print(len(path_list[1]))

with open("Repeat_3_Raster_Scan_5400.txt", "w") as file:
    for i in range(5400):
        file.write(f"{path_list[1][i][0]} {path_list[1][i][1]}\n")

###################################################################################################################################################

from math import pi

from vortex.scan import FreeformScanConfig, FreeformScan, SequentialPattern
from vortex_tools.scan import plot_annotated_waveforms_space

(r, theta) = np.meshgrid(
    [1, 2, 3, 4],
    np.linspace(0, 2*pi, 750),
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
cfg.inactive_policy = inactive_policy.FixedDynamicLimited(500, 500)

scan = FreeformScan()
scan.initialize(cfg)
_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Freeform Scan')

print(len(scan.scan_buffer()))
path = scan.scan_buffer()

with open("Freeform_Scan_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path[i][0]} {path[i][1]}\n")

################################################################################################################################################

from math import pi

from vortex.scan import FreeformScanConfig, FreeformScan, SequentialPattern
from vortex_tools.scan import plot_annotated_waveforms_space

(r, theta) = np.meshgrid(
    [2, 3],
    np.linspace(0, 2*pi, 1500),
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
cfg.inactive_policy = inactive_policy.FixedDynamicLimited(1000, 1000)

scan = FreeformScan()
scan.initialize(cfg)
_, ax = plot_annotated_waveforms_space(scan.scan_buffer(), scan.scan_markers(), inactive_marker=None, scan_line='w-')
ax.set_title('Freeform Scan')

print(len(scan.scan_buffer()))
path = scan.scan_buffer()

with open("Freeform_Scan_two_circles_5000.txt", "w") as file:
    for i in range(5000):
        file.write(f"{path[i][0]} {path[i][1]}\n")
