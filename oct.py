# prevent Fortran routines in NumPy from catching interrupt signal
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import sys
import logging
from pathlib import Path
from time import time_ns

import numpy as np
from matplotlib import cm

from rainbow_logging_handler import RainbowLoggingHandler

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QCheckBox, QDoubleSpinBox
from PyQt5.QtGui import QMouseEvent, QTransform

from vortex.driver.alazar import Board, Channel
from vortex.acquire import AlazarConfig, AlazarAcquisition, InternalClock, ExternalClock, Input, AuxIOTriggerOut
from vortex.process import CUDAProcessor, CUDAProcessorConfig
from vortex.io import DAQmxIO, DAQmxConfig, AnalogVoltageOutput, AnalogVoltageInput, DigitalOutput
from vortex.engine import Block, source, acquire_alazar_clock, find_rising_edges, compute_resampling, dispersion_phasor, SampleStrobe, SegmentStrobe, VolumeStrobe, Polarity

from vortex import Range, get_console_logger as gcl
from vortex.engine import EngineConfig, Engine, StackDeviceTensorEndpointInt8 as StackDeviceTensorEndpoint, AscanStreamEndpoint, CounterStreamEndpoint, GalvoTargetStreamEndpoint, GalvoActualStreamEndpoint, MarkerLogStorage, NullEndpoint
from vortex.format import FormatPlanner, FormatPlannerConfig, StackFormatExecutorConfig, StackFormatExecutor, SimpleSlice
from vortex.marker import Flags, VolumeBoundary, ScanBoundary
from vortex.scan import RasterScanConfig, RadialScanConfig, FreeformScanConfig, FreeformScan, SpiralScanConfig, SpiralScan, limits
from vortex.storage import SimpleStreamInt8, SimpleStreamUInt64, SimpleStreamFloat64, SimpleStreamConfig, SimpleStreamHeader, MarkerLog, MarkerLogConfig

from vortex_tools.ui.backend import NumpyImageWidget
from vortex_tools.ui.display import RasterEnFaceWidget, CrossSectionImageWidget


# from argus.widget import VolumeImageWidget
# from argus.viz.math import scale, translate

RASTER_FLAGS = Flags(0x1)
AIMING_FLAGS = Flags(0x2)

_log = logging.getLogger(__name__)

class Config:
    scan_dimension=3
    bidirectional=False
    ascans_per_bscan=500
    bscans_per_volume=10
    galvo_delay=190e-6

    aiming_count = 10

    clock_samples_per_second=int(1000e6)
    blocks_to_acquire=0
    ascans_per_block=512
    samples_per_ascan=int(1376*2.25)
    trigger_delay_seconds=0

    blocks_to_allocate=128
    preload_count=32

    swept_source=source.Axsun100k
    internal_clock=True
    clock_channel=Channel.B
    input_channel=Channel.A

    process_slots=2
    dispersion=(-3e-6, 0)

    log_level=1

class System(QWidget):
    _stop_signal = pyqtSignal()

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(**kwargs)
        self._cfg = cfg

        self._use_aiming = False
        self._use_raster = True

        self._setup_scan()
        self._setup_core()
        self._setup_display()
        self._setup_save()
        self._setup_engine()
        self._setup_interface()

    def _setup_scan(self):
        rsc = RasterScanConfig()
        rsc.bscan_extent = Range(-self._cfg.scan_dimension, self._cfg.scan_dimension)
        rsc.volume_extent = Range(-self._cfg.scan_dimension, self._cfg.scan_dimension)
        rsc.bscans_per_volume = self._cfg.bscans_per_volume
        rsc.ascans_per_bscan = self._cfg.ascans_per_bscan
        # rsc.offset = [1.85, -1.05]
        rsc.bidirectional_segments = self._cfg.bidirectional
        rsc.bidirectional_volumes = self._cfg.bidirectional
        rsc.limits = [limits.ScannerMax_Saturn_5B]*2
        rsc.samples_per_second = self._cfg.swept_source.triggers_per_second
        rsc.flags = RASTER_FLAGS
        self._raster_config = rsc

        asc = RadialScanConfig()
        asc.ascans_per_bscan = self._cfg.ascans_per_bscan
        asc.bscan_extent = Range(-self._cfg.scan_dimension, self._cfg.scan_dimension)
        asc.offset = rsc.offset
        asc.limits = rsc.limits
        asc.samples_per_second = self._cfg.swept_source.triggers_per_second
        asc.set_aiming()
        asc.flags = AIMING_FLAGS
        self._aiming_config = asc

        self._update_scan()

    def _update_scan(self):
        raster_segments = self._raster_config.to_segments()
        aiming_segments = self._aiming_config.to_segments()

        pattern = []
        if self._use_aiming and self._use_raster:
            idx = np.linspace(0, len(raster_segments), self._cfg.aiming_count, dtype=int)
            for (i, (a, b)) in enumerate(zip(idx[:-1], idx[1:])):
                if i > 0:
                    markers = raster_segments[a].markers
                    markers.insert(0, VolumeBoundary(0, 0, False))
                    markers.insert(0, ScanBoundary(0, 0))

                pattern += raster_segments[a:b]
                pattern += aiming_segments
        elif self._use_aiming:
            pattern = aiming_segments
        else:
            pattern = raster_segments

        ffsc = FreeformScanConfig()
        ffsc.pattern = pattern
        ffsc.loop = True

        scan = FreeformScan()
        scan.initialize(ffsc)
        self._scan = scan

    def _setup_core(self):
        #
        # acquisition
        #

        ac = AlazarConfig()
        if self._cfg.internal_clock:
            ac.clock = InternalClock(self._cfg.clock_samples_per_second)
        else:
            ac.clock = ExternalClock()
        
        ac.trigger.range_millivolts = 0
        ac.inputs.append(Input(self._cfg.input_channel))
        ac.options.append(AuxIOTriggerOut())

        ac.records_per_block = self._cfg.ascans_per_block
        ac.trigger.delay_samples = int(self._cfg.trigger_delay_seconds * self._cfg.clock_samples_per_second)

        #
        # clocking
        #

        board = Board(ac.device.system_index, ac.device.board_index)
        if self._cfg.internal_clock:
            clock_path = Path(__file__).parent / 'clock.npz'
            if clock_path.exists():
                data = np.load(clock_path)
                clock_samples_per_second = data['clock_samples_per_second']
                clock = data['clock']
            else:
                (clock_samples_per_second, clock) = acquire_alazar_clock(self._cfg.swept_source, ac, self._cfg.clock_channel, gcl('acquire', self._cfg.log_level))
                np.savez(clock_path, clock_samples_per_second=clock_samples_per_second, clock=clock)

            self._cfg.swept_source.clock_edges_seconds = find_rising_edges(clock, clock_samples_per_second, len(self._cfg.swept_source.clock_edges_seconds))
            resampling = compute_resampling(self._cfg.swept_source, ac.samples_per_second, self._cfg.samples_per_ascan)

            # acquire enough samples to obtain the required ones
            ac.samples_per_record = board.info.smallest_aligned_samples_per_record(resampling.max())
        else:
            resampling = []
            ac.samples_per_record = board.info.smallest_aligned_samples_per_record(self._cfg.swept_source.clock_rising_edges_per_trigger)

        acquire = AlazarAcquisition(gcl('acquire', self._cfg.log_level))
        acquire.initialize(ac)
        self._acquire = acquire

        #
        # OCT processing setup
        #

        pc = CUDAProcessorConfig()

        # match acquisition settings
        pc.samples_per_record = ac.samples_per_record
        pc.ascans_per_block = ac.records_per_block

        pc.slots = self._cfg.process_slots

        # reasmpling
        pc.resampling_samples = resampling

        # spectral filter with dispersion correction
        window = np.hanning(pc.samples_per_ascan)
        phasor = dispersion_phasor(len(window), self._cfg.dispersion)
        pc.spectral_filter = window * phasor

        # DC subtraction per block
        pc.average_window = 2 * pc.ascans_per_block

        process = CUDAProcessor(gcl('process', self._cfg.log_level))
        process.initialize(pc)
        self._process = process

        #
        # galvo control
        #

        # output
        ioc_out = DAQmxConfig()
        ioc_out.persistent_task = False
        ioc_out.samples_per_block = ac.records_per_block
        ioc_out.samples_per_second = self._cfg.swept_source.triggers_per_second
        ioc_out.blocks_to_buffer = self._cfg.preload_count
        ioc_in = ioc_out.copy()
        sc = ioc_out.copy()

        ioc_out.name = 'output'
        ioc_out.clock.divisor = 2
        stream = Block.StreamIndex.GalvoTarget
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao0', 15 / 10, stream, 0))
        ioc_out.channels.append(AnalogVoltageOutput('Dev1/ao1', 15 / 10, stream, 1))

        io_out = DAQmxIO(gcl(ioc_out.name, self._cfg.log_level))
        io_out.initialize(ioc_out)
        self._io_out = io_out

        ioc_in.name = 'input'
        ioc_in.clock.divisor = 1
        stream = Block.StreamIndex.GalvoActual
        ioc_in.channels.append(AnalogVoltageInput('Dev1/ai0', 24.02 / 10, stream, 0))
        ioc_in.channels.append(AnalogVoltageInput('Dev1/ai1', 24.78 / 10, stream, 1))
        # ioc_in.channels.append(AnalogVoltageInput('Dev1/ai2', 1, stream, 2))
        # ioc_in.channels.append(AnalogVoltageInput('Dev1/ai3', 1, stream, 3))
        # ioc_in.channels.append(AnalogVoltageInput('Dev1/ai16', 600e-6, stream, 4))
        # ioc_in.channels.append(AnalogVoltageInput('Dev1/ai4', 1, stream, 5))
        # ioc_in.channels.append(AnalogVoltageInput('Dev1/ai5', 1, stream, 6))

        io_in = DAQmxIO(gcl(ioc_in.name, self._cfg.log_level))
        io_in.initialize(ioc_in)
        self._io_in = io_in

        sc.name = 'strobe'
        sc.channels.append(DigitalOutput('Dev1/port0', Block.StreamIndex.Strobes))
        strobe = DAQmxIO(gcl(sc.name, self._cfg.log_level))
        strobe.initialize(sc)
        self._strobe = strobe

    def _setup_engine(self):
        ec = EngineConfig()
        ec.add_acquisition(self._acquire, [self._process])
        ec.add_processor(self._process, [self._generic_format, self._aiming_format, self._raster_format])
        ec.add_formatter(self._generic_format, self._storage_endpoints)
        ec.add_formatter(self._aiming_format, [self._aiming_tensor_endpoint])
        ec.add_formatter(self._raster_format, [self._raster_tensor_endpoint])
        ec.add_io(self._io_out, lead_samples=round(self._cfg.galvo_delay * self._io_out.config.samples_per_second))
        ec.add_io(self._io_in, preload=False)
        ec.add_io(self._strobe)

        ec.preload_count = self._cfg.preload_count
        ec.records_per_block = self._cfg.ascans_per_block
        ec.blocks_to_allocate = self._cfg.blocks_to_allocate
        ec.blocks_to_acquire = self._cfg.blocks_to_acquire

        ec.galvo_output_channels = len(self._io_out.config.channels)
        ec.galvo_input_channels = len(self._io_in.config.channels)

        ec.lead_marker = ScanBoundary()

        ec.strobes = [SampleStrobe(0, 2), SampleStrobe(1, 333), SampleStrobe(2, 333, Polarity.Low), SegmentStrobe(3), VolumeStrobe(4)]

        engine = Engine(gcl('engine', self._cfg.log_level))
        self._engine = engine

        # automatically stop when engine exits
        def _engine_handler(event, exc):
            if event == Engine.Event.Exit:
                self._stop_signal.emit()
            if exc:
                _log.error(exc)
        self._engine.event_callback = _engine_handler
        self._stop_signal.connect(self.wait_engine)

        engine.initialize(ec)
        engine.prepare()

    def _setup_display(self):
        #
        # output setup
        #

        # format planners
        fc = FormatPlannerConfig()
        fc.strip_inactive = False

        generic_format = FormatPlanner(gcl('generic-format', self._cfg.log_level))
        generic_format.initialize(fc)
        self._generic_format = generic_format

        fc = FormatPlannerConfig()
        fc.segments_per_volume = self._raster_config.bscans_per_volume
        fc.records_per_segment = self._raster_config.ascans_per_bscan
        fc.adapt_shape = False

        fc.mask = self._raster_config.flags
        raster_format = FormatPlanner(gcl('format-raster', self._cfg.log_level))
        raster_format.initialize(fc)
        self._raster_format = raster_format

        fc.mask = self._aiming_config.flags
        fc.segments_per_volume = self._aiming_config.bscans_per_volume
        fc.records_per_segment = self._aiming_config.ascans_per_bscan
        aiming_format = FormatPlanner(gcl('format-aiming', self._cfg.log_level))
        aiming_format.initialize(fc)
        self._aiming_format = aiming_format

        # format executors
        cfec = StackFormatExecutorConfig()
        cfec.sample_slice = SimpleSlice(50 , self._process.config.samples_per_ascan // 2 )
        # cfec.sample_slice = SimpleSlice(self._process.config.samples_per_ascan // 2 + 10, self._process.config.samples_per_ascan - 50)
        samples_to_save = cfec.sample_slice.count()

        sfe = StackFormatExecutor()
        sfe.initialize(cfec)

        self._raster_tensor_endpoint = StackDeviceTensorEndpoint(sfe, (self._raster_config.bscans_per_volume, self._raster_config.ascans_per_bscan, samples_to_save), gcl('endpoint-raster', self._cfg.log_level))
        self._aiming_tensor_endpoint = StackDeviceTensorEndpoint(sfe, (self._aiming_config.bscans_per_volume, self._aiming_config.ascans_per_bscan, samples_to_save), gcl('endpoint-aiming', self._cfg.log_level))

    def _setup_save(self):
        self._ascan_stream = SimpleStreamInt8(gcl('storage', self._cfg.log_level))
        self._counter_stream = SimpleStreamUInt64(gcl('storage', self._cfg.log_level))

        self._signal_in_stream = SimpleStreamFloat64(gcl('storage', self._cfg.log_level))
        self._signal_out_stream = SimpleStreamFloat64(gcl('storage', self._cfg.log_level))

        self._marker_log = MarkerLog(gcl('storage', self._cfg.log_level))

        self._storage_objects = [
            self._ascan_stream,
            self._counter_stream,
            self._signal_in_stream,
            self._signal_out_stream,
            self._marker_log
        ]

        # get a callback before each storage endpoint processes a block
        self._monitor = NullEndpoint()
        self._monitor_queue = []
        def _monitor():
            while self._monitor_queue:
                work = self._monitor_queue.pop()
                work()
        self._monitor.update_callback = _monitor

        self._storage_endpoints = [
            self._monitor,
            AscanStreamEndpoint(self._ascan_stream),
            CounterStreamEndpoint(self._counter_stream),
            GalvoActualStreamEndpoint(self._signal_in_stream),
            GalvoTargetStreamEndpoint(self._signal_out_stream),
            MarkerLogStorage(self._marker_log)
        ]

    def _setup_interface(self):
        self.setWindowTitle('RAOCT DMC OCT')
        self.resize(1280, 720)

        # register widgets
        # self._raster_widget = RasterEnFaceWidget(self._raster_tensor_endpoint, cmap=cm.gray, debug=False)
        self._raster_widget = RasterEnFaceWidget(self._raster_tensor_endpoint, debug=False)
        self._raster_widget.setStyleSheet('background-color: black;')
        self._raster_widget.resize(500, 500)
        # self._flythrough_widget = CrossSectionImageWidget(self._raster_tensor_endpoint, sizing=[NumpyImageWidget.Sizing.Stretch]*2, cmap=cm.gray, range=[0, 30], debug=False)
        self._flythrough_widget = CrossSectionImageWidget(self._raster_tensor_endpoint, sizing=[NumpyImageWidget.Sizing.Stretch]*2, debug=False)
        self._flythrough_widget.setStyleSheet('background-color: black;')
        self._flythrough_widget.resize(1200, 300)

        # self._volume_widget = VolumeImageWidget(self._raster_tensor_endpoint)
        # self._volume_widget.resize(500, 500)
        # self._volume_widget.model_matrix = translate([-0.35, 0, 0]) @ scale([1.3]*3)
        # # self._volume_widget._renderer.set_transfer_function([[0, 0, 0, 0], [1, 1, 1, 1]], vmin=0, vmax=25)

        def _raster_handler(idxs):
            self._raster_widget.notify_segments(idxs)
            self._flythrough_widget.notify_segments(idxs)
            # self._volume_widget.notify_segments(idxs)
        self._raster_tensor_endpoint.aggregate_segment_callback = _raster_handler

        def _enface_click(e: QMouseEvent):
            w = self._raster_widget

            # get voxel coordinates
            xform = w._make_draw_transform()
            xform.translate(-w.image.width() / 2.0, -w.image.height() / 2.0)
            xform = xform.inverted()[0]

            # get scan position
            xform = xform * xform.fromScale(*[ex.length / s for (ex, s) in zip(self._raster_config.extents, self._raster_config.shape)][::-1])
            xform = xform * QTransform.fromTranslate(*[-ex.length / 2 for ex in self._raster_config.extents][::-1])

            # shift
            pt = xform.map(e.localPos())
            self._raster_config.offset = self._raster_config.offset + np.asanyarray([pt.y(), pt.x()])
            self._aiming_config.offset = self._raster_config.offset
            self._update_scan()
            self._engine.scan_queue.interrupt(self._scan)

            _log.info(f'changing scan offset to {self._raster_config.offset}')

        self._raster_widget.mouseDoubleClickEvent = _enface_click

        self._aiming_widgets = [CrossSectionImageWidget(self._aiming_tensor_endpoint, fixed=i, sizing=[NumpyImageWidget.Sizing.Stretch]*2, debug=False) for i in range(self._aiming_config.bscans_per_volume)]
        def _aiming_handler(idxs):
            for w in self._aiming_widgets:
                w.notify_segments(idxs)
        self._aiming_tensor_endpoint.aggregate_segment_callback = _aiming_handler

        layout = QVBoxLayout(self)

        # display panel
        panel = QHBoxLayout(self)
        layout.addLayout(panel)

        for w in self._aiming_widgets:
            panel.addWidget(w, 1)
        panel.addWidget(self._raster_widget, 2)
        panel.addWidget(self._flythrough_widget, 2)
        # panel.addWidget(self._volume_widget, 1)

        # control panel
        panel = QHBoxLayout()
        layout.addLayout(panel)

        for (label, target) in [('Start', self.start_engine), ('Stop', self.stop_engine)]:
            btn = QPushButton(label)
            btn.clicked.connect(target)
            panel.addWidget(btn)

        label = QLabel('Dispersion (e-6):')
        panel.addWidget(label)

        self._dispersion_spin = QDoubleSpinBox()
        self._dispersion_spin.setMinimum(-100)
        self._dispersion_spin.setMaximum(100)
        self._dispersion_spin.setValue(self._cfg.dispersion[0] * 1e6)
        self._dispersion_spin.setSingleStep(1)
        self._dispersion_spin.setDecimals(1)

        def _dispersion_handler():
            pc = self._process.config

            window = np.hanning(pc.samples_per_ascan)
            phasor = dispersion_phasor(len(window), (self._dispersion_spin.value() * 1e-6, 0))
            pc.spectral_filter = window * phasor

            self._process.change(pc)
        self._dispersion_spin.valueChanged.connect(_dispersion_handler)
        panel.addWidget(self._dispersion_spin)

        for (label, target) in [('Open', self.start_save), ('Close', self.stop_save)]:
            btn = QPushButton(label)
            btn.clicked.connect(target)
            panel.addWidget(btn)

        self._path_textbox = QLineEdit()
        self._path_textbox.setText('D:/Mark/RAOCTg2')
        panel.addWidget(self._path_textbox)

        for (label, shift_) in [('<', (0, 1)), ('v', (-1, 0)), ('o', None), ('^', (1, 0)), ('>', (0, -1))]:
            btn = QPushButton(label)
            btn.setMaximumWidth(25)

            def _click_handler(shift):
                if shift is None:
                    self._raster_config.offset = (0, 0)
                else:
                    self._raster_config.offset = self._raster_config.offset + 0.2 * np.asanyarray(shift)
                self._aiming_config.offset = self._raster_config.offset
                self._update_scan()
                self._engine.scan_queue.interrupt(self._scan)

            btn.clicked.connect(lambda _, shift=shift_: _click_handler(shift))
            panel.addWidget(btn)

        acbx = QCheckBox('Aiming')
        acbx.setChecked(self._use_aiming)
        rcbx = QCheckBox('Raster')
        rcbx.setChecked(self._use_raster)
        bcbx = QCheckBox('Zero Volume')
        bcbx.setChecked(False)
        a2cbx = QCheckBox('Zero B-scan')
        a2cbx.setChecked(False)

        def _check_handler():
            self._use_aiming = acbx.isChecked()
            self._use_raster = rcbx.isChecked()

            self._raster_config.volume_extent = Range.symmetric(0 if bcbx.isChecked() else self._cfg.scan_dimension)
            self._raster_config.bscan_extent = Range.symmetric(0 if a2cbx.isChecked() else self._cfg.scan_dimension)

            self._update_scan()
            self._engine.scan_queue.interrupt(self._scan)

        acbx.stateChanged.connect(_check_handler)
        rcbx.stateChanged.connect(_check_handler)
        bcbx.stateChanged.connect(_check_handler)
        a2cbx.stateChanged.connect(_check_handler)

        panel.addWidget(acbx)
        panel.addWidget(rcbx)
        panel.addWidget(bcbx)
        panel.addWidget(a2cbx)

    def start_save(self):
        if self.saving:
            _log.warning('saving is already started')
            return

        def _save_handler(base: Path):
            ssc = SimpleStreamConfig()
            ssc.header = SimpleStreamHeader.Empty

            ssc.path = base.with_suffix('.ascans.bin').as_posix()
            self._ascan_stream.open(ssc)

            ssc.path = base.with_suffix('.counter.bin').as_posix()
            self._counter_stream.open(ssc)

            ssc.path = base.with_suffix('.signals_in.bin').as_posix()
            self._signal_in_stream.open(ssc)

            ssc.path = base.with_suffix('.signals_out.bin').as_posix()
            self._signal_out_stream.open(ssc)

            mdc = MarkerLogConfig()
            mdc.path = base.with_suffix('.markers.bin').as_posix()
            mdc.binary = True
            self._marker_log.open(mdc)

        self._path_textbox.setEnabled(False)
        base = Path(self._path_textbox.text())

        # check for conflicts
        if list(Path.glob(base.parent, f'{base.name}*')):
            base = base.with_name(f'{base.name}-{time_ns():d}')
            _log.warning(f'changing save path to {base} due to conflicts')

        base.parent.mkdir(exist_ok=True)
        _log.info(f'starting save at "{base.as_posix()}"')

        if not self.running:
            # open storage now
            _log.info('opening storage objects')
            _save_handler(base)
        else:
            # queue for engine
            _log.info('requesting deferred opening of storage objects')
            self._monitor_queue.insert(0, lambda: _save_handler(base))

    def stop_save(self):
        if not self.saving:
            _log.warning('saving is already stopped')
            return

        self._path_textbox.setEnabled(True)

        def _close_handler():
            for storage in self._storage_objects:
                storage.close()

        if not self.running:
            # close immediately
            _log.info('closing storage objects')
            _close_handler()
        else:
            # queue for engine
            _log.info('requesting deferred closing of storage objects')
            self._monitor_queue.insert(0, lambda: _close_handler())

    def start_engine(self):
        # check if engine is startable
        if self.running:
            _log.warning('engine is already started')
            return
        self._engine.wait()

        # clear data state
        with self._raster_tensor_endpoint.tensor as volume:
            volume[:] = 0
        self._raster_widget.notify_segments(range(self._raster_config.bscans_per_volume))
        with self._aiming_tensor_endpoint.tensor as volume:
            volume[:] = 0
        # # for w in self._aiming_widgets:
        # #     w.notify_segments(range(self._aiming_config.bscans_per_volume))

        # clear formatting state
        self._raster_format.reset()
        self._aiming_format.reset()

        # restart the scan
        self._engine.scan_queue.reset()
        self._engine.scan_queue.interrupt(self._scan)
        self._monitor_queue.clear()

        # start the engine
        _log.info('starting engine')
        self._engine.start()

    def stop_engine(self):
        # check if engine is stoppable
        if not self.running:
            _log.warning('engine is already stopped')
            return

        # request that the engine stop
        # NOTE: that the engine will complete pending blocks in the background
        _log.info('requesting engine stop')
        self._engine.stop()

    def wait_engine(self):
        try:
            self._engine.wait()
        except RuntimeError as e:
            _log.error(f'engine error: {e}')
            QMessageBox.critical(self, 'Engine Error', f'The session aborted prematurely due to an error.\n\n\t{e}\n\nPlease check the log for additional information.')
        finally:
            self.stop_engine()

        # close out the save session
        if self.saving:
            self.stop_save()

    @property
    def running(self):
        return not self._engine.done
    @property
    def saving(self):
        return self._ascan_stream.ready

def setup_logging():
    # configure the root logger to accept all records
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(name)s] %(filename)s:%(lineno)d\t%(levelname)s:\t%(message)s')

    # set up colored logging to console
    console_handler = RainbowLoggingHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

if __name__ == '__main__':
    import win32api, win32con, win32process
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, win32api.GetCurrentProcessId())
    win32process.SetProcessAffinityMask(handle, 0x000000f0)

    setup_logging()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # catch unhandled exceptions
    import traceback
    def handler(cls, ex, trace):
        traceback.print_exception(cls, ex, trace)
        app.closeAllWindows()
    sys.excepthook = handler

    # cause KeyboardInterrupt to exit the Qt application
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: app.exit())

    # regularly re-enter Python so the signal handler runs
    def keepalive(msec):
        QTimer.singleShot(msec, lambda: keepalive(msec))
    keepalive(10)

    window = System(Config())
    window.show()
    # window._raster_widget.show()
    # window._flythrough_widget.show()
    # window._volume_widget.show()

    app.exec_()

    if window.running:
        window.stop_engine()
        window.wait_engine()
