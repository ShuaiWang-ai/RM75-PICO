import contextlib
import copy
import io
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Assets' / 'Scripts'))

from rm75_live_control import (LiveControlFault, LiveMotionController,
                               MotionOutcome, MotionResult)
from rm75_log_replay import replay_log
from rm75_target_preview import build_parser, run
from test_rm75_preview_core import PEER
from test_rm75_target_preview import FakeArm, STATE, payload


class LiveSocket:
    def __init__(self, events, clock, event_log):
        self.events = iter(events)
        self.clock = clock
        self.event_log = event_log
        self.closed = False
        self.bound = False

    def bind(self, address):
        self.bound = True
        self.event_log.append('udp-bind')

    def settimeout(self, timeout):
        pass

    def recvfrom(self, size):
        try:
            item = next(self.events)
        except StopIteration:
            raise KeyboardInterrupt
        now, data, *rest = item
        self.clock[0] = now
        if isinstance(data, BaseException):
            raise data
        if data is None:
            raise socket.timeout()
        return data, rest[0] if rest else PEER

    def close(self):
        self.closed = True
        self.event_log.append('udp-close')


class LiveArm(FakeArm):
    def __init__(self, event_log):
        super().__init__()
        self.info[1]['force_type'] = '6FB'
        self.event_log = event_log

    def rm_delete_robot_arm(self):
        self.event_log.append('robot-delete')
        return super().rm_delete_robot_arm()


class ControllerArm(LiveArm):
    def __init__(self, event_log):
        super().__init__(event_log)
        self.moves = []
        self.slow_interrupt = False

    def rm_movej_follow(self, joints):
        self.moves.append(list(joints))
        self.event_log.append('send')
        return 0

    def rm_set_arm_slow_stop(self):
        self.event_log.append('slow-stop')
        if self.slow_interrupt:
            raise KeyboardInterrupt
        return 0

    def rm_set_arm_stop(self):
        self.event_log.append('fast-stop')
        return 0


class ControllerKinematics:
    def __init__(self, clock, slow_on_solve=None):
        self.clock = clock
        self.solve_count = 0
        self.slow_on_solve = slow_on_solve

    def calculate_arm_angle(self, joints):
        return 179.5

    def solve(self, seed, pose, angle):
        self.solve_count += 1
        if self.solve_count == self.slow_on_solve:
            self.clock[0] += .251
        return list(seed)


class FakeMotion:
    def __init__(self, event_log):
        self.event_log = event_log
        self.prepared = False
        self.has_sent = False
        self.prepare_calls = []
        self.targets = []
        self.monitor_calls = []
        self.effective_stops = []
        self.fault_on = None
        self.monitor_state = None
        self.results = []

    def prepare(self, state):
        self.prepare_calls.append(state)
        if self.fault_on == 'prepare':
            raise LiveControlFault('prepare fault')
        self.prepared = True

    def send_target(self, target, now):
        self.has_sent = True
        self.targets.append((list(target), now))
        self.event_log.append('send')
        if self.fault_on == 'send':
            raise LiveControlFault('send fault')
        if self.fault_on == 'interrupt':
            raise KeyboardInterrupt
        if self.results:
            return self.results.pop(0)
        return MotionResult(MotionOutcome.SENT, 'Command accepted')

    def monitor(self, now):
        self.monitor_calls.append(now)
        if self.fault_on == 'monitor':
            raise LiveControlFault('monitor fault')
        return copy.deepcopy(self.monitor_state)

    def stop(self, reason):
        if self.has_sent:
            self.effective_stops.append(reason)
            self.event_log.append('stop')
        self.prepared = False
        self.has_sent = False


class FakeStateBroadcaster:
    def __init__(self, event_log):
        self.event_log = event_log
        self.published = []
        self.fail_publish = False
        self.closed = False

    def publish(self, state, pico_ip):
        self.event_log.append('state-publish')
        if self.fail_publish:
            raise OSError('visualization send failed')
        self.published.append((copy.deepcopy(state), pico_ip))

    def close(self):
        self.closed = True
        self.event_log.append('state-close')


def packet(seq, now, grip=0, session='a', valid=True, verified=False, **changes):
    if verified:
        changes.update(device_valid=True, position_valid=True,
                       rotation_valid=True, grip_valid=True,
                       tracked_valid=True, tracked=True)
    return payload(seq, now, grip, session=session, valid=valid, **changes)


class LiveLoopTests(unittest.TestCase):
    def execute(self, events, configure_motion=None, argv=None, arm=None,
                with_state=False, configure_state=None,
                confirmation='MOVE_RM75', orientation_factory=None):
        self.clock = [0.0]
        self.event_log = []
        self.arm = arm or LiveArm(self.event_log)
        self.motion = FakeMotion(self.event_log)
        if configure_motion:
            configure_motion(self.motion)
        self.sock = LiveSocket(events, self.clock, self.event_log)
        self.state = FakeStateBroadcaster(self.event_log)
        if configure_state:
            configure_state(self.state)
        args = build_parser().parse_args(['--live', *(argv or [])])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = run(args, lambda: self.arm, lambda *unused: self.sock,
                       lambda: self.clock[0], input_func=lambda prompt: confirmation,
                       stdin_isatty=lambda: True,
                       live_factory=lambda *unused: self.motion,
                       state_factory=(lambda port: self.state) if with_state else None,
                       orientation_factory=orientation_factory)
        return code, output.getvalue()

    def active_events(self, tail):
        return [(0.0, packet(0, 0.0)), (.01, packet(1, .01)),
                (.02, packet(2, .02, 1)), *tail]

    def test_grip_press_sends_monitors_and_release_stops(self):
        events = self.active_events([(.23, packet(3, .23, 1, z=.01)),
                                     (.24, packet(4, .24, 0))])
        code, output = self.execute(events)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.prepare_calls), 1)
        self.assertEqual(len(self.motion.targets), 2)
        self.assertEqual(self.motion.monitor_calls, [.02, .23])
        self.assertEqual(self.motion.effective_stops, ['Grip released; ready'])
        self.assertIn('LIVE', output)
        self.assertIn('STOPPED', output)
        self.assertLess(self.event_log.index('stop'), self.event_log.index('robot-delete'))
        self.assertLess(self.event_log.index('robot-delete'), self.event_log.index('udp-close'))

    def test_recoverable_motion_hold_does_not_stop_and_reverse_resumes(self):
        def configure(motion):
            motion.results = [
                MotionResult(MotionOutcome.SENT, 'Command accepted'),
                MotionResult(MotionOutcome.HOLD, 'IK joint step limit'),
                MotionResult(MotionOutcome.SENT, 'Command accepted'),
            ]
        events = self.active_events([
            (.23, packet(3, .23, 1, z=.02)),
            (.24, packet(4, .24, 1, z=-.001)),
            (.25, packet(5, .25, 0, z=-.001)),
        ])

        code, output = self.execute(events, configure)

        self.assertEqual(code, 0)
        self.assertIn('HOLD: IK joint step limit', output)
        self.assertIn('LIVE ACTIVE', output)
        self.assertEqual(self.motion.effective_stops, ['Grip released; ready'])

    def test_control_log_from_live_loop_replays_without_robot_or_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'control.jsonl')
            events = self.active_events([
                (.23, packet(3, .23, 1, z=.02)),
                (.24, packet(4, .24, 0, z=.02)),
            ])

            code, output = self.execute(events, argv=['--control-log', path])
            replayed = replay_log(path)

        self.assertEqual(code, 0, output)
        self.assertEqual([frame.status for frame in replayed.frames],
                         ['WAIT_RELEASE', 'READY', 'ACTIVE', 'ACTIVE', 'READY'])
        self.assertEqual(replayed.config.axes, ('z', '-x', 'y'))

    def test_control_log_replays_slow_reference_read_disarm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'slow-reference.jsonl')
            arm = LiveArm([])
            def delay_second_read(reads):
                if reads == 2:
                    self.clock[0] += .3
            arm.read_hook = delay_second_read
            events = self.active_events([])

            code, output = self.execute(
                events, argv=['--control-log', path], arm=arm)
            replayed = replay_log(path)

        self.assertEqual(code, 0, output)
        self.assertEqual(replayed.frames[-1].status, 'WAIT_RELEASE')
        self.assertIn('Reference read timeout', replayed.frames[-1].reason)

    def test_live_position_radius_holds_without_stopping(self):
        tail = [(sequence / 100, packet(sequence, sequence / 100, 1, z=z))
                for sequence, z in enumerate((.08, .16, .24, .32, .40, .48),
                                             start=3)]
        events = self.active_events(tail)

        code, output = self.execute(events)

        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 6)
        self.assertIn('outside reference radius', output)
        self.assertIn('HOLD', output)
        self.assertEqual(self.motion.effective_stops, ['Program exit'])

    def test_live_target_speed_is_limited_to_twenty_millimeters_per_second(self):
        events = self.active_events([(.22, packet(3, .22, 1, z=.1))])

        code, unused_output = self.execute(events)

        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 2)
        reference_x = STATE['pose'][0]
        self.assertAlmostEqual(self.motion.targets[1][0][0], reference_x + .004)

    def test_single_axis_orientation_mode_locks_xyz_and_maps_selected_axis(self):
        factory_calls = []
        mapper_calls = []
        def orientation_factory(axes, project_axis):
            factory_calls.append((axes, project_axis))
            def mapper(reference, current, robot_rpy):
                mapper_calls.append((reference, current, robot_rpy))
                return [.4, .5, .6]
            return mapper
        angle = .01
        events = [
            (0.0, packet(0, 0.0, verified=True)),
            (.01, packet(1, .01, verified=True)),
            (.02, packet(2, .02, 1, verified=True)),
            (.23, packet(3, .23, 1, verified=True, z=.04,
                         qz=math.sin(angle / 2), qw=math.cos(angle / 2))),
            (.24, packet(4, .24, 0, verified=True)),
        ]

        code, output = self.execute(
            events,
            argv=['--live-orientation-axis', 'x'],
            confirmation='MOVE_RM75_ORIENTATION_X',
            orientation_factory=orientation_factory,
        )

        self.assertEqual(code, 0, output)
        self.assertEqual(factory_calls, [(('z', '-x', 'y'), 'x')])
        self.assertEqual(len(mapper_calls), 1)
        self.assertEqual(len(self.motion.targets), 2)
        self.assertEqual(self.motion.targets[1][0][:3], STATE['pose'][:3])
        self.assertEqual(self.motion.targets[1][0][3:], [.4, .5, .6])

    def test_live_6dof_maps_relative_xyz_and_full_orientation(self):
        factory_calls = []
        mapper_calls = []

        def orientation_factory(axes):
            factory_calls.append(axes)
            def mapper(reference, current, robot_rpy):
                mapper_calls.append((reference, current, robot_rpy))
                return [.4, .5, .6]
            return mapper

        angle = math.radians(1)
        events = [
            (0.0, packet(0, 0.0, verified=True)),
            (.01, packet(1, .01, verified=True)),
            (.02, packet(2, .02, 1, verified=True)),
            (.23, packet(3, .23, 1, verified=True, z=.04,
                         qz=math.sin(angle / 2), qw=math.cos(angle / 2))),
            (.24, packet(4, .24, 0, verified=True)),
        ]

        code, output = self.execute(
            events, argv=['--live-6dof'], confirmation='MOVE_RM75_6DOF',
            orientation_factory=orientation_factory)

        self.assertEqual(code, 0, output)
        self.assertEqual(factory_calls, [('z', '-x', 'y')])
        self.assertEqual(len(mapper_calls), 1)
        self.assertEqual(self.motion.targets[1][0][3:], [.4, .5, .6])
        self.assertGreater(self.motion.targets[1][0][0], STATE['pose'][0])
        self.assertIn('LIVE 6DOF ARMED', output)

    def test_live_6dof_disarms_unverified_tracking(self):
        events = self.active_events([(.23, packet(3, .23, 1, qz=.005, qw=.999))])

        code, output = self.execute(
            events, argv=['--live-6dof'], confirmation='MOVE_RM75_6DOF',
            orientation_factory=lambda axes: lambda *items: [.4, .5, .6])

        self.assertEqual(code, 0)
        self.assertEqual(self.motion.targets, [])
        self.assertIn('Verified tracking', output)

    def test_single_axis_orientation_mode_rejects_unverified_bridge_packets(self):
        events = self.active_events([(.23, packet(3, .23, 1, qz=.005, qw=.999))])

        code, output = self.execute(
            events,
            argv=['--live-orientation-axis', 'x'],
            confirmation='MOVE_RM75_ORIENTATION_X',
            orientation_factory=lambda axes, project_axis: lambda *items: [.4, .5, .6],
        )

        self.assertEqual(code, 0)
        self.assertEqual(self.motion.targets, [])
        self.assertIn('Verified tracking', output)

    def test_single_axis_orientation_mode_enforces_two_degree_radius(self):
        angle = math.radians(2.001)
        events = [
            (0.0, packet(0, 0.0, verified=True)),
            (.01, packet(1, .01, verified=True)),
            (.02, packet(2, .02, 1, verified=True)),
            (.23, packet(3, .23, 1, verified=True,
                         qz=math.sin(angle / 2), qw=math.cos(angle / 2))),
        ]

        code, output = self.execute(
            events,
            argv=['--live-orientation-axis', 'x'],
            confirmation='MOVE_RM75_ORIENTATION_X',
            orientation_factory=lambda axes, project_axis: lambda *items: [.4, .5, .6],
        )

        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 1)
        self.assertIn('outside orientation radius', output)

    def test_single_axis_orientation_mode_limits_rate_to_point_zero_five_radians(self):
        mapped_rotations = []
        def orientation_factory(unused_axes, unused_project_axis):
            def mapper(unused_reference, current, robot_rpy):
                mapped_rotations.append(current)
                return list(robot_rpy)
            return mapper
        requested = math.radians(1.0)
        events = [
            (0.0, packet(0, 0.0, verified=True)),
            (.01, packet(1, .01, verified=True)),
            (.02, packet(2, .02, 1, verified=True)),
            (.12, packet(3, .12, 1, verified=True,
                         qz=math.sin(requested / 2), qw=math.cos(requested / 2))),
            (.13, packet(4, .13, 0, verified=True)),
        ]

        code, output = self.execute(
            events,
            argv=['--live-orientation-axis', 'x'],
            confirmation='MOVE_RM75_ORIENTATION_X',
            orientation_factory=orientation_factory,
        )

        self.assertEqual(code, 0, output)
        limited_angle = 2 * math.acos(abs(mapped_rotations[0][3]))
        self.assertAlmostEqual(limited_angle, .005, places=6)

    def test_active_input_faults_stop_once(self):
        cases = {
            'malformed': [(.03, b'bad')],
            'tracking': [(.03, packet(3, .03, 1, valid=False))],
            'timeout': [(.28, None)],
            'jump': [(.03, packet(3, .03, 1, z=.101))],
            'source': [(.03, packet(3, .03, 1, session='b'), ('10.0.0.2', 5005))],
        }
        for name, tail in cases.items():
            with self.subTest(name=name):
                code, output = self.execute(self.active_events(tail))
                self.assertEqual(code, 0)
                self.assertEqual(len(self.motion.effective_stops), 1)
                self.assertIn('WAIT_RELEASE', output)

    def test_ready_malformed_does_not_stop_but_active_malformed_does(self):
        events = [(0, packet(0, 0)), (.01, packet(1, .01)),
                  (.02, b'bad'), (.03, packet(2, .03)),
                  (.04, packet(3, .04, 1)), (.05, b'bad')]
        code, unused = self.execute(events)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 1)
        self.assertEqual(len(self.motion.effective_stops), 1)

    def test_motion_fault_stops_before_next_udp_target(self):
        def fault(motion):
            motion.fault_on = 'monitor'
        events = self.active_events([(.03, packet(3, .03, 1, z=.01))])
        code, output = self.execute(events, fault)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 1)
        self.assertEqual(len(self.motion.effective_stops), 1)
        self.assertIn('FAULT', output)

    def test_monitor_runs_during_socket_poll_without_new_target(self):
        events = self.active_events([(.22, None), (.23, packet(3, .23, 0))])
        code, unused = self.execute(events)
        self.assertEqual(code, 0)
        self.assertEqual(self.motion.monitor_calls, [.02, .22])

    def test_state_stream_uses_startup_idle_and_active_measured_states(self):
        measured = {'joint': [float(index) for index in range(7)],
                    'pose': [0.0] * 6}
        def configure(motion):
            motion.monitor_state = measured
        events = self.active_events([(.22, None), (.23, packet(3, .23, 0))])
        code, unused = self.execute(events, configure, with_state=True)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(self.state.published), 2)
        self.assertEqual(self.state.published[0][1], None)
        self.assertEqual(self.state.published[-1], (measured, PEER[0]))
        self.assertTrue(self.state.closed)
        self.assertLess(self.event_log.index('robot-delete'),
                        self.event_log.index('state-close'))

    def test_state_publish_failure_does_not_change_motion_or_stop(self):
        events = self.active_events([(.03, packet(3, .03, 1, z=.001)),
                                     (.04, packet(4, .04, 0))])
        code, output = self.execute(
            events, with_state=True,
            configure_state=lambda state: setattr(state, 'fail_publish', True))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 2)
        self.assertEqual(self.motion.effective_stops, ['Grip released; ready'])
        self.assertIn('STATE OUTPUT', output)

    def test_invalid_state_port_rejected_before_robot_connection(self):
        for value in ('0', '65536'):
            created = []
            args = build_parser().parse_args(['--live', '--state-port', value])
            output = io.StringIO()
            with self.subTest(value=value), contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                code = run(args, lambda: created.append('arm'), socket.socket,
                           lambda: 0.0, input_func=lambda prompt: 'MOVE_RM75',
                           stdin_isatty=lambda: True,
                           live_factory=lambda *unused: None)
            self.assertEqual(code, 1)
            self.assertEqual(created, [])

    def test_poll_monitor_fault_stops_before_next_target(self):
        calls = [0]
        def fault_on_second_monitor(motion):
            original = motion.monitor
            def monitor(now):
                calls[0] += 1
                if calls[0] == 2:
                    motion.monitor_calls.append(now)
                    raise LiveControlFault('poll state fault')
                return original(now)
            motion.monitor = monitor
        events = self.active_events([(.22, None),
                                     (.23, packet(3, .23, 1, z=.01))])
        code, output = self.execute(events, fault_on_second_monitor)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 1)
        self.assertEqual(len(self.motion.effective_stops), 1)
        self.assertIn('poll state fault', output)

    def test_send_fault_stops(self):
        code, output = self.execute(self.active_events([]),
                                    lambda motion: setattr(motion, 'fault_on', 'send'))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.effective_stops), 1)
        self.assertIn('send fault', output)

    def test_ctrl_c_after_send_stops_before_delete(self):
        code, output = self.execute(self.active_events([]),
                                    lambda motion: setattr(motion, 'fault_on', 'interrupt'))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.effective_stops), 1)
        self.assertLess(self.event_log.index('stop'), self.event_log.index('robot-delete'))

    def test_unhandled_receive_exception_stops_before_delete(self):
        code, output = self.execute(self.active_events([(.03, RuntimeError('receive failed'))]))
        self.assertEqual(code, 1)
        self.assertEqual(len(self.motion.effective_stops), 1)
        self.assertLess(self.event_log.index('stop'), self.event_log.index('robot-delete'))

    def test_retired_source_cannot_restart(self):
        other = ('10.0.0.2', 5005)
        events = self.active_events([(.03, packet(0, .03, session='b'), other),
                                     (.04, packet(3, .04)),
                                     (.05, packet(4, .05, 1, z=.02))])
        code, unused = self.execute(events)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.motion.targets), 1)
        self.assertEqual(len(self.motion.effective_stops), 1)

    def test_live_preview_options_cannot_relax_fixed_limits(self):
        cases = [('--scale', '.3'), ('--axes', 'x,y,z'), ('--radius', '.04'),
                 ('--speed', '.02'), ('--timeout', '.5'), ('--jump-limit', '.2')]
        for flag, value in cases:
            created = []
            args = build_parser().parse_args(['--live', flag, value])
            output = io.StringIO()
            with self.subTest(flag=flag), contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                code = run(args, lambda: created.append('arm'), socket.socket,
                           lambda: 0.0, input_func=lambda prompt: 'MOVE_RM75',
                           stdin_isatty=lambda: True,
                           live_factory=lambda *unused: None)
            self.assertEqual(code, 1)
            self.assertEqual(created, [])

    def test_startup_validation_precedes_confirmation(self):
        for configure, message in (
                (lambda arm: arm.info[1].update(force_type='B'), '6FB'),
                (lambda arm: setattr(arm, 'run_mode', (0, 0)), 'run mode')):
            arm = LiveArm([])
            configure(arm)
            prompted = []
            clock = [0.0]
            sock = LiveSocket([], clock, [])
            args = build_parser().parse_args(['--live'])
            output = io.StringIO()
            with self.subTest(message=message), contextlib.redirect_stdout(output), \
                    contextlib.redirect_stderr(output):
                code = run(args, lambda: arm, lambda *unused: sock,
                           lambda: clock[0], input_func=lambda prompt: prompted.append(prompt),
                           stdin_isatty=lambda: True,
                           live_factory=lambda *unused: self.fail('created motion'))
            self.assertEqual(code, 1)
            self.assertEqual(prompted, [])
            self.assertIn(message, output.getvalue())

    def test_live_udp_binding_occurs_after_confirmation(self):
        self.clock = [0.0]
        self.event_log = []
        arm = LiveArm(self.event_log)
        motion = FakeMotion(self.event_log)
        sock = LiveSocket([], self.clock, self.event_log)
        def confirm(prompt):
            self.assertFalse(sock.bound)
            self.event_log.append('confirmed')
            return 'MOVE_RM75'
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = run(build_parser().parse_args(['--live']), lambda: arm,
                       lambda *unused: sock, lambda: self.clock[0],
                       input_func=confirm, stdin_isatty=lambda: True,
                       live_factory=lambda *unused: motion)
        self.assertEqual(code, 0)
        self.assertLess(self.event_log.index('confirmed'),
                        self.event_log.index('udp-bind'))


class PreviewBoundaryTests(unittest.TestCase):
    def test_preview_never_calls_motion_or_stop(self):
        arm = FakeArm()
        arm.rm_movep_canfd = lambda *args: self.fail('preview called motion')
        arm.rm_set_arm_slow_stop = lambda: self.fail('preview called stop')
        clock = [0.0]
        event_log = []
        sock = LiveSocket([(0, packet(0, 0)), (.01, packet(1, .01)),
                           (.02, packet(2, .02, 1))], clock, event_log)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = run(build_parser().parse_args([]), lambda: arm,
                       lambda *unused: sock, lambda: clock[0],
                       live_factory=lambda *unused: self.fail('preview created live controller'))
        self.assertEqual(code, 0)


class RealControllerIntegrationTests(unittest.TestCase):
    def execute(self, events, arm, clock, kinematics):
        event_log = arm.event_log
        sock = LiveSocket(events, clock, event_log)
        controllers = []
        def factory(sdk_arm, read_state, monotonic):
            controller = LiveMotionController(sdk_arm, read_state,
                                              kinematics, monotonic)
            controllers.append(controller)
            return controller
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = run(build_parser().parse_args(['--live']), lambda: arm,
                       lambda *unused: sock, lambda: clock[0],
                       input_func=lambda prompt: 'MOVE_RM75',
                       stdin_isatty=lambda: True, live_factory=factory)
        return code, output.getvalue(), controllers[0]

    def test_active_slow_ik_rejects_second_target_and_stops(self):
        clock = [0.0]
        event_log = []
        arm = ControllerArm(event_log)
        kinematics = ControllerKinematics(clock, slow_on_solve=2)
        events = [(0, packet(0, 0)), (.01, packet(1, .01)),
                  (.02, packet(2, .02, 1)),
                  (.23, packet(3, .23, 1, z=.01))]
        code, output, controller = self.execute(events, arm, clock, kinematics)
        self.assertEqual(code, 0)
        self.assertEqual(len(arm.moves), 1)
        self.assertIn('stale', output)
        self.assertIn('slow-stop', event_log)
        self.assertLess(event_log.index('slow-stop'), event_log.index('robot-delete'))

    def test_slow_stop_keyboard_interrupt_falls_back_before_delete(self):
        clock = [0.0]
        event_log = []
        arm = ControllerArm(event_log)
        arm.slow_interrupt = True
        events = [(0, packet(0, 0)), (.01, packet(1, .01)),
                  (.02, packet(2, .02, 1)), (.03, packet(3, .03, 0))]
        code, output, controller = self.execute(
            events, arm, clock, ControllerKinematics(clock))
        self.assertEqual(code, 0)
        self.assertEqual(event_log.count('slow-stop'), 1)
        self.assertEqual(event_log.count('fast-stop'), 1)
        self.assertLess(event_log.index('slow-stop'), event_log.index('fast-stop'))
        self.assertLess(event_log.index('fast-stop'), event_log.index('robot-delete'))


if __name__ == '__main__':
    unittest.main()
