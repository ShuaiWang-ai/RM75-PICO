import contextlib
import copy
import io
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Assets' / 'Scripts'))
from rm75_target_preview import ReadOnlyRobot, build_parser, confirm_live, run
from test_rm75_preview_core import POSE, PEER

STATE = dict(joint=[.438, .125, .3, .093, -1.961, -4.363, -.004],
             pose=POSE, err={'err_len': 1, 'err': ['0']})


class FakeArm:
    """Only the SDK's four read/connection methods exist; motion calls fail."""
    def __init__(self):
        self.info = (0, dict(arm_dof=7, arm_model='RM_75', force_type='B', robot_controller_version=3))
        self.state = (0, copy.deepcopy(STATE))
        self.handle_id = 1
        self.reads = 0
        self.closed = 0
        self.read_hook = None
        self.run_mode = (0, 1)

    def rm_create_robot_arm(self, host, port):
        if (host, port) != ('192.168.1.18', 8080):
            raise AssertionError('Wrong default robot endpoint')
        return SimpleNamespace(id=self.handle_id)

    def rm_get_robot_info(self):
        return self.info

    def rm_get_current_arm_state(self):
        self.reads += 1
        if self.read_hook:
            self.read_hook(self.reads)
        return copy.deepcopy(self.state)

    def rm_get_arm_run_mode(self):
        return self.run_mode

    def rm_delete_robot_arm(self):
        self.closed += 1
        return 0


class FakeSocket:
    def __init__(self, events, clock, bind_error=None):
        self.events = iter(events)
        self.clock = clock
        self.bind_error = bind_error
        self.closed = False

    def bind(self, address):
        if self.bind_error:
            raise self.bind_error
        if address != ('0.0.0.0', 5005):
            raise AssertionError('Wrong default UDP endpoint')

    def settimeout(self, timeout):
        if not 0 < timeout <= .05:
            raise AssertionError('Watchdog polling too slow')

    def recvfrom(self, size):
        try:
            now, payload = next(self.events)
        except StopIteration:
            raise KeyboardInterrupt
        self.clock[0] = now
        if payload is None:
            raise socket.timeout()
        return payload, PEER

    def close(self):
        self.closed = True


class FakeStateBroadcaster:
    def __init__(self):
        self.published = []
        self.closed = False

    def publish(self, state, pico_ip):
        self.published.append((copy.deepcopy(state), pico_ip))

    def close(self):
        self.closed = True


def payload(seq, now, grip=0, **changes):
    data = dict(session='a', sequence=seq, sent_time=now, x=0, y=0, z=0,
                qx=0, qy=0, qz=0, qw=1, trigger=0, grip=grip, valid=True)
    data.update(changes)
    return json.dumps(data).encode()


class AdapterTests(unittest.TestCase):
    def test_good_state_and_cleanup(self):
        arm = FakeArm()
        with ReadOnlyRobot(arm, '192.168.1.18', 8080) as robot:
            self.assertEqual(robot.read_pose(), POSE)
            self.assertEqual(robot.last_state, {'joint': STATE['joint'], 'pose': POSE})
            self.assertEqual(robot.read_state(), {'joint': STATE['joint'], 'pose': POSE})
        self.assertEqual(arm.closed, 1)

    def test_run_mode(self):
        arm = FakeArm()
        with ReadOnlyRobot(arm, '192.168.1.18', 8080) as robot:
            self.assertEqual(robot.get_run_mode(), 1)
            for response in ((-1, 1), (0, 0), (0, True), 'bad'):
                arm.run_mode = response
                with self.subTest(response=response), self.assertRaises(RuntimeError):
                    robot.get_run_mode()

    def test_failed_connect_not_read(self):
        arm = FakeArm()
        arm.handle_id = -1
        with self.assertRaises(RuntimeError):
            with ReadOnlyRobot(arm, '192.168.1.18', 8080):
                self.fail('Invalid connection accepted')
        self.assertEqual(arm.reads, 0)

    def test_wrong_model_and_dof(self):
        for changes in [dict(arm_model='RM_65'), dict(arm_dof=6), dict(arm_dof=True)]:
            arm = FakeArm()
            arm.info[1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                with ReadOnlyRobot(arm, '192.168.1.18', 8080):
                    self.fail('Wrong robot accepted')
            self.assertEqual(arm.closed, 1)

    def test_failed_info_read_closes(self):
        arm = FakeArm()
        arm.info = (-2, {})
        with self.assertRaises(RuntimeError):
            with ReadOnlyRobot(arm, '192.168.1.18', 8080):
                pass
        self.assertEqual(arm.closed, 1)

    def test_state_failures_and_cleanup(self):
        variants = [(-2, {}), (0, dict(STATE, pose=[float('nan')]*6)),
                    (0, dict(STATE, joint=[0]*6)),
                    (0, dict(STATE, joint=[float('inf')]*7)),
                    (0, dict(STATE, err={'err_len': 1, 'err': ['42']})),
                    (0, dict(STATE, err={'err_len': 2, 'err': ['0']})),
                    (0, dict(STATE, err={})), (0, dict(STATE, err=None))]
        for state in variants:
            arm = FakeArm()
            arm.state = state
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                with ReadOnlyRobot(arm, '192.168.1.18', 8080):
                    pass
            self.assertEqual(arm.closed, 1)

    def test_empty_error_list_is_valid(self):
        arm = FakeArm()
        arm.state[1]['err'] = {'err_len': 0, 'err': []}
        with ReadOnlyRobot(arm, '192.168.1.18', 8080) as robot:
            self.assertEqual(robot.read_pose(), POSE)

    def test_cleanup_on_body_exception(self):
        arm = FakeArm()
        with self.assertRaises(ValueError):
            with ReadOnlyRobot(arm, '192.168.1.18', 8080):
                raise ValueError('operator error')
        self.assertEqual(arm.closed, 1)


class CliTests(unittest.TestCase):
    def execute(self, events, arm=None, bind_error=None, with_state=False):
        self.arm = arm or FakeArm()
        self.clock = [0.0]
        self.sock = FakeSocket(events, self.clock, bind_error)
        self.state = FakeStateBroadcaster()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = run(build_parser().parse_args([]), lambda: self.arm,
                         lambda *args: self.sock, lambda: self.clock[0],
                         state_factory=(lambda port: self.state) if with_state else None)
        return result, output.getvalue()

    def test_preview_and_read_only_cleanup(self):
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                     (.02, payload(2, .02, 1)), (.23, payload(3, .23, 1, z=.01))])
        self.assertEqual(code, 0)
        self.assertIn('ACTIVE', output)
        self.assertIn('TARGET', output)
        self.assertIn('0.100500', output)
        self.assertEqual(self.arm.reads, 2)  # Startup and fresh grip reference.
        self.assertEqual(self.arm.closed, 1)
        self.assertTrue(self.sock.closed)

    def test_malformed_packet_disarms(self):
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                    (.02, payload(2, .02, 1)), (.03, b'bad'),
                                    (.24, payload(3, .24, 1, z=.01))])
        self.assertEqual(code, 0)
        self.assertEqual(output.count('TARGET'), 1)
        self.assertIn('Invalid', output)

    def test_watchdog_without_datagrams(self):
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                    (.02, payload(2, .02, 1)), (.28, None)])
        self.assertEqual(code, 0)
        self.assertIn('Packet timeout', output)

    def test_state_stream_publishes_startup_and_idle_snapshot(self):
        code, unused = self.execute([(0, payload(0, 0)),
                                     (.01, payload(1, .01)),
                                     (.22, None)], with_state=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.state.published), 2)
        self.assertEqual(self.state.published[0][1], None)
        self.assertEqual(self.state.published[1][1], PEER[0])
        self.assertEqual(self.arm.reads, 2)
        self.assertTrue(self.state.closed)

    def test_idle_state_read_failure_exits_and_closes_broadcaster(self):
        arm = FakeArm()
        def fail_second_read(reads):
            if reads == 2:
                raise RuntimeError('idle state failed')
        arm.read_hook = fail_second_read
        code, output = self.execute([(0, payload(0, 0)),
                                     (.01, payload(1, .01)),
                                     (.22, None)], arm=arm, with_state=True)
        self.assertEqual(code, 1)
        self.assertIn('idle state failed', output)
        self.assertEqual(arm.closed, 1)
        self.assertTrue(self.state.closed)

    def test_bind_conflict(self):
        code, output = self.execute([], bind_error=OSError(98, 'Address already in use'))
        self.assertEqual(code, 1)
        self.assertIn('pico_udp_receiver.py', output)
        self.assertTrue(self.sock.closed)

    def test_legacy_packet_message(self):
        code, output = self.execute([(0, b'{"valid":true}')])
        self.assertEqual(code, 0)
        self.assertIn('APK', output)
        self.assertNotIn('TARGET', output)

    def test_read_error_exits(self):
        arm = FakeArm()
        def fail_on_engage(reads):
            if reads == 2:
                raise RuntimeError('SDK read failure')
        arm.read_hook = fail_on_engage
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                    (.02, payload(2, .02, 1))], arm)
        self.assertEqual(code, 1)
        self.assertNotIn('TARGET', output)
        self.assertEqual(arm.closed, 1)
        self.assertTrue(self.sock.closed)

    def test_slow_reference_read_invalidates_packet(self):
        arm = FakeArm()
        def delay(reads):
            if reads == 2:
                self.clock[0] += .3
        arm.read_hook = delay
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                    (.02, payload(2, .02, 1))], arm)
        self.assertEqual(code, 0)
        self.assertNotIn('TARGET', output)
        self.assertIn('timeout', output.lower())

    def test_oversized_integer_disarms_and_can_recover(self):
        code, output = self.execute([(0, payload(0, 0)), (.01, payload(1, .01)),
                                    (.02, payload(2, .02, 1)),
                                    (.03, payload(3, .03, 1, x=10**400)),
                                    (.04, payload(4, .04)), (.05, payload(5, .05, 1)),
                                    (.26, payload(6, .26, 1, z=.01))])
        self.assertEqual(code, 0)
        self.assertIn('finite numbers', output)
        self.assertEqual(output.count('TARGET'), 2)
        self.assertEqual(self.arm.reads, 3)
        self.assertTrue(self.sock.closed)
        self.assertEqual(self.arm.closed, 1)

    def test_help_without_sdk(self):
        script = Path(__file__).resolve().parents[1] / 'Assets/Scripts/rm75_target_preview.py'
        result = subprocess.run([sys.executable, '-S', str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--axes', result.stdout)
        self.assertIn('0.25', result.stdout)

    def test_orientation_preview_flag_is_explicit(self):
        try:
            args = build_parser().parse_args(['--orientation-preview'])
        except SystemExit as exc:
            self.fail(f'orientation preview flag is unavailable: {exc}')
        self.assertIs(args.orientation_preview, True)

    def test_live_orientation_axis_flag_is_explicit(self):
        args = build_parser().parse_args(['--live', '--live-orientation-axis', 'x'])
        self.assertEqual(args.live_orientation_axis, 'x')

    def test_live_orientation_axis_requires_live_before_robot_connection(self):
        args = build_parser().parse_args(['--live-orientation-axis', 'x'])
        created = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = run(args, lambda: created.append(True) or FakeArm(),
                         lambda *unused: FakeSocket([], [0.0]), lambda: 0.0)
        self.assertEqual(result, 1)
        self.assertEqual(created, [])
        self.assertIn('requires --live', output.getvalue())

    def test_orientation_preview_cannot_enable_live_motion(self):
        args = build_parser().parse_args(['--live', '--orientation-preview'])
        created = []
        clock = [0.0]
        sock = FakeSocket([], clock)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = run(args, lambda: created.append(True) or FakeArm(),
                         lambda *unused: sock, lambda: clock[0])
        self.assertEqual(result, 1)
        self.assertEqual(created, [])
        self.assertIn('orientation', output.getvalue().lower())

    def test_orientation_preview_outputs_mapped_rpy(self):
        args = build_parser().parse_args(['--orientation-preview'])
        arm = FakeArm()
        clock = [0.0]
        angle = .05
        events = [(0, payload(0, 0)), (.01, payload(1, .01)),
                  (.02, payload(2, .02, 1)),
                  (.23, payload(3, .23, 1, qz=math.sin(angle),
                                 qw=math.cos(angle)))]
        sock = FakeSocket(events, clock)
        factory_axes = []
        mapper_calls = []
        def orientation_factory(axes):
            factory_axes.append(axes)
            def mapper(reference, current, robot_rpy):
                mapper_calls.append((reference, current, robot_rpy))
                return [.4, .5, .6]
            return mapper
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = run(args, lambda: arm, lambda *unused: sock,
                             lambda: clock[0],
                             orientation_factory=orientation_factory)
        except TypeError as exc:
            self.fail(f'orientation preview is not wired into the CLI: {exc}')
        self.assertEqual(result, 0)
        self.assertEqual(factory_axes, [('z', '-x', 'y')])
        self.assertEqual(len(mapper_calls), 1)
        self.assertIn('rpy(rad)=(0.400000, 0.500000, 0.600000)', output.getvalue())
        self.assertNotIn('fixed_rpy', output.getvalue())

    def test_orientation_preview_applies_half_radian_per_second_rate_limit(self):
        args = build_parser().parse_args(['--orientation-preview'])
        arm = FakeArm()
        clock = [0.0]
        requested_angle = .20
        events = [(0, payload(0, 0)), (.01, payload(1, .01)),
                  (.02, payload(2, .02, 1)),
                  (.12, payload(3, .12, 1,
                                 qz=math.sin(requested_angle / 2),
                                 qw=math.cos(requested_angle / 2)))]
        sock = FakeSocket(events, clock)
        mapped_rotations = []
        def orientation_factory(unused_axes):
            def mapper(unused_reference, current, robot_rpy):
                mapped_rotations.append(current)
                return list(robot_rpy)
            return mapper

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run(args, lambda: arm, lambda *unused: sock,
                         lambda: clock[0],
                         orientation_factory=orientation_factory)

        self.assertEqual(result, 0)
        limited_angle = 2 * math.acos(abs(mapped_rotations[0][3]))
        self.assertAlmostEqual(limited_angle, .05, places=6)

    def test_confirmation_requires_tty_and_exact_phrase(self):
        for answer, is_tty in (('move_rm75', True), (' MOVE_RM75', True),
                               ('MOVE_RM75 ', True), ('MOVE_RM75', False)):
            with self.subTest(answer=answer, is_tty=is_tty), self.assertRaises(RuntimeError):
                confirm_live(lambda prompt: answer, lambda: is_tty)
        confirm_live(lambda prompt: 'MOVE_RM75', lambda: True)

    def test_orientation_confirmation_rejects_position_mode_phrase(self):
        with self.assertRaises(RuntimeError):
            confirm_live(lambda prompt: 'MOVE_RM75', lambda: True,
                         expected='MOVE_RM75_ORIENTATION_X')
        confirm_live(lambda prompt: 'MOVE_RM75_ORIENTATION_X', lambda: True,
                     expected='MOVE_RM75_ORIENTATION_X')

    def test_live_rejection_precedes_motion_factory(self):
        args = build_parser().parse_args(['--live'])
        arm = FakeArm()
        arm.info[1]['force_type'] = '6FB'
        clock = [0.0]
        sock = FakeSocket([], clock)
        created = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = run(args, lambda: arm, lambda *unused: sock,
                         lambda: clock[0], input_func=lambda prompt: 'NO',
                         stdin_isatty=lambda: True,
                         live_factory=lambda *items: created.append(items))
        self.assertEqual(result, 1)
        self.assertEqual(created, [])
        self.assertEqual(arm.closed, 1)


if __name__ == '__main__':
    unittest.main()
