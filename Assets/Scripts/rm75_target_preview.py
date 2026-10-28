"""PICO target preview with an explicitly armed RM75 live mode."""
import argparse
from contextlib import closing, nullcontext
import math
import socket
import sys
import time
from typing import Callable

from rm75_live_control import (LiveControlFault, LiveLimits, LiveMotionController,
                               MotionOutcome, SdkRm75Kinematics)
from rm75_log_replay import ControlLogWriter
from rm75_preview_core import (ControllerPacket, PreviewConfig, PreviewEngine,
                               finite_number, map_relative_orientation,
                               validate_pose)
from rm75_state_broadcast import StateBroadcaster


class ReadOnlyRobot:
    def __init__(self, arm: object, host: str, port: int):
        self.arm, self.host, self.port = arm, host, port
        self.info = None
        self.last_state = None
        self._connected = False

    def __enter__(self):
        try:
            handle = self.arm.rm_create_robot_arm(self.host, self.port)
            if type(getattr(handle, 'id', None)) is not int or handle.id <= 0:
                raise RuntimeError(f'Robot connection failed: {self.host}:{self.port}')
            self._connected = True
            code, info = self.arm.rm_get_robot_info()
            if (type(code) is not int or code != 0 or not isinstance(info, dict) or
                    type(info.get('arm_dof')) is not int or info['arm_dof'] != 7 or
                    info.get('arm_model') != 'RM_75'):
                raise RuntimeError(f'Expected RM_75 with 7 DOF; SDK returned {code}, {info}')
            self.info = info
            self.read_pose()  # Validate current status before accepting input.
            return self
        except BaseException:
            self.close()
            raise

    def read_state(self) -> dict:
        code, state = self.arm.rm_get_current_arm_state()
        if type(code) is not int or code != 0 or not isinstance(state, dict):
            raise RuntimeError(f'Robot state read failed: {code}')
        joints, errors = state.get('joint'), state.get('err')
        if (not isinstance(joints, list) or len(joints) != 7 or
                not all(map(finite_number, joints))):
            raise RuntimeError('Robot state has invalid joint data')
        if (not isinstance(errors, dict) or type(errors.get('err_len')) is not int or
                errors['err_len'] < 0 or not isinstance(errors.get('err'), list) or
                errors['err_len'] != len(errors['err']) or
                any(type(item) not in (str, int) or item not in (0, '0') for item in errors['err'])):
            raise RuntimeError(f'Robot reports errors or malformed error data: {errors}')
        try:
            pose = validate_pose(state.get('pose'))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self.last_state = {'joint': [float(item) for item in joints], 'pose': pose}
        return {'joint': list(self.last_state['joint']),
                'pose': list(self.last_state['pose'])}

    def read_pose(self) -> list[float]:
        return self.read_state()['pose']

    def get_run_mode(self) -> int:
        result = self.arm.rm_get_arm_run_mode()
        if (not isinstance(result, tuple) or len(result) != 2 or
                type(result[0]) is not int or result[0] != 0 or
                type(result[1]) is not int or result[1] != 1):
            raise RuntimeError(f'Live control requires robot run mode 1; SDK returned {result!r}')
        return result[1]

    def close(self):
        if self._connected:
            self._connected = False
            result = self.arm.rm_delete_robot_arm()
            if result != 0:
                print(f'Robot disconnect returned {result}', file=sys.stderr)

    def __exit__(self, *exc):
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='PICO → RM75 target preview; --live enables fixed-limit position motion.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=('Live modes fix axes=z,-x,y, scale=0.2, timeout=0.25s and '
                'jump-limit=0.10m. --live-6dof uses radius=0.05m and '
                'speed=0.02m/s; preview options cannot override live limits.'))
    parser.add_argument('--live', action='store_true',
                        help='Enable fixed-limit live motion after interactive confirmation')
    parser.add_argument('--orientation-preview', action='store_true',
                        help='Map right-controller rotation in preview mode only')
    parser.add_argument('--live-orientation-axis', choices=('x', 'y', 'z'),
                        help=('Experimental live orientation mode: lock XYZ and map only '
                              'one robot-base rotation axis'))
    parser.add_argument('--live-6dof', action='store_true',
                        help='Live XYZ + full relative orientation control')
    parser.add_argument('--robot-ip', default='192.168.1.18', help='Robot IP')
    parser.add_argument('--robot-port', type=int, default=8080, help='Robot TCP port')
    parser.add_argument('--bind', default='0.0.0.0', help='UDP listen address')
    parser.add_argument('--port', type=int, default=5005, help='UDP listen port')
    parser.add_argument('--state-port', type=int, default=5006,
                        help='Unity/PICO robot-state UDP destination port')
    parser.add_argument('--scale', type=float, default=.2, help='Preview position multiplier')
    parser.add_argument('--axes', default='z,-x,y', help='Preview PICO axes mapped to robot X,Y,Z')
    parser.add_argument('--radius', type=float, default=.03, help='Preview workspace radius, meters')
    parser.add_argument('--speed', type=float, default=.01, help='Preview target speed limit, m/s')
    parser.add_argument('--timeout', type=float, default=.25, help='Preview packet timeout, seconds')
    parser.add_argument('--jump-limit', type=float, default=.10,
                        help='Preview controller packet jump limit, meters')
    parser.add_argument('--control-log',
                        help='Write deterministic JSONL frames for offline replay')
    return parser


def confirm_live(input_func: Callable[[str], str],
                 stdin_isatty: Callable[[], bool],
                 expected: str = 'MOVE_RM75') -> None:
    if not stdin_isatty():
        raise RuntimeError('Live control requires an interactive terminal')
    answer = input_func(f'Type {expected} exactly to enable live robot motion: ')
    if answer != expected:
        raise RuntimeError('Live control confirmation rejected')


def run(args: argparse.Namespace, arm_factory: Callable,
        socket_factory: Callable, clock: Callable, input_func=input,
        stdin_isatty=sys.stdin.isatty, live_factory=None,
        state_factory=None, orientation_factory=None) -> int:
    try:
        if args.live_orientation_axis and not args.live:
            raise ValueError('--live-orientation-axis requires --live')
        if args.live_6dof and not args.live:
            raise ValueError('--live-6dof requires --live')
        if args.live_6dof and args.live_orientation_axis:
            raise ValueError('--live-6dof cannot be combined with --live-orientation-axis')
        if args.live and args.orientation_preview:
            raise ValueError('--orientation-preview is read-only and cannot be used with --live')
        if args.live:
            expected = {'scale': .2, 'axes': 'z,-x,y', 'radius': .03,
                        'speed': .01, 'timeout': .25, 'jump_limit': .1}
            changed = [name for name, value in expected.items()
                       if getattr(args, name) != value]
            if changed:
                raise ValueError('Live mode does not allow overriding preview options: ' +
                                 ', '.join(changed))
            if args.live_6dof:
                config = PreviewConfig(
                    scale=.2, axes=('z', '-x', 'y'), radius=.05,
                    speed=.02, timeout=.25, jump_limit=.1,
                    orientation_radius=math.radians(10),
                    orientation_jump_limit=math.radians(3),
                    orientation_speed=.05,
                    lock_position=False,
                    require_verified_tracking=True)
            elif args.live_orientation_axis:
                config = PreviewConfig(
                    scale=.2, axes=('z', '-x', 'y'), radius=.01,
                    speed=.005, timeout=.25, jump_limit=.1,
                    orientation_radius=math.radians(2),
                    orientation_jump_limit=math.radians(3),
                    orientation_speed=.05,
                    lock_position=True,
                    require_verified_tracking=True)
            else:
                config = PreviewConfig(scale=.2, axes=('z', '-x', 'y'), radius=.05,
                                       speed=.02, timeout=.25, jump_limit=.1)
            live_limits = LiveLimits()
        else:
            config = PreviewConfig(scale=args.scale, axes=tuple(args.axes.split(',')),
                                   radius=args.radius, speed=args.speed,
                                   timeout=args.timeout, jump_limit=args.jump_limit,
                                   orientation_speed=(.5 if args.orientation_preview
                                                      else None))
        orientation_enabled = bool(args.orientation_preview or
                                   args.live_orientation_axis or
                                   args.live_6dof)
        if orientation_enabled and orientation_factory is None:
            raise RuntimeError('Orientation preview converter is unavailable')
        if args.live_orientation_axis:
            map_orientation = orientation_factory(
                config.axes, args.live_orientation_axis)
        elif args.live_6dof:
            map_orientation = orientation_factory(config.axes)
        elif args.orientation_preview:
            map_orientation = orientation_factory(config.axes)
        else:
            map_orientation = None
        if any(type(port) is not int or not 1 <= port <= 65535
               for port in (args.port, args.robot_port, args.state_port)):
            raise ValueError('Ports must be between 1 and 65535')
        log_context = (ControlLogWriter(
            args.control_log, config,
            warn=lambda message: print(message, file=sys.stderr, flush=True),
            orientation_axis=args.live_orientation_axis)
            if args.control_log else nullcontext())
        with closing(socket_factory(socket.AF_INET, socket.SOCK_DGRAM)) as udp, \
                log_context as control_log:
            def bind_udp():
                try:
                    udp.bind((args.bind, args.port))
                except OSError as exc:
                    raise RuntimeError(f'Cannot bind UDP {args.bind}:{args.port}: {exc}. '
                                       'Stop pico_udp_receiver.py if it is using this port.') from exc
                udp.settimeout(min(.05, config.timeout / 2))

            if not args.live:
                bind_udp()
            state_output = None
            try:
                with ReadOnlyRobot(arm_factory(), args.robot_ip, args.robot_port) as robot:
                    motion = None
                    if args.live:
                        if robot.info.get('force_type') != '6FB':
                            raise RuntimeError('Live control requires RM75 6FB force type')
                        robot.get_run_mode()
                        if args.live_6dof:
                            confirmation = 'MOVE_RM75_6DOF'
                        elif args.live_orientation_axis:
                            confirmation = ('MOVE_RM75_ORIENTATION_' +
                                            args.live_orientation_axis.upper())
                        else:
                            confirmation = 'MOVE_RM75'
                        confirm_live(input_func, stdin_isatty,
                                     expected=confirmation)
                        if live_factory is None:
                            raise RuntimeError('Live motion controller is unavailable')
                        motion = live_factory(robot.arm, robot.read_state, clock)
                        bind_udp()  # Establish a fresh input boundary after confirmation.
                    print(f'Robot: {robot.info}', flush=True)
                    if args.live_6dof:
                        mode = 'LIVE 6DOF ARMED'
                    elif args.live_orientation_axis:
                        mode = ('LIVE ORIENTATION ' +
                                args.live_orientation_axis.upper() + ' ARMED')
                    else:
                        mode = 'LIVE ARMED' if args.live else 'PREVIEW ONLY'
                    print(f'{mode} | UDP {args.bind}:{args.port} | axes={",".join(config.axes)} '
                          f'scale={config.scale} radius={config.radius}m '
                          f'speed={config.speed}m/s', flush=True)
                    if args.live:
                        print('LIVE LIMITS | '
                              f'deadband={live_limits.deadband}m '
                              f'packet_timeout={config.timeout}s jump={config.jump_limit}m '
                              f'ik_step={live_limits.max_joint_step}deg '
                              f'joint_excursion={live_limits.max_joint_excursion}deg '
                              f'monitor={live_limits.monitor_interval}s '
                              f'position_error={live_limits.max_position_error}m '
                              f'joint_error={live_limits.max_joint_error}deg '
                              f'state_timeout={live_limits.max_state_read_time}s', flush=True)
                    reference_capture = [None]

                    def read_reference_pose():
                        pose = robot.read_pose()
                        reference_capture[0] = list(pose)
                        return pose

                    engine = PreviewEngine(config, read_reference_pose,
                                           map_orientation=map_orientation)
                    if state_factory is not None:
                        state_output = state_factory(args.state_port)
                    last_status = None
                    last_print = float('-inf')
                    last_error_print = float('-inf')
                    last_state_error_print = float('-inf')
                    last_live_target = None
                    last_actual = None
                    next_state_read = clock() + .20

                    def publish_state(state):
                        nonlocal last_state_error_print
                        if state_output is None:
                            return
                        peer = engine.source_peer
                        try:
                            state_output.publish(state, None if peer is None else peer[0])
                        except Exception as exc:
                            now = clock()
                            if now - last_state_error_print >= 1:
                                print(f'STATE OUTPUT: {exc}', file=sys.stderr, flush=True)
                                last_state_error_print = now

                    publish_state(robot.last_state)
                    try:
                        while True:
                            target = None
                            frame_payload = None
                            frame_peer = None
                            frame_received = clock()
                            frame_completed = frame_received
                            reference_capture[0] = None
                            motion_outcome = None
                            was_engaged = engine.engaged
                            command_checkpoint = (engine.command_state()
                                                  if engine.engaged else None)
                            try:
                                data, peer = udp.recvfrom(4096)
                            except socket.timeout:
                                frame_received = clock()
                                frame_completed = frame_received
                                engine.tick(frame_received)
                            else:
                                received = clock()
                                frame_payload = data
                                frame_peer = peer
                                frame_received = received
                                try:
                                    packet = ControllerPacket.parse(data)
                                except ValueError as exc:
                                    engine.disarm(f'Invalid packet: {exc}')
                                    # Malformed traffic must not flood the operator console.
                                    if received - last_error_print >= 1:
                                        print(f'INPUT: {exc}', flush=True)
                                        last_error_print = received
                                else:
                                    target = engine.accept(packet, received, peer)
                                    completed = clock()
                                    frame_completed = completed
                                    engine.tick(completed)
                                    if completed - received >= config.timeout:
                                        engine.disarm(
                                            'Reference read timeout; release grip before retrying')
                                    if not engine.engaged:
                                        target = None

                            if args.live and was_engaged and not engine.engaged:
                                motion.stop(engine.reason)
                                print(f'STOPPED: {engine.reason}', flush=True)

                            actual = None
                            if (args.live and
                                    (target is not None or
                                     (engine.engaged and motion.has_sent))):
                                try:
                                    if target is not None:
                                        if not motion.prepared:
                                            motion.prepare(robot.last_state)
                                        result = motion.send_target(target, received)
                                        motion_outcome = result.outcome.value
                                        if result.outcome == MotionOutcome.HOLD:
                                            engine.hold(result.reason, received,
                                                        restore=command_checkpoint)
                                            target = None
                                        else:
                                            last_live_target = list(target)
                                    if engine.engaged and motion.has_sent:
                                        actual = motion.monitor(clock())
                                        if actual is not None:
                                            last_actual = actual
                                            publish_state(actual)
                                            next_state_read = clock() + .20
                                    if engine.status == 'ACTIVE':
                                        engine.reason = 'Live motion active'
                                except LiveControlFault as exc:
                                    reason = f'Live control fault: {exc}'
                                    engine.disarm(reason)
                                    motion.stop(reason)
                                    motion_outcome = 'DISARM'
                                    print(f'FAULT: {exc}', flush=True)
                                    target = None

                            if control_log is not None:
                                control_log.record_frame(
                                    frame_payload, frame_peer, frame_received,
                                    completed=frame_completed,
                                    reference_pose=reference_capture[0],
                                    status=engine.status, reason=engine.reason,
                                    target=target,
                                    motion_outcome=motion_outcome)

                            now = clock()
                            active_live_monitor = (args.live and
                                                   engine.engaged and
                                                   motion.has_sent)
                            if (state_output is not None and not active_live_monitor and
                                    now >= next_state_read):
                                publish_state(robot.read_state())
                                next_state_read = clock() + .20

                            status = (engine.status, engine.reason)
                            if status != last_status:
                                status_name = ('LIVE ACTIVE' if args.live and
                                               engine.status == 'ACTIVE' else engine.status)
                                print(f'{status_name}: {engine.reason}', flush=True)
                                last_status = status
                            now = clock()
                            display_target = (last_live_target if args.live and
                                              engine.engaged else target)
                            if display_target is not None and now - last_print >= .2:
                                orientation_label = ('rpy(rad)' if orientation_enabled
                                                     else 'fixed_rpy(rad)')
                                print('TARGET xyz(m)=(' + ', '.join(
                                    f'{v:.6f}' for v in display_target[:3]) +
                                      f') {orientation_label}=(' + ', '.join(
                                          f'{v:.6f}' for v in display_target[3:]) + ')', flush=True)
                                if args.live and last_actual is not None:
                                    print('ACTUAL xyz(m)=(' + ', '.join(
                                        f'{v:.6f}' for v in last_actual['pose'][:3]) + ')', flush=True)
                                last_print = now
                    finally:
                        if motion is not None and motion.has_sent:
                            motion.stop('Program exit')
            finally:
                if state_output is not None:
                    state_output.close()
    except KeyboardInterrupt:
        mode = 'Live control' if args.live else 'Preview'
        print(f'{mode} stopped; connections closed.', flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    def arm_factory():
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as exc:
            raise RuntimeError('SDK missing; run .venv/bin/python -m pip install Robotic_Arm') from exc
        return RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    def live_factory(arm, read_state, clock):
        try:
            from Robotic_Arm.rm_robot_interface import (
                Algo, rm_force_type_e, rm_inverse_kinematics_params_t,
                rm_robot_arm_model_e)
        except ImportError as exc:
            raise RuntimeError('SDK algorithm support is unavailable') from exc
        algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E,
                    rm_force_type_e.RM_MODEL_RM_ISF_E)
        kinematics = SdkRm75Kinematics(algo, rm_inverse_kinematics_params_t, ik_owner=arm)
        return LiveMotionController(arm, read_state, kinematics, clock)

    def state_factory(port):
        state_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return StateBroadcaster(
            state_udp, time.monotonic, port=port,
            warn=lambda message: print(message, file=sys.stderr, flush=True))

    def orientation_factory(axes, project_axis=None):
        try:
            from Robotic_Arm.rm_robot_interface import (
                Algo, rm_force_type_e, rm_robot_arm_model_e)
        except ImportError as exc:
            raise RuntimeError('SDK orientation conversion support is unavailable') from exc
        algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_75_E,
                    rm_force_type_e.RM_MODEL_RM_ISF_E)
        return lambda reference, current, robot_rpy: map_relative_orientation(
            reference, current, robot_rpy, axes,
            algo.rm_algo_euler2quaternion,
            algo.rm_algo_quaternion2euler,
            project_axis=project_axis)

    return run(args, arm_factory, socket.socket, time.monotonic,
               live_factory=live_factory, state_factory=state_factory,
               orientation_factory=orientation_factory)


if __name__ == '__main__':
    sys.exit(main())
