"""Safety boundary for explicitly armed RM75 TCP-relative live control."""
from dataclasses import dataclass, fields
from enum import Enum
import math
import ctypes
from typing import Callable


class LiveControlFault(RuntimeError):
    """A fault that must disarm live control and stop any sent motion."""


class RecoverableLimit(LiveControlFault):
    """A geometric limit that must hold the last command and permit retreat."""


class MotionOutcome(str, Enum):
    SENT = 'SENT'
    UNCHANGED = 'UNCHANGED'
    HOLD = 'HOLD'


@dataclass(frozen=True)
class MotionResult:
    outcome: MotionOutcome
    reason: str


def _finite_list(value: object, length: int, label: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != length or
            any(type(item) not in (int, float) or not math.isfinite(item)
                for item in value)):
        raise LiveControlFault(f'{label} must contain {length} finite numbers')
    return [float(item) for item in value]


def _state(value: object) -> dict:
    if not isinstance(value, dict):
        raise LiveControlFault('Robot state must be a dictionary')
    return {'joint': _finite_list(value.get('joint'), 7, 'Robot joints'),
            'pose': _finite_list(value.get('pose'), 6, 'Robot pose')}


def _distance(left: list[float], right: list[float], count: int) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2
                         for index in range(count)))


def _rpy_quaternion(pose: list[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = pose[3:]
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def _orientation_distance(left: list[float], right: list[float]) -> float:
    left_quaternion = _rpy_quaternion(left)
    right_quaternion = _rpy_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(left_quaternion, right_quaternion)))
    return 2 * math.acos(min(1.0, dot))


@dataclass(frozen=True)
class LiveLimits:
    deadband: float = 0.0002
    orientation_deadband: float = math.radians(0.3)
    max_joint_step: float = 3.0
    max_joint_excursion: float = 30.0
    monitor_interval: float = 0.20
    max_position_error: float = 0.015
    max_orientation_error: float = math.radians(5.0)
    max_joint_error: float = 10.0
    max_state_read_time: float = 0.25
    max_command_age: float = 0.25

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{item.name} must be a finite positive number')


class SdkRm75Kinematics:
    """Validate the RealMan RM75 arm-angle kinematics API boundary."""

    def __init__(self, algo: object, params_type: type,
                 ik_owner: object = None):
        self._algo = algo
        self._params_type = params_type
        self._ik_owner = ik_owner if ik_owner is not None else algo
        self._remote_ik_initialized = False

        # RM75 连续遥操作：
        # 使用 single-step redundant-parameter solving，
        # 避免连续小位姿变化时发生不必要的构型跳变。
        setter = getattr(
            self._algo,
            'rm_algo_set_redundant_parameter_traversal_mode',
            None
        )
        if setter is not None:
            setter(False)

    @property
    def uses_remote_ik(self) -> bool:
        """True when RealMan dedicated teleoperation IK is available."""
        return (
            callable(getattr(self._algo, 'rm_algo_ik_remote_init', None))
            and callable(getattr(self._algo, 'rm_algo_ik_remote', None))
            and callable(getattr(self._algo, 'rm_algo_pos2matrix', None))
        )

    def calculate_arm_angle(self, joints: list[float]) -> float:
        joints = _finite_list(joints, 7, 'Arm-angle joint input')
        try:
            result = self._algo.rm_algo_calculate_arm_angle_from_config_rm75(joints)
        except Exception as exc:
            raise LiveControlFault(f'RM75 arm-angle calculation failed: {exc}') from exc
        if (not isinstance(result, tuple) or len(result) != 2 or
                type(result[0]) is not int or result[0] != 0 or
                type(result[1]) not in (int, float) or
                not math.isfinite(result[1])):
            raise LiveControlFault(f'RM75 arm-angle calculation returned {result!r}')
        return float(result[1])

    def solve(self, seed: list[float], pose: list[float],
              arm_angle: float) -> list[float]:
        seed = _finite_list(seed, 7, 'IK seed')
        pose = _finite_list(pose, 6, 'IK pose')
        if (type(arm_angle) not in (int, float) or
                not math.isfinite(arm_angle)):
            raise LiveControlFault('Arm angle must be finite')
        try:
            params = self._params_type(seed, pose, 1)

            # --------------------------------------------------
            # RM75 continuous teleoperation IK
            #
            # Prefer RealMan's dedicated remote-IK solver.
            # q_in is the last ACCEPTED joint solution, not a
            # rejected HOLD candidate. This keeps the redundant
            # 7-DoF configuration continuous.
            # --------------------------------------------------
            remote_init = getattr(
                self._algo,
                'rm_algo_ik_remote_init',
                None
            )
            remote_ik = getattr(
                self._algo,
                'rm_algo_ik_remote',
                None
            )
            pos2matrix = getattr(
                self._algo,
                'rm_algo_pos2matrix',
                None
            )

            if (callable(remote_init) and
                    callable(remote_ik) and
                    callable(pos2matrix)):

                if not self._remote_ik_initialized:
                    # 50 Hz nominal teleoperation cycle.
                    # 1 = work/base-coordinate relative mode.
                    remote_init(0.02, 1)

                    # Limit RM75 remote-IK joint velocity.
                    # Previous default produced ~4.49 deg/cycle,
                    # which correctly triggered our 3 deg safety HOLD.
                    # Keep the safety limit unchanged and slow the
                    # remote IK itself instead.
                    dq_setter = getattr(
                        self._algo,
                        'rm_algo_set_dq_weight',
                        None
                    )
                    if callable(dq_setter):
                        dq_setter([0.4] * 7)

                    self._remote_ik_initialized = True

                # rm_algo_pos2matrix() returns rm_matrix_t,
                # while rm_algo_ik_remote() requires rm_Mat_t.
                source_matrix = pos2matrix(pose)

                try:
                    from Robotic_Arm.rm_robot_interface import rm_Mat_t

                    matrix = rm_Mat_t()
                    matrix.row = 4
                    matrix.col = 4

                    def _matrix_read(obj, i, j):
                        """Read either 2-D or flat ctypes matrix storage."""
                        data = obj.data

                        # First try normal 2-D representation.
                        try:
                            return float(data[i][j])
                        except (TypeError, IndexError, AttributeError):
                            pass

                        # Some SDK Python wrappers expose C matrices flat.
                        length = len(data)

                        # Typical layouts:
                        # rm_matrix_t : 4 x 4  -> length 16
                        # rm_Mat_t    : 18 x 18 -> length 324
                        side = int(math.isqrt(length))

                        if side * side == length:
                            stride = side
                        else:
                            stride = int(getattr(obj, 'col', 4))
                            if stride <= 0:
                                stride = 4

                        return float(data[i * stride + j])

                    def _matrix_write(obj, i, j, value):
                        """Write either 2-D or flat ctypes matrix storage."""
                        data = obj.data
                        value = float(value)

                        try:
                            data[i][j] = value
                            return
                        except (TypeError, IndexError, AttributeError):
                            pass

                        length = len(data)
                        side = int(math.isqrt(length))

                        if side * side == length:
                            stride = side
                        else:
                            # rm_Mat_t underlying storage is 18 x 18.
                            stride = 18 if length >= 18 else 4

                        data[i * stride + j] = value

                    for i in range(4):
                        for j in range(4):
                            _matrix_write(
                                matrix,
                                i,
                                j,
                                _matrix_read(source_matrix, i, j)
                            )

                except LiveControlFault:
                    raise
                except Exception as exc:
                    raise LiveControlFault(
                        'RM75 remote IK matrix conversion failed: '
                        f'{exc}'
                    ) from exc

                q_out = (ctypes.c_float * 7)()

                code = remote_ik(
                    matrix,
                    seed,
                    q_out
                )

                if code == 0:
                    result = (
                        0,
                        [float(q_out[i]) for i in range(7)]
                    )

                elif code == -1:
                    raise RecoverableLimit(
                        'RM75 teleoperation IK failed; '
                        'reverse controller to resume'
                    )

                elif code == -2:
                    raise RecoverableLimit(
                        'RM75 teleoperation IK reached '
                        'joint/kinematic limit; '
                        'reverse controller to resume'
                    )

                else:
                    raise LiveControlFault(
                        f'RM75 teleoperation IK returned '
                        f'{code!r}'
                    )

            else:
                # ----------------------------------------------
                # Compatibility fallback:
                # old SDK / existing FakeRobot tests continue
                # using the previously validated IK path.
                # ----------------------------------------------
                ik_fn = getattr(
                    self._ik_owner,
                    'rm_algo_inverse_kinematics',
                    None
                )

                if callable(ik_fn):
                    result = ik_fn(params)

                else:
                    result = (
                        self._algo
                        .rm_algo_inverse_kinematics_rm75_for_arm_angle(
                            params,
                            float(arm_angle)
                        )
                    )

        except LiveControlFault:
            raise
        except Exception as exc:
            raise LiveControlFault(f'RM75 inverse kinematics failed: {exc}') from exc
        if (not isinstance(result, tuple) or len(result) != 2 or
                type(result[0]) is not int or result[0] != 0):
            raise LiveControlFault(f'RM75 inverse kinematics returned {result!r}')

        solution = _finite_list(result[1], 7, 'IK solution')

        # Active singularity guard:
        # Check the candidate RM75 joint configuration BEFORE any Cartesian
        # command is sent to the real robot.
        singularity_fn = getattr(
            self._algo, 'rm_algo_universal_singularity_analyse', None
        )
        if not callable(singularity_fn):
            raise LiveControlFault(
                'RM75 SDK does not expose rm_algo_universal_singularity_analyse; '
                'refusing live motion without singularity protection'
            )

        try:
            singularity = singularity_fn(solution, 0.01)
        except Exception as exc:
            raise LiveControlFault(
                f'RM75 singularity analysis failed: {exc}'
            ) from exc

        if singularity == -1:
            # Diagnostic only: expose the rejected candidate configuration.
            # No safety behavior is changed.
            candidate_text = ', '.join(
                f'J{i + 1}={value:.3f}deg'
                for i, value in enumerate(solution)
            )
            print(
                'SINGULAR CANDIDATE | ' + candidate_text,
                flush=True
            )

            # Show how conservative the classification is.
            threshold_results = []
            for threshold in (0.001, 0.002, 0.005, 0.008,
                              0.010, 0.015, 0.020):
                try:
                    result_at_threshold = singularity_fn(
                        solution, threshold
                    )
                except Exception:
                    result_at_threshold = 'ERR'

                threshold_results.append(
                    f'{threshold:.3f}:{result_at_threshold}'
                )

            print(
                'SINGULAR THRESHOLDS | ' +
                ' '.join(threshold_results),
                flush=True
            )

            raise RecoverableLimit(
                'RM75 candidate IK solution entered singular region; '
                'move away from this configuration'
            )
        if singularity != 0:
            raise LiveControlFault(
                f'RM75 singularity analysis returned {singularity!r}'
            )

        # Joint-limit guard.
        # Keep every candidate joint at least 5 degrees away from
        # the algorithm-configured joint limits.
        try:
            joint_min = self._algo.rm_algo_get_joint_min_limit()
            joint_max = self._algo.rm_algo_get_joint_max_limit()
        except Exception as exc:
            raise LiveControlFault(
                f'RM75 joint-limit query failed: {exc}'
            ) from exc

        joint_min = _finite_list(joint_min, 7, 'Joint minimum limits')
        joint_max = _finite_list(joint_max, 7, 'Joint maximum limits')

        margin = 5.0
        for index, (q, qmin, qmax) in enumerate(
                zip(solution, joint_min, joint_max), start=1):
            if q <= qmin + margin:
                raise RecoverableLimit(
                    f'J{index} too close to minimum joint limit: '
                    f'{q:.3f}deg <= {qmin + margin:.3f}deg safety boundary'
                )
            if q >= qmax - margin:
                raise RecoverableLimit(
                    f'J{index} too close to maximum joint limit: '
                    f'{q:.3f}deg >= {qmax - margin:.3f}deg safety boundary'
                )

        return solution


class LiveMotionController:
    def __init__(self, arm: object, read_state: Callable[[], dict],
                 kinematics: object, clock: Callable[[], float],
                 limits: LiveLimits = LiveLimits()):
        self._arm = arm
        self._read_state = read_state
        self._kinematics = kinematics
        self._clock = clock
        self._limits = limits
        self.prepared = False
        self.has_sent = False
        self.last_target = None
        self.last_solution = None
        self._reference_solution = None
        self.stop_reason = None
        self._arm_angle = None
        self._next_monitor = None
        self._stop_complete = False
        self._slow_stop_attempted = False
        self._fast_stop_attempted = False
        self.last_result = None

    def prepare(self, state: dict) -> None:
        measured = _state(state)
        self.prepared = False
        self.has_sent = False
        self.last_target = None
        self.last_solution = None
        self._reference_solution = None
        self.stop_reason = None
        self._arm_angle = None
        self._next_monitor = None
        self._stop_complete = False
        self._slow_stop_attempted = False
        self._fast_stop_attempted = False
        self.last_result = None
        try:
            arm_angle = self._kinematics.calculate_arm_angle(measured['joint'])
        except LiveControlFault:
            raise
        except Exception as exc:
            raise LiveControlFault(f'RM75 arm-angle calculation failed: {exc}') from exc
        if type(arm_angle) not in (int, float) or not math.isfinite(arm_angle):
            raise LiveControlFault('RM75 arm-angle calculation returned an invalid value')
        self._arm_angle = float(arm_angle)
        self.last_solution = measured['joint']
        self._reference_solution = list(measured['joint'])
        self.prepared = True

    def send_target(self, pose: list[float], now: float) -> MotionResult:
        if not self.prepared:
            raise LiveControlFault('Live motion is not prepared')
        if type(now) not in (int, float) or not math.isfinite(now):
            raise LiveControlFault('Target receive time must be finite')
        target = _finite_list(pose, 6, 'Live target pose')
        # Dedicated remote IK is a continuous, cycle-by-cycle solver.
        # Even when the desired Cartesian target is unchanged, it must
        # continue solving from the CURRENT measured joint state until
        # the robot converges to the target.
        #
        # The old deadband shortcut is valid only for ordinary one-shot IK.
        remote_mode = bool(
            getattr(self._kinematics, 'uses_remote_ik', False)
        )

        changed = True
        if ((not remote_mode) and
                self.last_target is not None and
                _distance(target, self.last_target, 3) < self._limits.deadband and
                _orientation_distance(target, self.last_target) <
                self._limits.orientation_deadband):
            target = list(self.last_target)
            solution = list(self.last_solution)
            changed = False
        else:
            # --------------------------------------------------
            # Dedicated RealMan remote IK requires q_in to be the
            # CURRENT measured joint configuration.
            #
            # Do NOT use last_solution here:
            # last_solution is the previous commanded solution and
            # the real robot may still be following it.
            # --------------------------------------------------
            ik_seed = list(self.last_solution)

            if remote_mode:
                read_started = self._clock()

                try:
                    measured_for_ik = _state(self._read_state())
                except LiveControlFault:
                    raise
                except Exception as exc:
                    raise LiveControlFault(
                        f'Robot state read for remote IK failed: {exc}'
                    ) from exc

                read_finished = self._clock()
                read_duration = read_finished - read_started

                if (not math.isfinite(read_duration)
                        or read_duration < 0
                        or read_duration >=
                        self._limits.max_state_read_time):
                    raise LiveControlFault(
                        'Robot state read for remote IK exceeded '
                        f'{self._limits.max_state_read_time:g}s limit '
                        f'({read_duration!r}s)'
                    )

                ik_seed = list(measured_for_ik['joint'])

            try:
                solution = self._kinematics.solve(
                    ik_seed, target, self._arm_angle)
            except RecoverableLimit as exc:
                return self._hold(str(exc))
            except LiveControlFault:
                raise
            except Exception as exc:
                raise LiveControlFault(f'RM75 inverse kinematics failed: {exc}') from exc
            solution = _finite_list(solution, 7, 'IK solution')
            step = max(abs(current - previous)
                       for current, previous in zip(solution, ik_seed))
            if step > self._limits.max_joint_step:
                return self._hold(
                    f'IK joint step {step:.6f}° exceeds '
                    f'{self._limits.max_joint_step:g}° limit')
            excursion = max(abs(current - reference)
                            for current, reference in zip(
                                solution, self._reference_solution))
            if excursion > self._limits.max_joint_excursion:
                return self._hold(
                    f'IK joint excursion {excursion:.6f}° exceeds '
                    f'{self._limits.max_joint_excursion:g}° limit')

        command_age = self._clock() - now
        if (not math.isfinite(command_age) or
                command_age >= self._limits.max_command_age):
            raise LiveControlFault(
                f'Target is stale: age {command_age!r}s reached '
                f'{self._limits.max_command_age:g}s limit')

        self.has_sent = True
        try:
            # RM75 continuous joint following of the validated IK solution.
            command_name = 'rm_movej_follow'
            result = self._arm.rm_movej_follow(solution)
        except Exception as exc:
            raise LiveControlFault(f'{command_name} command raised: {exc}') from exc
        if type(result) is not int or result != 0:
            raise LiveControlFault(f'{command_name} command returned {result!r}')
        self.last_target = target
        self.last_solution = solution
        if self._next_monitor is None:
            self._next_monitor = float(now) + self._limits.monitor_interval
        outcome = MotionOutcome.SENT if changed else MotionOutcome.UNCHANGED
        self.last_result = MotionResult(outcome, 'Command accepted')
        return self.last_result

    def _hold(self, reason: str) -> MotionResult:
        self.last_result = MotionResult(MotionOutcome.HOLD, str(reason))
        return self.last_result

    def monitor(self, now: float) -> dict | None:
        if (not self.prepared or not self.has_sent or
                self._next_monitor is None or now < self._next_monitor):
            return None
        started = self._clock()
        try:
            measured = _state(self._read_state())
        except LiveControlFault:
            raise
        except Exception as exc:
            raise LiveControlFault(f'Robot state read failed: {exc}') from exc
        finished = self._clock()
        duration = finished - started
        if (not math.isfinite(duration) or duration < 0 or
                duration >= self._limits.max_state_read_time):
            raise LiveControlFault(
                f'Robot state read exceeded {self._limits.max_state_read_time:g}s limit '
                f'({duration!r}s)')
        position_error = _distance(measured['pose'], self.last_target, 3)
        if position_error > self._limits.max_position_error:
            raise LiveControlFault(
                f'Robot position error {position_error:.6f}m exceeds '
                f'{self._limits.max_position_error:g}m limit')
        orientation_error = _orientation_distance(measured['pose'], self.last_target)
        if orientation_error > self._limits.max_orientation_error:
            raise LiveControlFault(
                f'Robot orientation error {math.degrees(orientation_error):.6f}° exceeds '
                f'{math.degrees(self._limits.max_orientation_error):g}° limit')
        joint_error = max(abs(actual - expected) for actual, expected in zip(
            measured['joint'], self.last_solution))
        if joint_error > self._limits.max_joint_error:
            raise LiveControlFault(
                f'Robot joint error {joint_error:.6f}° exceeds '
                f'{self._limits.max_joint_error:g}° limit')
        self._next_monitor = finished + self._limits.monitor_interval
        return measured

    def stop(self, reason: str) -> None:
        if self._stop_complete:
            return
        should_stop = self.has_sent
        self.stop_reason = str(reason)
        if not should_stop:
            self._finish_stop()
            return

        slow_ok = False
        slow_base_exception = None
        if not self._slow_stop_attempted:
            try:
                slow_result = self._arm.rm_set_arm_slow_stop()
            except Exception:
                self._slow_stop_attempted = True
            except BaseException as exc:
                self._slow_stop_attempted = True
                slow_base_exception = exc
            else:
                self._slow_stop_attempted = True
                slow_ok = type(slow_result) is int and slow_result == 0
        if slow_ok:
            self._finish_stop()
            return

        if self._fast_stop_attempted:
            self._finish_stop()
            raise LiveControlFault('Fast stop was interrupted; completion is not confirmed')
        try:
            fast_result = self._arm.rm_set_arm_stop()
        except BaseException as exc:
            self._fast_stop_attempted = True
            self._finish_stop()
            raise LiveControlFault(f'Fast stop raised: {exc}') from exc
        self._fast_stop_attempted = True
        if type(fast_result) is not int or fast_result != 0:
            self._finish_stop()
            raise LiveControlFault(f'Fast stop returned {fast_result!r}')
        self._finish_stop()
        if slow_base_exception is not None:
            raise slow_base_exception

    def _finish_stop(self) -> None:
        self.prepared = False
        self.has_sent = False
        self.last_target = None
        self.last_solution = None
        self._reference_solution = None
        self._arm_angle = None
        self._next_monitor = None
        self._stop_complete = True
        self.last_result = None
