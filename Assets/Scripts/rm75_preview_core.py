"""Validated controller packets and target-pose preview math; no robot SDK."""
from dataclasses import dataclass
import json
import math
from typing import Callable


def finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_pose(pose: object) -> list[float]:
    if not isinstance(pose, (list, tuple)) or len(pose) != 6 or not all(map(finite_number, pose)):
        raise ValueError('Robot pose must contain six finite numbers')
    return list(pose)


def _normalized_quaternion(value: object, order: str) -> tuple[float, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            not all(map(finite_number, value))):
        raise ValueError(f'{order} quaternion must contain four finite numbers')
    norm = math.hypot(*value)
    if norm < 1e-6:
        raise ValueError(f'{order} quaternion must be nonzero')
    return tuple(float(item) / norm for item in value)


def _quaternion_matrix_wxyz(quaternion: tuple[float, ...]) -> list[list[float]]:
    w, x, y, z = quaternion
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
         2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
         2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x),
         1 - 2 * (x * x + y * y)],
    ]


def _matrix_multiply(left: list[list[float]],
                     right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[row][index] * right[index][column]
                 for index in range(3))
             for column in range(3)] for row in range(3)]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matrix_quaternion_wxyz(matrix: list[list[float]]) -> tuple[float, ...]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = 2 * math.sqrt(trace + 1)
        quaternion = (scale / 4,
                      (matrix[2][1] - matrix[1][2]) / scale,
                      (matrix[0][2] - matrix[2][0]) / scale,
                      (matrix[1][0] - matrix[0][1]) / scale)
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            scale = 2 * math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2])
            quaternion = ((matrix[2][1] - matrix[1][2]) / scale, scale / 4,
                          (matrix[0][1] + matrix[1][0]) / scale,
                          (matrix[0][2] + matrix[2][0]) / scale)
        elif index == 1:
            scale = 2 * math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2])
            quaternion = ((matrix[0][2] - matrix[2][0]) / scale,
                          (matrix[0][1] + matrix[1][0]) / scale, scale / 4,
                          (matrix[1][2] + matrix[2][1]) / scale)
        else:
            scale = 2 * math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1])
            quaternion = ((matrix[1][0] - matrix[0][1]) / scale,
                          (matrix[0][2] + matrix[2][0]) / scale,
                          (matrix[1][2] + matrix[2][1]) / scale, scale / 4)
    return _normalized_quaternion(quaternion, 'robot')


def _axis_matrix(axes: tuple[str, str, str]) -> list[list[float]]:
    if (len(axes) != 3 or
            any(axis not in ('x', 'y', 'z', '-x', '-y', '-z') for axis in axes) or
            len({axis.lstrip('-') for axis in axes}) != 3):
        raise ValueError('axes must be a signed permutation of x,y,z')
    source_index = {'x': 0, 'y': 1, 'z': 2}
    matrix = [[0.0] * 3 for _ in range(3)]
    for row, axis in enumerate(axes):
        matrix[row][source_index[axis.lstrip('-')]] = -1.0 if axis.startswith('-') else 1.0
    return matrix


def map_relative_orientation(reference_controller: tuple[float, ...],
                             current_controller: tuple[float, ...],
                             reference_robot_rpy: list[float],
                             axes: tuple[str, str, str],
                             euler_to_quaternion: Callable[[list[float]], list[float]],
                             quaternion_to_euler: Callable[[list[float]], list[float]],
                             project_axis: str | None = None) -> list[float]:
    """Map a PICO world-space relative rotation onto the robot base frame."""
    reference_xyzw = _normalized_quaternion(reference_controller, 'controller')
    current_xyzw = _normalized_quaternion(current_controller, 'controller')
    # XRoboToolkit PC Service reports a left-handed X-right/Y-up/Z-in pose.
    # Conjugating its quaternion expresses the same physical rotation with
    # the right-handed matrix convention used by the robot calculations.
    reference_pico = _quaternion_matrix_wxyz(
        (reference_xyzw[3], *(-value for value in reference_xyzw[:3])))
    current_pico = _quaternion_matrix_wxyz(
        (current_xyzw[3], *(-value for value in current_xyzw[:3])))
    relative_pico = _matrix_multiply(current_pico, _transpose(reference_pico))
    basis = _axis_matrix(axes)
    relative_robot = _matrix_multiply(
        _matrix_multiply(basis, relative_pico), _transpose(basis))
    if project_axis is not None:
        if project_axis not in ('x', 'y', 'z'):
            raise ValueError('project_axis must be x, y, or z')
        relative_quaternion = _matrix_quaternion_wxyz(relative_robot)
        if relative_quaternion[0] < 0:
            relative_quaternion = tuple(-value for value in relative_quaternion)
        vector_norm = math.hypot(*relative_quaternion[1:])
        component_angle = 0.0
        if vector_norm > 1e-12:
            total_angle = 2 * math.atan2(vector_norm, relative_quaternion[0])
            axis_index = {'x': 1, 'y': 2, 'z': 3}[project_axis]
            component_angle = (total_angle * relative_quaternion[axis_index] /
                               vector_norm)
        projected = [0.0, 0.0, 0.0]
        projected['xyz'.index(project_axis)] = math.sin(component_angle / 2)
        relative_robot = _quaternion_matrix_wxyz(
            (math.cos(component_angle / 2), *projected))
    if (not isinstance(reference_robot_rpy, (list, tuple)) or
            len(reference_robot_rpy) != 3 or
            not all(map(finite_number, reference_robot_rpy))):
        raise ValueError('Robot RPY must contain three finite numbers')
    reference_robot_quaternion = _normalized_quaternion(
        euler_to_quaternion(list(reference_robot_rpy)), 'robot')
    reference_robot = _quaternion_matrix_wxyz(reference_robot_quaternion)
    target_robot = _matrix_multiply(relative_robot, reference_robot)
    result = quaternion_to_euler(list(_matrix_quaternion_wxyz(target_robot)))
    if (not isinstance(result, (list, tuple)) or len(result) != 3 or
            not all(map(finite_number, result))):
        raise ValueError('Robot quaternion conversion returned invalid RPY')
    return [float(item) for item in result]


def _quaternion_angle(left: tuple[float, ...],
                      right: tuple[float, ...]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return 2 * math.acos(min(1.0, dot))


def _step_quaternion(current: tuple[float, ...], target: tuple[float, ...],
                     max_angle: float) -> tuple[float, ...]:
    """Move toward target by at most max_angle on the quaternion sphere."""
    current = _normalized_quaternion(current, 'controller')
    target = _normalized_quaternion(target, 'controller')
    dot = sum(a * b for a, b in zip(current, target))
    if dot < 0:
        target = tuple(-value for value in target)
        dot = -dot
    dot = min(1.0, max(0.0, dot))
    angle = 2 * math.acos(dot)
    if angle <= max_angle:
        return target
    half_angle = math.acos(dot)
    fraction = max_angle / angle
    if half_angle < 1e-8:
        return target
    scale = math.sin(half_angle)
    left_weight = math.sin((1 - fraction) * half_angle) / scale
    right_weight = math.sin(fraction * half_angle) / scale
    return _normalized_quaternion(
        tuple(left_weight * left + right_weight * right
              for left, right in zip(current, target)),
        'controller')


@dataclass(frozen=True)
class PreviewConfig:
    scale: float = 0.2
    axes: tuple[str, str, str] = ('z', '-x', 'y')
    radius: float = 0.03
    speed: float = 0.01
    timeout: float = 0.25
    jump_limit: float = 0.10
    orientation_radius: float = math.radians(45)
    orientation_jump_limit: float = math.radians(20)
    orientation_speed: float | None = None
    lock_position: bool = False
    require_verified_tracking: bool = False

    def __post_init__(self):
        for name in ('scale', 'radius', 'speed', 'timeout', 'jump_limit',
                     'orientation_radius', 'orientation_jump_limit'):
            value = getattr(self, name)
            if not finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if (self.orientation_speed is not None and
                (not finite_number(self.orientation_speed) or
                 self.orientation_speed <= 0)):
            raise ValueError('orientation_speed must be finite and positive')
        for name in ('lock_position', 'require_verified_tracking'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if (len(self.axes) != 3 or
                any(axis not in ('x', 'y', 'z', '-x', '-y', '-z') for axis in self.axes) or
                len({axis.lstrip('-') for axis in self.axes}) != 3):
            raise ValueError('axes must be a signed permutation of x,y,z, e.g. z,-x,y')


@dataclass(frozen=True)
class ControllerPacket:
    session: str
    sequence: int
    sent_time: float
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    grip: float
    valid: bool
    tracking_verified: bool

    @classmethod
    def parse(cls, payload: bytes) -> 'ControllerPacket':
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Invalid UTF-8 JSON packet') from exc
        if not isinstance(data, dict):
            raise ValueError('Packet must be a JSON object')
        if any(key not in data for key in ('session', 'sequence', 'sent_time')):
            raise ValueError('Missing metadata: rebuild/install the updated PICO APK')
        if not isinstance(data['session'], str) or not data['session'].strip():
            raise ValueError('session must be a nonempty string')
        if type(data['sequence']) is not int or data['sequence'] < 0:
            raise ValueError('sequence must be a nonnegative integer')
        fields = ('sent_time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'trigger', 'grip')
        if any(not finite_number(data.get(key)) for key in fields):
            raise ValueError('Packet fields must be finite numbers, excluding booleans')
        if data['sent_time'] < 0 or not all(0 <= data[key] <= 1 for key in ('trigger', 'grip')):
            raise ValueError('Invalid sender time or button range')
        if type(data.get('valid')) is not bool:
            raise ValueError('valid must be boolean')
        quaternion = tuple(data[key] for key in ('qx', 'qy', 'qz', 'qw'))
        quaternion_norm = math.hypot(*quaternion)
        if data['valid'] and quaternion_norm < 1e-6:
            raise ValueError('Valid tracking requires a nonzero quaternion')
        rotation = ((0.0, 0.0, 0.0, 1.0) if quaternion_norm < 1e-6 else
                    tuple(value / quaternion_norm for value in quaternion))
        diagnostic_fields = ('device_valid', 'position_valid', 'rotation_valid',
                             'grip_valid', 'tracked_valid', 'tracked')
        tracking_verified = (data['valid'] and
                             all(data.get(key) is True
                                 for key in diagnostic_fields))
        return cls(data['session'], data['sequence'], data['sent_time'],
                   (data['x'], data['y'], data['z']), rotation,
                   data['grip'], data['valid'], tracking_verified)


@dataclass(frozen=True)
class PreviewCommandState:
    target: tuple[float, ...]
    limited_rotation: tuple[float, ...] | None


class PreviewEngine:
    def __init__(self, config: PreviewConfig, read_pose: Callable[[], list[float]],
                 map_orientation: Callable | None = None):
        self.config = config
        self.read_pose = read_pose
        self.map_orientation = map_orientation
        self.status = 'WAIT_RELEASE'
        self.reason = 'Waiting for fresh packets and grip release'
        self._source = None
        self._retired_sources = set()
        self._retired_sessions = set()
        self._sequence = -1
        self._sent_time = -1.0
        self._received = None
        self._previous_position = None
        self._filtered_position = None
        self._previous_rotation = None
        self._reference_position = None
        self._reference_rotation = None
        self._limited_rotation = None
        self._reference_pose = None
        self._target = None
        self._target_time = None

    @property
    def source_peer(self) -> tuple[str, int] | None:
        return None if self._source is None else self._source[0]

    @property
    def engaged(self) -> bool:
        return self.status in ('ACTIVE', 'HOLD')

    def command_state(self) -> PreviewCommandState:
        if not self.engaged or self._target is None:
            raise RuntimeError('No engaged command state is available')
        return PreviewCommandState(
            tuple(self._target),
            None if self._limited_rotation is None else
            tuple(self._limited_rotation))

    def hold(self, reason: str, now: float | None = None,
             restore: PreviewCommandState | None = None) -> None:
        if not self.engaged:
            raise RuntimeError('Cannot hold a disarmed preview engine')
        if restore is not None:
            if not isinstance(restore, PreviewCommandState):
                raise TypeError('restore must be a PreviewCommandState')
            self._target = list(restore.target)
            self._limited_rotation = restore.limited_rotation
        self.status = 'HOLD'
        self.reason = str(reason)
        if now is not None:
            if not finite_number(now):
                raise ValueError('Hold time must be finite')
            self._target_time = float(now)

    def disarm(self, reason: str) -> None:
        self.status = 'WAIT_RELEASE'
        self.reason = reason
        self._reference_position = self._reference_pose = self._target = None
        self._reference_rotation = None
        self._limited_rotation = None
        self._target_time = None
        self._previous_position = None
        self._filtered_position = None
        self._previous_rotation = None

    def tick(self, now: float) -> None:
        if self._received is not None and now - self._received >= self.config.timeout:
            self.disarm('Packet timeout; release grip before retrying')
            self._received = None

    def accept(self, packet: ControllerPacket, now: float,
               peer: tuple[str, int]) -> list[float] | None:
        self.tick(now)
        source = (peer, packet.session)
        if source in self._retired_sources or packet.session in self._retired_sessions:
            return None  # Old sessions/addresses cannot re-arm after replacement.
        if source != self._source:
            if self._source is not None:
                self._retired_sources.add(self._source)
                if self._source[1] != packet.session:
                    self._retired_sessions.add(self._source[1])
            self.disarm('New sender/session; release grip before retrying')
            self._source = source
            self._sequence = packet.sequence
            self._sent_time = packet.sent_time
            self._received = now
            return None
        if packet.sequence <= self._sequence or packet.sent_time <= self._sent_time:
            return None  # Replays do not refresh the watchdog.
        sender_gap = packet.sent_time - self._sent_time
        self._sequence, self._sent_time = packet.sequence, packet.sent_time
        self._received = now
        if sender_gap >= self.config.timeout:
            self.disarm('Sender time gap; release grip before retrying')
        if not packet.valid:
            self.disarm('Tracking invalid; release grip before retrying')
            return None
        if self.config.require_verified_tracking and not packet.tracking_verified:
            self.disarm('Verified tracking diagnostics required; release grip before retrying')
            return None
        if packet.grip <= .3:
            self.disarm('Grip released; ready')
            self.status = 'READY'
            return None
        if self.status == 'WAIT_RELEASE':
            return None
        if self.status == 'READY':
            if packet.grip < .7:
                return None
            # Clear readiness before the blocking read, including when it fails.
            self.disarm('Reading robot reference')
            self._reference_pose = validate_pose(self.read_pose())
            self._reference_position = packet.position
            self._reference_rotation = packet.rotation
            self._limited_rotation = packet.rotation
            self._previous_position = packet.position
            self._filtered_position = packet.position
            self._previous_rotation = packet.rotation
            self._target = self._reference_pose.copy()
            self._target_time = now
            self.status, self.reason = 'ACTIVE', 'Preview active; robot does not move'
            return self._target.copy()
        if math.dist(packet.position, self._previous_position) > self.config.jump_limit:
            self.disarm('Controller position jump; release grip before retrying')
            return None
        self._previous_position = packet.position

        alpha = 0.25
        self._filtered_position = tuple(
            alpha * raw + (1.0 - alpha) * old
            for raw, old in zip(packet.position, self._filtered_position)
        )

        if self.map_orientation is not None:
            if (_quaternion_angle(packet.rotation, self._previous_rotation) >
                    self.config.orientation_jump_limit):
                self.disarm('Controller rotation jump; release grip before retrying')
                return None
            if (_quaternion_angle(packet.rotation, self._reference_rotation) >

                    self.config.orientation_radius):

                self._previous_rotation = packet.rotation
                self.hold('Target outside orientation radius; reverse controller to resume',
                          now)
                return None
            self._previous_rotation = packet.rotation
        delta = dict(zip(('x', 'y', 'z'),
                         (a - b for a, b in zip(
                             self._filtered_position,
                             self._reference_position))))
        offset = [delta[axis.lstrip('-')] * (-1 if axis.startswith('-') else 1) * self.config.scale
                  for axis in self.config.axes]
        if math.hypot(*offset) > self.config.radius:
            self.hold('Target outside reference radius; reverse controller to resume', now)
            return None
        requested = (list(self._reference_pose[:3]) if self.config.lock_position else
                     [a + b for a, b in zip(self._reference_pose[:3], offset)])
        distance = math.dist(requested, self._target[:3])
        step = self.config.speed * max(0, now - self._target_time)
        fraction = min(1.0, step / distance) if distance else 1.0
        self._target[:3] = [old + fraction * (new - old)
                            for old, new in zip(self._target[:3], requested)]
        if self.map_orientation is not None:
            mapped_rotation = packet.rotation
            if self.config.orientation_speed is not None:
                mapped_rotation = _step_quaternion(
                    self._limited_rotation, packet.rotation,
                    self.config.orientation_speed * max(0, now - self._target_time))
            self._limited_rotation = mapped_rotation
            self._target[3:] = self.map_orientation(
                self._reference_rotation, mapped_rotation,
                self._reference_pose[3:])
        self._target_time = now
        self.status = 'ACTIVE'
        self.reason = 'Preview active; robot does not move'
        return self._target.copy()
