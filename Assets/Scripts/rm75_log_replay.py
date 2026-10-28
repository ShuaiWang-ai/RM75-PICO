"""Deterministic JSONL recording and robot-free replay for RM75 control frames."""
import argparse
import base64
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Callable, TextIO

from rm75_preview_core import (ControllerPacket, PreviewConfig, PreviewEngine,
                               map_relative_orientation, validate_pose)


LOG_TYPE = 'rm75_control_log'
LOG_VERSION = 1


@dataclass(frozen=True)
class ReplayFrame:
    received: float
    status: str
    reason: str
    target: list[float] | None


@dataclass(frozen=True)
class ReplayResult:
    config: PreviewConfig
    frames: list[ReplayFrame]


class ControlLogWriter:
    """Write complete control frames; logging failures are warned and disabled."""

    def __init__(self, destination: str | Path | TextIO, config: PreviewConfig,
                 warn: Callable[[str], None] | None = None,
                 orientation_axis: str | None = None):
        if orientation_axis not in (None, 'x', 'y', 'z'):
            raise ValueError('orientation_axis must be x, y, z, or None')
        self._warn = warn or (lambda message: print(message, file=sys.stderr))
        self._owns_stream = not hasattr(destination, 'write')
        self._stream = (open(destination, 'x', encoding='utf-8', buffering=1)
                        if self._owns_stream else destination)
        self._enabled = True
        self._write({'type': LOG_TYPE, 'version': LOG_VERSION,
                     'record': 'header', 'config': asdict(config),
                     'orientation_axis': orientation_axis})

    def _write(self, value: dict) -> None:
        if not self._enabled:
            return
        try:
            self._stream.write(json.dumps(
                value, separators=(',', ':'), allow_nan=False) + '\n')
            self._stream.flush()
        except Exception as exc:
            self._enabled = False
            try:
                self._warn(f'CONTROL LOG disabled after write failure: {exc}')
            except Exception:
                pass

    def record_frame(self, payload: bytes | None,
                     peer: tuple[str, int] | None, received: float, *,
                     completed: float | None = None,
                     reference_pose: list[float] | None = None,
                     status: str | None = None, reason: str | None = None,
                     target: list[float] | None = None,
                     motion_outcome: str | None = None) -> None:
        record = {'type': LOG_TYPE, 'version': LOG_VERSION, 'record': 'frame',
                  'received': received,
                  'completed': received if completed is None else completed,
                  'payload_b64': (None if payload is None else
                                  base64.b64encode(payload).decode('ascii')),
                  'peer': None if peer is None else [peer[0], peer[1]],
                  'reference_pose': reference_pose,
                  'status': status, 'reason': reason, 'target': target,
                  'motion_outcome': motion_outcome}
        self._write(record)

    def close(self) -> None:
        if self._owns_stream and not self._stream.closed:
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def _euler_to_quaternion(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def _quaternion_to_euler(quaternion: list[float]) -> list[float]:
    w, x, y, z = quaternion
    return [
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ]


def offline_orientation_mapper(axes: tuple[str, str, str],
                               project_axis: str | None = None):
    return lambda reference, current, robot_rpy: map_relative_orientation(
        reference, current, robot_rpy, axes,
        _euler_to_quaternion, _quaternion_to_euler,
        project_axis=project_axis)


def _read_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f'Invalid control log JSON at line {line_number}') from exc
            if (not isinstance(value, dict) or value.get('type') != LOG_TYPE or
                    value.get('version') != LOG_VERSION):
                raise ValueError(f'Invalid control log record at line {line_number}')
            records.append(value)
    if not records:
        raise ValueError('Control log is empty')
    return records


def _validated_frame(record: dict, line: int,
                     previous_received: float | None) -> tuple[float, float, bytes | None]:
    required = {'received', 'completed', 'payload_b64', 'peer',
                'reference_pose', 'status', 'reason', 'target',
                'motion_outcome'}
    missing = required - record.keys()
    if missing:
        raise ValueError(f'Control log frame line {line} missing {sorted(missing)}')
    received = record['received']
    if (type(received) not in (int, float) or not math.isfinite(received) or
            received < 0 or
            (previous_received is not None and received < previous_received)):
        raise ValueError(f'Invalid received time at line {line}')
    completed = record['completed']
    if (type(completed) not in (int, float) or not math.isfinite(completed) or
            completed < received):
        raise ValueError(f'Invalid completed time at line {line}')
    if record['status'] not in ('WAIT_RELEASE', 'READY', 'ACTIVE', 'HOLD'):
        raise ValueError(f'Invalid status at line {line}')
    if not isinstance(record['reason'], str) or not record['reason']:
        raise ValueError(f'Invalid reason at line {line}')
    if record['motion_outcome'] not in (None, 'SENT', 'UNCHANGED', 'HOLD', 'DISARM'):
        raise ValueError(f'Invalid motion_outcome at line {line}')
    if record['reference_pose'] is not None:
        validate_pose(record['reference_pose'])
    if record['target'] is not None:
        validate_pose(record['target'])
    encoded = record['payload_b64']
    if encoded is None:
        if record['peer'] is not None:
            raise ValueError(f'Timeout frame has peer at line {line}')
        payload = None
    else:
        if not isinstance(encoded, str):
            raise ValueError(f'Invalid payload encoding at line {line}')
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid payload encoding at line {line}') from exc
        peer = record['peer']
        if (not isinstance(peer, list) or len(peer) != 2 or
                not isinstance(peer[0], str) or not peer[0] or
                type(peer[1]) is not int or not 1 <= peer[1] <= 65535):
            raise ValueError(f'Invalid peer at line {line}')
    return float(received), float(completed), payload


def replay_log(path: str | Path, config: PreviewConfig | None = None, *,
               orientation: bool | None = None) -> ReplayResult:
    """Replay frames through the real parser/preview engine without robot I/O."""
    records = _read_records(path)
    header = records[0]
    if header.get('record') != 'header' or not isinstance(header.get('config'), dict):
        raise ValueError('Control log must start with a header')
    if config is None:
        values = dict(header['config'])
        values['axes'] = tuple(values['axes'])
        config = PreviewConfig(**values)

    current_reference = [None]

    def read_pose():
        if current_reference[0] is None:
            raise ValueError('Replay activation frame is missing reference_pose')
        return validate_pose(current_reference[0])

    if orientation is None:
        orientation = config.orientation_speed is not None
    orientation_axis = header.get('orientation_axis')
    if orientation_axis not in (None, 'x', 'y', 'z'):
        raise ValueError('Control log has invalid orientation_axis')
    mapper = (offline_orientation_mapper(config.axes, orientation_axis)
              if orientation else None)
    engine = PreviewEngine(config, read_pose, map_orientation=mapper)
    replayed = []
    previous_received = None
    for index, record in enumerate(records[1:], start=2):
        if record.get('record') != 'frame':
            raise ValueError(f'Invalid control log frame at line {index}')
        received, completed, payload = _validated_frame(
            record, index, previous_received)
        previous_received = received
        current_reference[0] = record.get('reference_pose')
        checkpoint = engine.command_state() if engine.engaged else None
        target = None
        if payload is None:
            engine.tick(received)
        else:
            try:
                peer_value = record.get('peer')
                packet = ControllerPacket.parse(payload)
                target = engine.accept(packet, received,
                                       (peer_value[0], peer_value[1]))
            except ValueError as exc:
                engine.disarm(f'Invalid packet: {exc}')
        engine.tick(completed)
        if completed - received >= config.timeout:
            engine.disarm('Reference read timeout; release grip before retrying')
            target = None
        if record.get('motion_outcome') == 'HOLD' and engine.engaged:
            engine.hold(record.get('reason', 'Recorded recoverable limit'),
                        float(received), restore=checkpoint)
            target = None
        elif record.get('motion_outcome') == 'DISARM':
            engine.disarm(record.get('reason', 'Recorded live control fault'))
            target = None
        expected = record.get('status')
        if expected is not None and expected != engine.status:
            raise ValueError(
                f'Replay line {index} expected status {expected}, got {engine.status}')
        expected_target = record.get('target')
        if expected_target is not None:
            expected_target = validate_pose(expected_target)
            if (target is None or any(not math.isclose(actual, wanted,
                                                       abs_tol=1e-12)
                                      for actual, wanted in zip(
                                          target, expected_target))):
                raise ValueError(f'Replay line {index} target does not match log')
        replayed.append(ReplayFrame(float(received), engine.status,
                                    engine.reason,
                                    None if target is None else list(target)))
    return ReplayResult(config, replayed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Replay an RM75 JSONL control log offline; never connects to a robot.')
    parser.add_argument('log', help='JSONL control log path')
    parser.add_argument('--orientation', action='store_true',
                        help='Replay relative orientation mapping with offline math')
    args = parser.parse_args(argv)
    try:
        result = replay_log(args.log,
                            orientation=True if args.orientation else None)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    for index, frame in enumerate(result.frames, start=1):
        print(f'{index}: t={frame.received:.6f} {frame.status}: {frame.reason}')
    print(f'Replayed {len(result.frames)} frames; axes={result.config.axes}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
