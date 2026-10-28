"""Non-critical UDP output for RM75 measured joint visualization."""
import json
import math
from typing import Callable


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def encode_state_packet(joints: object, sequence: int,
                        sent_time: float) -> bytes:
    if (not isinstance(joints, (list, tuple)) or len(joints) != 7 or
            any(not _finite_number(value) for value in joints)):
        raise ValueError('joint_deg must contain seven finite numbers')
    if type(sequence) is not int or sequence < 0:
        raise ValueError('sequence must be a nonnegative integer')
    if not _finite_number(sent_time) or sent_time < 0:
        raise ValueError('sent_time must be a finite nonnegative number')
    payload = {
        'type': 'rm75_state',
        'version': 1,
        'sequence': sequence,
        'sent_time': float(sent_time),
        'connected': True,
        'joint_deg': [float(value) for value in joints],
    }
    return json.dumps(payload, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


class StateBroadcaster:
    def __init__(self, udp: object, clock: Callable[[], float],
                 port: int = 5006, warning_interval: float = 1.0,
                 warn: Callable[[str], None] = print):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('State port must be between 1 and 65535')
        if (not _finite_number(warning_interval) or
                warning_interval <= 0):
            raise ValueError('warning_interval must be finite and positive')
        self._udp = udp
        self._clock = clock
        self._port = port
        self._warning_interval = float(warning_interval)
        self._warn = warn
        self._sequence = 0
        self._last_warning_time = None
        self.last_warning = None

    def publish(self, state: object, pico_ip: str | None) -> None:
        if not isinstance(state, dict):
            raise ValueError('Robot state must be a dictionary')
        now = self._clock()
        payload = encode_state_packet(state.get('joint'), self._sequence, now)
        self._sequence += 1
        destinations = [('127.0.0.1', self._port)]
        if isinstance(pico_ip, str) and pico_ip and pico_ip != '127.0.0.1':
            destinations.append((pico_ip, self._port))
        for destination in destinations:
            try:
                self._udp.sendto(payload, destination)
            except OSError as exc:
                self.last_warning = (
                    f'State UDP {destination[0]}:{destination[1]} failed: {exc}')
                if (self._last_warning_time is None or
                        now - self._last_warning_time >= self._warning_interval):
                    self._warn(self.last_warning)
                    self._last_warning_time = now

    def close(self) -> None:
        self._udp.close()
