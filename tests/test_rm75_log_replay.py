import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Assets' / 'Scripts'))

from rm75_log_replay import ControlLogWriter, replay_log
from rm75_preview_core import PreviewConfig
from test_rm75_preview_core import PEER, POSE, packet


def raw_packet(seq, now, grip=0.0, **changes):
    value = packet(seq, now, grip, **changes)
    data = dict(session=value.session, sequence=value.sequence,
                sent_time=value.sent_time,
                x=value.position[0], y=value.position[1], z=value.position[2],
                qx=value.rotation[0], qy=value.rotation[1],
                qz=value.rotation[2], qw=value.rotation[3],
                trigger=0.0, grip=value.grip, valid=value.valid)
    return json.dumps(data, separators=(',', ':')).encode()


def test_control_log_replays_active_hold_reverse_disarm_and_timeout(tmp_path):
    path = tmp_path / 'control.jsonl'
    config = PreviewConfig(speed=1.0)
    with ControlLogWriter(path, config) as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='WAIT_RELEASE', reason='new source')
        log.record_frame(raw_packet(1, .01), PEER, .01,
                         status='READY', reason='released')
        log.record_frame(raw_packet(2, .02, 1), PEER, .02,
                         reference_pose=POSE, status='ACTIVE', reason='active',
                         target=POSE, motion_outcome='SENT')
        log.record_frame(raw_packet(3, .03, 1, z=.02), PEER, .03,
                         status='HOLD', reason='IK joint step limit',
                         motion_outcome='HOLD')
        log.record_frame(raw_packet(4, .04, 1, z=0), PEER, .04,
                         status='ACTIVE', reason='active',
                         motion_outcome='SENT')
        log.record_frame(raw_packet(5, .05, 1, valid=False), PEER, .05,
                         status='WAIT_RELEASE', reason='tracking invalid')
        log.record_frame(None, None, .31, status='WAIT_RELEASE',
                         reason='packet timeout')

    result = replay_log(path)

    assert [frame.status for frame in result.frames] == [
        'WAIT_RELEASE', 'READY', 'ACTIVE', 'HOLD', 'ACTIVE',
        'WAIT_RELEASE', 'WAIT_RELEASE',
    ]
    assert result.frames[3].target is None
    assert result.frames[4].target is not None
    assert 'Tracking invalid' in result.frames[5].reason
    assert 'Packet timeout' in result.frames[6].reason
    assert result.config.axes == ('z', '-x', 'y')


def test_replay_handles_orientation_with_builtin_offline_math(tmp_path):
    path = tmp_path / 'orientation.jsonl'
    config = PreviewConfig(speed=1.0, orientation_speed=1.0)
    angle = math.radians(1.0)
    with ControlLogWriter(path, config) as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='WAIT_RELEASE', reason='new source')
        log.record_frame(raw_packet(1, .01), PEER, .01,
                         status='READY', reason='released')
        log.record_frame(raw_packet(2, .02, 1), PEER, .02,
                         reference_pose=POSE, status='ACTIVE', reason='active')
        log.record_frame(raw_packet(3, .03, 1, qz=math.sin(angle / 2),
                                    qw=math.cos(angle / 2)), PEER, .03,
                         status='ACTIVE', reason='active')

    result = replay_log(path, orientation=True)

    assert result.frames[-1].target is not None
    assert result.frames[-1].target[3:] != POSE[3:]


def test_replay_applies_recorded_reference_read_completion_timeout(tmp_path):
    path = tmp_path / 'slow-reference.jsonl'
    with ControlLogWriter(path, PreviewConfig()) as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='WAIT_RELEASE', reason='new source')
        log.record_frame(raw_packet(1, .01), PEER, .01,
                         status='READY', reason='released')
        log.record_frame(raw_packet(2, .02, 1), PEER, .02,
                         completed=.32, reference_pose=POSE,
                         status='WAIT_RELEASE', reason='reference timeout')

    result = replay_log(path)

    assert result.frames[-1].status == 'WAIT_RELEASE'
    assert 'Reference read timeout' in result.frames[-1].reason


def test_replay_uses_recorded_single_orientation_projection_axis(tmp_path):
    path = tmp_path / 'axis-x.jsonl'
    config = PreviewConfig(speed=1.0, orientation_speed=1.0)
    angle = math.radians(1.0)
    with ControlLogWriter(path, config, orientation_axis='x') as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='WAIT_RELEASE', reason='new source')
        log.record_frame(raw_packet(1, .01), PEER, .01,
                         status='READY', reason='released')
        log.record_frame(raw_packet(2, .02, 1), PEER, .02,
                         reference_pose=POSE, status='ACTIVE', reason='active')
        log.record_frame(raw_packet(3, .03, 1, qx=math.sin(angle / 2),
                                    qw=math.cos(angle / 2)), PEER, .03,
                         status='ACTIVE', reason='active', target=POSE)

    result = replay_log(path)

    assert result.frames[-1].target == pytest.approx(POSE)


def test_replay_rejects_malformed_or_inconsistent_log_without_io_side_effects(tmp_path):
    malformed = tmp_path / 'bad.jsonl'
    malformed.write_text('{bad\n', encoding='utf-8')
    with pytest.raises(ValueError, match='line 1'):
        replay_log(malformed)

    inconsistent = tmp_path / 'mismatch.jsonl'
    with ControlLogWriter(inconsistent, PreviewConfig()) as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='ACTIVE', reason='wrong expected state')
    with pytest.raises(ValueError, match='expected status ACTIVE'):
        replay_log(inconsistent)


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('payload_b64', '%%%', 'payload encoding'),
    ('motion_outcome', 'FLY', 'motion_outcome'),
    ('received', -1.0, 'received time'),
])
def test_replay_rejects_corrupt_frame_envelope(tmp_path, field, value, message):
    path = tmp_path / f'bad-{field}.jsonl'
    with ControlLogWriter(path, PreviewConfig()) as log:
        log.record_frame(raw_packet(0, 0), PEER, 0.0,
                         status='WAIT_RELEASE', reason='new source')
    lines = path.read_text(encoding='utf-8').splitlines()
    frame = json.loads(lines[1])
    frame[field] = value
    lines[1] = json.dumps(frame)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        replay_log(path)


def test_log_write_failure_is_nonfatal_and_disables_further_writes():
    class FailingStream:
        def __init__(self):
            self.calls = 0

        def write(self, unused):
            self.calls += 1
            raise OSError('disk full')

        def flush(self):
            raise AssertionError('flush must not follow failed write')

    stream = FailingStream()
    warnings = []
    writer = ControlLogWriter(stream, PreviewConfig(), warn=warnings.append)

    writer.record_frame(raw_packet(0, 0), PEER, 0.0)

    assert stream.calls == 1
    assert len(warnings) == 1
    assert 'disk full' in warnings[0]


def test_log_write_and_warning_failures_do_not_escape():
    class BrokenStream:
        def write(self, unused):
            raise OSError('disk full')

    def broken_warning(unused):
        raise OSError('stderr closed')

    writer = ControlLogWriter(BrokenStream(), PreviewConfig(), warn=broken_warning)
    writer.record_frame(raw_packet(0, 0), PEER, 0.0)
