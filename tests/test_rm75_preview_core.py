import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Assets' / 'Scripts'))
import rm75_preview_core as preview_core
from rm75_preview_core import ControllerPacket, PreviewConfig, PreviewEngine

PEER = ('192.168.30.2', 40000)
POSE = [0.1, 0.2, 0.3, 0.0, -0.072, -0.021]


def packet(seq, now, grip=0.0, **changes):
    data = dict(session='session-a', sequence=seq, sent_time=now,
                x=0.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                trigger=0.0, grip=grip, valid=True)
    data.update(changes)
    return ControllerPacket.parse(json.dumps(data).encode())


def euler_to_quaternion(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.reads = 0
        self.pose = POSE.copy()
        def read_pose():
            self.reads += 1
            return self.pose.copy()
        self.engine = PreviewEngine(PreviewConfig(), read_pose)

    def ready(self):
        self.assertIsNone(self.engine.accept(packet(0, 0), 0, PEER))
        self.engine.accept(packet(1, 0.01), 0.01, PEER)
        self.assertEqual(self.engine.status, 'READY')

    def active(self):
        self.ready()
        return self.engine.accept(packet(2, 0.02, 1), 0.02, PEER)

    def test_default_mapping(self):
        self.engine = PreviewEngine(PreviewConfig(speed=1), lambda: POSE.copy())
        self.active()
        target = self.engine.accept(packet(3, 0.12, 1, x=.01, y=.02, z=.03), .12, PEER)
        for actual, wanted in zip(target, [.1015, .1995, .301, 0, -.072, -.021]):
            self.assertAlmostEqual(actual, wanted)

    def test_press_has_zero_jump(self):
        self.assertEqual(self.active(), POSE)
        self.assertEqual(self.reads, 1)

    def test_first_packet_never_arms(self):
        self.engine.accept(packet(0, 0, 1), 0, PEER)
        self.engine.accept(packet(1, .01, 1), .01, PEER)
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')
        self.assertEqual(self.reads, 0)

    def test_repress_reads_new_pose(self):
        self.active()
        self.engine.accept(packet(3, .03), .03, PEER)
        self.pose[0] = .5
        self.assertEqual(self.engine.accept(packet(4, .04, 1), .04, PEER)[0], .5)
        self.assertEqual(self.reads, 2)

    def test_grip_hysteresis(self):
        self.ready()
        self.assertIsNone(self.engine.accept(packet(2, .02, .69), .02, PEER))
        self.engine.accept(packet(3, .03, .7), .03, PEER)
        self.engine.accept(packet(4, .04, .31), .04, PEER)
        self.assertEqual(self.engine.status, 'ACTIVE')
        self.assertIsNone(self.engine.accept(packet(5, .05, .3), .05, PEER))
        self.assertEqual(self.engine.status, 'READY')

    def test_timeout_requires_release(self):
        self.active()
        self.engine.tick(.28)
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')
        self.assertIsNone(self.engine.accept(packet(3, .29, 1), .29, PEER))
        self.assertEqual(self.reads, 1)

    def test_late_packet_expires_before_processing(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(3, .28, 1), .28, PEER))
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_invalid_tracking_disarms(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(3, .03, 1, valid=False), .03, PEER))
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_position_radius_holds_command_state_and_reverse_exits(self):
        self.active()
        reference_pose = self.engine._reference_pose.copy()
        for sequence, z in enumerate((.08, .16, .24), start=3):
            self.assertIsNotNone(self.engine.accept(
                packet(sequence, sequence / 100, 1, z=z),
                sequence / 100, PEER))
        accepted_target = self.engine._target.copy()
        self.assertIsNone(self.engine.accept(packet(6, .06, 1, z=.32), .06, PEER))
        self.assertEqual(self.engine.status, 'HOLD')
        self.assertTrue(self.engine.engaged)
        self.assertEqual(self.engine._reference_pose, reference_pose)
        self.assertEqual(self.engine._target, accepted_target)
        self.assertEqual(self.engine._target_time, .06)

        recovered = None
        for sequence, z in enumerate((.24, .16, .08, 0.0), start=7):
            recovered = self.engine.accept(
                packet(sequence, sequence / 100, 1, z=z),
                sequence / 100, PEER)
            if recovered is not None:
                break
        self.assertIsNotNone(recovered)
        self.assertEqual(self.engine.status, 'ACTIVE')

    def test_controller_jump(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(3, .03, 1, x=.101), .03, PEER))
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_speed_limit(self):
        self.active()
        target = self.engine.accept(packet(3, .12, 1, z=.03), .12, PEER)
        self.assertAlmostEqual(target[0], .101)
        self.assertEqual(target[3:], POSE[3:])

    def test_position_lock_holds_reference_xyz(self):
        engine = PreviewEngine(
            PreviewConfig(speed=1, lock_position=True),
            lambda: POSE.copy(),
        )
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)

        target = engine.accept(packet(3, .12, 1, z=.05), .12, PEER)

        self.assertEqual(target[:3], POSE[:3])

    def test_source_restart_disarms(self):
        for changes, peer in [({'session': 'new'}, PEER), ({}, ('192.168.30.3', 40000))]:
            with self.subTest(changes=changes, peer=peer):
                self.setUp()
                self.active()
                self.assertIsNone(self.engine.accept(packet(0, .03, 1, **changes), .03, peer))
                self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_source_peer_tracks_current_source_and_ignores_retired_source(self):
        self.assertIsNone(self.engine.source_peer)
        self.engine.accept(packet(0, 0), 0, PEER)
        self.assertEqual(self.engine.source_peer, PEER)
        replacement = ('192.168.30.3', 40001)
        self.engine.accept(packet(0, .01, session='session-b'), .01, replacement)
        self.assertEqual(self.engine.source_peer, replacement)
        self.engine.accept(packet(1, .02), .02, PEER)
        self.assertEqual(self.engine.source_peer, replacement)

    def test_reordered_packet_does_not_refresh_timeout(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(1, .01, 1), .2, PEER))
        self.engine.tick(.28)
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_duplicate_packet_ignored(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(2, .02, 1), .03, PEER))
        self.assertEqual(self.reads, 1)

    def test_backwards_sender_time_ignored(self):
        self.active()
        self.assertIsNone(self.engine.accept(packet(3, .01, 1), .03, PEER))

    def test_malformed_numeric_fields(self):
        for changes in [dict(x=float('nan')), dict(qw=float('inf')), dict(x=True),
                        dict(sequence=True), dict(sequence=-1), dict(sequence=1.5),
                        dict(grip=1.1), dict(valid=1), dict(qw=0), dict(session=''),
                        dict(sent_time=-1), dict(trigger=-.1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                packet(0, 0, **changes)

    def test_packet_preserves_normalized_controller_rotation(self):
        parsed = packet(0, 0, qx=0.0, qy=0.0, qz=2.0, qw=2.0)
        expected = (0.0, 0.0, 2 ** -0.5, 2 ** -0.5)
        rotation = getattr(parsed, 'rotation', None)
        self.assertIsNotNone(rotation)
        for actual, wanted in zip(rotation, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_only_complete_tracking_diagnostics_are_verified(self):
        ordinary = packet(0, 0)
        diagnostics = dict(device_valid=True, position_valid=True,
                           rotation_valid=True, grip_valid=True,
                           tracked_valid=True, tracked=True)
        verified = packet(1, .01, **diagnostics)
        lost = packet(2, .02, **dict(diagnostics, tracked=False))

        self.assertFalse(ordinary.tracking_verified)
        self.assertTrue(verified.tracking_verified)
        self.assertFalse(lost.tracking_verified)

    def test_verified_tracking_mode_cannot_arm_from_bridge_packet(self):
        diagnostics = dict(device_valid=True, position_valid=True,
                           rotation_valid=True, grip_valid=True,
                           tracked_valid=True, tracked=True)
        engine = PreviewEngine(
            PreviewConfig(require_verified_tracking=True),
            lambda: POSE.copy(),
        )
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        self.assertEqual(engine.status, 'WAIT_RELEASE')
        self.assertIn('verified tracking', engine.reason.lower())

        engine.accept(packet(2, .02, **diagnostics), .02, PEER)
        self.assertEqual(engine.status, 'READY')
        target = engine.accept(packet(3, .03, 1, **diagnostics), .03, PEER)
        self.assertEqual(target, POSE)

    def test_relative_rotation_uses_the_position_axis_basis(self):
        map_orientation = getattr(preview_core, 'map_relative_orientation', None)
        self.assertIsNotNone(map_orientation)
        sine, cosine = .5, math.sqrt(.75)
        cases = [
            ((sine, 0.0, 0.0, cosine), (0.0, -math.pi / 3, 0.0)),
            ((0.0, sine, 0.0, cosine), (0.0, 0.0, math.pi / 3)),
            ((0.0, 0.0, sine, cosine), (math.pi / 3, 0.0, 0.0)),
        ]
        for current, expected in cases:
            with self.subTest(current=current):
                actual = map_orientation(
                    (0.0, 0.0, 0.0, 1.0), current,
                    [0.0, 0.0, 0.0], ('z', '-x', 'y'),
                    euler_to_quaternion, quaternion_to_euler)
                for value, wanted in zip(actual, expected):
                    self.assertAlmostEqual(value, wanted, places=6)

    def test_orientation_projection_keeps_only_selected_robot_axis(self):
        sine, cosine = .5, math.sqrt(.75)
        controller_x_rotation = (sine, 0.0, 0.0, cosine)

        projected_x = preview_core.map_relative_orientation(
            (0.0, 0.0, 0.0, 1.0), controller_x_rotation,
            [0.0, 0.0, 0.0], ('z', '-x', 'y'),
            euler_to_quaternion, quaternion_to_euler,
            project_axis='x')
        projected_y = preview_core.map_relative_orientation(
            (0.0, 0.0, 0.0, 1.0), controller_x_rotation,
            [0.0, 0.0, 0.0], ('z', '-x', 'y'),
            euler_to_quaternion, quaternion_to_euler,
            project_axis='y')

        for value in projected_x:
            self.assertAlmostEqual(value, 0.0, places=6)
        for value, wanted in zip(projected_y, (0.0, -math.pi / 3, 0.0)):
            self.assertAlmostEqual(value, wanted, places=6)

    def test_orientation_projection_rejects_unknown_axis(self):
        with self.assertRaises(ValueError):
            preview_core.map_relative_orientation(
                (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0),
                [0.0, 0.0, 0.0], ('z', '-x', 'y'),
                euler_to_quaternion, quaternion_to_euler,
                project_axis='roll')

    def test_orientation_preview_latches_then_maps_relative_rotation(self):
        calls = []
        def map_orientation(reference, current, robot_rpy):
            calls.append((reference, current, robot_rpy))
            return [0.4, 0.5, 0.6]
        try:
            engine = PreviewEngine(PreviewConfig(speed=1), lambda: POSE.copy(),
                                   map_orientation=map_orientation)
        except TypeError as exc:
            self.fail(f'orientation preview is unavailable: {exc}')
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        pressed = engine.accept(packet(2, .02, 1), .02, PEER)
        self.assertEqual(pressed, POSE)
        self.assertEqual(calls, [])
        rotated = (0.0, 0.0, math.sin(.05), math.cos(.05))
        target = engine.accept(packet(3, .03, 1, qz=rotated[2], qw=rotated[3]),
                               .03, PEER)
        self.assertEqual(target[3:], [0.4, 0.5, 0.6])
        self.assertEqual(calls, [((0.0, 0.0, 0.0, 1.0), rotated, POSE[3:])])

    def test_orientation_speed_limits_rotation_before_mapping(self):
        mapped_rotations = []
        def map_orientation(reference, current, robot_rpy):
            mapped_rotations.append(current)
            return list(robot_rpy)

        engine = PreviewEngine(
            PreviewConfig(speed=1, orientation_speed=0.5),
            lambda: POSE.copy(),
            map_orientation=map_orientation,
        )
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)
        requested_angle = .10
        target = engine.accept(
            packet(3, .12, 1, qz=math.sin(requested_angle / 2),
                   qw=math.cos(requested_angle / 2)),
            .12, PEER,
        )

        self.assertIsNotNone(target)
        self.assertEqual(len(mapped_rotations), 1)
        limited_angle = 2 * math.acos(abs(mapped_rotations[0][3]))
        self.assertAlmostEqual(limited_angle, .05, places=6)

    def test_orientation_packet_jump_disarms_before_mapping(self):
        calls = []
        try:
            config = PreviewConfig(
                orientation_jump_limit=math.radians(20),
                orientation_radius=math.radians(45))
        except TypeError as exc:
            self.fail(f'orientation safety limits are unavailable: {exc}')
        engine = PreviewEngine(config, lambda: POSE.copy(),
                               map_orientation=lambda *args: calls.append(args))
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)
        angle = math.radians(20.001) / 2
        result = engine.accept(packet(3, .03, 1, qx=math.sin(angle),
                                      qw=math.cos(angle)), .03, PEER)
        self.assertIsNone(result)
        self.assertEqual(engine.status, 'WAIT_RELEASE')
        self.assertIn('rotation jump', engine.reason.lower())
        self.assertEqual(calls, [])

    def test_orientation_radius_holds_without_advancing_command_state(self):
        config = PreviewConfig(
            orientation_jump_limit=math.radians(20),
            orientation_radius=math.radians(45))
        engine = PreviewEngine(config, lambda: POSE.copy(),
                               map_orientation=lambda *args: POSE[3:])
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)
        for sequence, degrees in ((3, 15), (4, 30)):
            angle = math.radians(degrees) / 2
            self.assertIsNotNone(engine.accept(
                packet(sequence, sequence / 100, 1,
                       qx=math.sin(angle), qw=math.cos(angle)),
                sequence / 100, PEER))
        accepted_target = engine._target.copy()
        accepted_limited_rotation = engine._limited_rotation
        reference_rotation = engine._reference_rotation
        angle = math.radians(45.001) / 2
        self.assertIsNone(engine.accept(
            packet(5, .05, 1, qx=math.sin(angle), qw=math.cos(angle)),
            .05, PEER))
        self.assertEqual(engine.status, 'HOLD')
        self.assertTrue(engine.engaged)
        self.assertIn('orientation radius', engine.reason.lower())
        self.assertEqual(engine._reference_rotation, reference_rotation)
        self.assertEqual(engine._limited_rotation, accepted_limited_rotation)
        self.assertEqual(engine._target, accepted_target)
        self.assertEqual(engine._target_time, .05)

        retreat = math.radians(30) / 2
        self.assertIsNotNone(engine.accept(
            packet(6, .06, 1, qx=math.sin(retreat), qw=math.cos(retreat)),
            .06, PEER))
        self.assertEqual(engine.status, 'ACTIVE')

    def test_grip_release_from_hold_reanchors_on_next_press(self):
        self.active()
        for sequence, z in enumerate((.08, .16, .24, .32), start=3):
            self.engine.accept(packet(sequence, sequence / 100, 1, z=z),
                               sequence / 100, PEER)
        self.assertEqual(self.engine.status, 'HOLD')

        self.engine.accept(packet(7, .07, 0, z=.32), .07, PEER)
        self.assertEqual(self.engine.status, 'READY')
        self.pose[0] = .5
        target = self.engine.accept(packet(8, .08, 1, z=.32), .08, PEER)
        self.assertEqual(target[0], .5)
        self.assertEqual(self.engine.status, 'ACTIVE')

    def test_motion_hold_restores_pre_candidate_target_and_limited_rotation(self):
        engine = PreviewEngine(
            PreviewConfig(speed=1, orientation_speed=1),
            lambda: POSE.copy(),
            map_orientation=lambda unused_reference, current, unused_rpy:
                [current[0], current[1], current[2]],
        )
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)
        checkpoint = engine.command_state()
        angle = math.radians(1) / 2

        candidate = engine.accept(
            packet(3, .12, 1, z=.02, qz=math.sin(angle), qw=math.cos(angle)),
            .12, PEER)
        self.assertNotEqual(candidate, list(checkpoint.target))
        engine.hold('IK limit', .12, restore=checkpoint)

        self.assertEqual(engine._target, list(checkpoint.target))
        self.assertEqual(engine._limited_rotation, checkpoint.limited_rotation)
        self.assertEqual(engine._target_time, .12)
        self.assertEqual(engine.status, 'HOLD')

    def test_opposite_quaternion_sign_is_not_a_rotation_jump(self):
        engine = PreviewEngine(PreviewConfig(), lambda: POSE.copy(),
                               map_orientation=lambda *args: POSE[3:])
        engine.accept(packet(0, 0), 0, PEER)
        engine.accept(packet(1, .01), .01, PEER)
        engine.accept(packet(2, .02, 1), .02, PEER)
        target = engine.accept(packet(3, .03, 1, qw=-1.0), .03, PEER)
        self.assertIsNotNone(target)
        self.assertEqual(engine.status, 'ACTIVE')

    def test_legacy_packet_message(self):
        with self.assertRaisesRegex(ValueError, 'APK|metadata'):
            ControllerPacket.parse(b'{"x":0,"valid":true}')
        for payload in [b'null', b'[]', b'bad', b'\xff']:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ControllerPacket.parse(payload)

    def test_invalid_config(self):
        for changes in [dict(scale=0), dict(radius=-1), dict(speed=float('nan')),
                        dict(timeout=float('inf')), dict(axes=('x', 'x', 'z')),
                        dict(axes=('x', 'y', 'foo')), dict(jump_limit=0),
                        dict(lock_position=1),
                        dict(require_verified_tracking=1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PreviewConfig(**changes)

    def test_invalid_orientation_limits_are_rejected(self):
        for changes in (dict(orientation_radius=0),
                        dict(orientation_radius=math.inf),
                        dict(orientation_jump_limit=-1),
                        dict(orientation_jump_limit=math.nan),
                        dict(orientation_speed=0),
                        dict(orientation_speed=math.inf)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PreviewConfig(**changes)

    def test_invalid_pose_cannot_activate(self):
        self.ready()
        self.pose[0] = float('nan')
        with self.assertRaises(ValueError):
            self.engine.accept(packet(2, .02, 1), .02, PEER)
        self.assertEqual(self.engine.status, 'WAIT_RELEASE')

    def test_retired_session_cannot_reengage(self):
        self.active()
        self.engine.accept(packet(0, .03, 1, session='session-b'), .03, PEER)
        self.assertIsNone(self.engine.accept(packet(3, .04, 1), .04, PEER))
        self.assertIsNone(self.engine.accept(packet(4, .05, 0), .05, PEER))
        self.assertIsNone(self.engine.accept(packet(5, .06, 1), .06, PEER))
        self.assertEqual(self.reads, 1)
        self.engine.accept(packet(1, .07, 0, session='session-b'), .07, PEER)
        self.assertEqual(self.engine.accept(packet(2, .08, 1, session='session-b'), .08, PEER), POSE)

    def test_retired_source_does_not_refresh_timeout(self):
        self.active()
        new_peer = ('192.168.30.3', 40001)
        self.engine.accept(packet(3, .03, 1), .03, new_peer)
        self.engine.accept(packet(4, .2, 1), .2, PEER)
        self.engine.tick(.29)
        self.assertIn('timeout', self.engine.reason.lower())

    def test_oversized_integer_is_rejected(self):
        with self.assertRaises(ValueError):
            packet(0, 0, x=10**400)

    def test_custom_mapping(self):
        self.engine = PreviewEngine(PreviewConfig(axes=('x', 'y', '-z'), speed=1), lambda: POSE.copy())
        self.active()
        target = self.engine.accept(packet(3, .12, 1, x=.01, y=.02, z=.03), .12, PEER)
        for actual, wanted in zip(target[:3], [.1005, .201, .2985]):
            self.assertAlmostEqual(actual, wanted)


if __name__ == '__main__':
    unittest.main()
