import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Assets' / 'Scripts'))

from rm75_live_control import (LiveControlFault, LiveLimits, LiveMotionController,
                               MotionOutcome, RecoverableLimit,
                               SdkRm75Kinematics)


JOINTS = [0.4, 9.0, 0.3, -34.9, -1.9, -56.0, -0.2]
POSE = [-0.211, 0.001, 0.705, 0.131, -1.428, -0.146]
STATE = {'joint': JOINTS, 'pose': POSE}


class FakeArm:
    def __init__(self):
        self.moves = []
        self.move_result = 0
        self.slow_stops = 0
        self.fast_stops = 0
        self.slow_result = 0
        self.fast_result = 0

    def rm_movej_follow(self, joints):
        self.moves.append(list(joints))
        if isinstance(self.move_result, BaseException):
            raise self.move_result
        return self.move_result

    def rm_set_arm_slow_stop(self):
        self.slow_stops += 1
        if isinstance(self.slow_result, BaseException):
            raise self.slow_result
        return self.slow_result

    def rm_set_arm_stop(self):
        self.fast_stops += 1
        if isinstance(self.fast_result, BaseException):
            raise self.fast_result
        return self.fast_result


class FakeKinematics:
    def __init__(self):
        self.angle = 179.5
        self.angle_calls = []
        self.solve_calls = []
        self.solution = list(JOINTS)

    def calculate_arm_angle(self, joints):
        self.angle_calls.append(list(joints))
        if isinstance(self.angle, BaseException):
            raise self.angle
        return self.angle

    def solve(self, seed, pose, arm_angle):
        self.solve_calls.append((list(seed), list(pose), arm_angle))
        if isinstance(self.solution, BaseException):
            raise self.solution
        return list(self.solution)


class FakeAlgo:
    def __init__(self):
        self.angle_result = (0, 179.5)
        self.ik_result = (0, list(JOINTS))
        self.angle_calls = []
        self.ik_calls = []
        self.singularity = 0
        self.joint_min = [-180.0] * 7
        self.joint_max = [180.0] * 7

    def rm_algo_calculate_arm_angle_from_config_rm75(self, joints):
        self.angle_calls.append(list(joints))
        return self.angle_result

    def rm_algo_inverse_kinematics(self, params):
        self.ik_calls.append(params)
        return self.ik_result

    def rm_algo_universal_singularity_analyse(self, joints, threshold):
        return self.singularity

    def rm_algo_get_joint_min_limit(self):
        return list(self.joint_min)

    def rm_algo_get_joint_max_limit(self):
        return list(self.joint_max)


class FakeParams:
    def __init__(self, seed, pose, flag):
        self.seed = list(seed)
        self.pose = list(pose)
        self.flag = flag


class CommandTests(unittest.TestCase):
    def make_controller(self):
        self.arm = FakeArm()
        self.kinematics = FakeKinematics()
        return LiveMotionController(self.arm, lambda: STATE,
                                    self.kinematics, lambda: 0.0)

    def test_prepare_fixes_arm_angle(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.assertTrue(controller.prepared)
        self.assertEqual(self.kinematics.angle_calls, [JOINTS])
        self.assertEqual(controller.last_solution, JOINTS)

    def test_arm_angle_failure(self):
        controller = self.make_controller()
        self.kinematics.angle = LiveControlFault('arm angle failed')
        with self.assertRaisesRegex(LiveControlFault, 'arm angle failed'):
            controller.prepare(STATE)
        self.assertFalse(controller.prepared)

    def test_first_target_uses_measured_joint_seed(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        result = controller.send_target(POSE, 0.0)
        self.assertEqual(result.outcome, MotionOutcome.SENT)
        self.assertEqual(self.kinematics.solve_calls, [(JOINTS, POSE, 179.5)])

    def test_deadband_holds_last_target(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        controller.send_target(POSE, 0.0)
        close = list(POSE)
        close[0] += 0.0001
        result = controller.send_target(close, 0.01)
        self.assertEqual(result.outcome, MotionOutcome.UNCHANGED)
        self.assertEqual(self.arm.moves[-1], JOINTS)
        self.assertEqual(len(self.kinematics.solve_calls), 1)

    def test_rotation_outside_angular_deadband_is_not_discarded(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        controller.send_target(POSE, 0.0)
        rotated = list(POSE)
        rotated[5] += math.radians(1.0)

        result = controller.send_target(rotated, 0.01)

        self.assertEqual(result.outcome, MotionOutcome.SENT)
        self.assertEqual(len(self.kinematics.solve_calls), 2)
        self.assertEqual(self.arm.moves[-1], JOINTS)

    def test_nonfinite_or_wrong_length_solution(self):
        for solution in ([0.0] * 6, [0.0] * 6 + [math.nan]):
            controller = self.make_controller()
            controller.prepare(STATE)
            self.kinematics.solution = solution
            with self.subTest(solution=solution), self.assertRaises(LiveControlFault):
                controller.send_target(POSE, 0.0)
            self.assertEqual(self.arm.moves, [])

    def test_ik_failure(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.kinematics.solution = LiveControlFault('IK failed')
        with self.assertRaisesRegex(LiveControlFault, 'IK failed'):
            controller.send_target(POSE, 0.0)
        self.assertEqual(self.arm.moves, [])

    def test_three_degree_boundary(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.kinematics.solution[3] += 3.0
        self.assertEqual(controller.send_target(POSE, 0.0).outcome,
                         MotionOutcome.SENT)
        self.assertEqual(len(self.arm.moves), 1)

    def test_discontinuous_solution_rejected_before_send(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.kinematics.solution[3] += 3.0001
        result = controller.send_target(POSE, 0.0)
        self.assertEqual(result.outcome, MotionOutcome.HOLD)
        self.assertIn('joint step', result.reason)
        self.assertEqual(self.arm.moves, [])

    def test_cumulative_joint_excursion_is_rejected_from_prepared_reference(self):
        controller = self.make_controller()
        controller.prepare(STATE)

        excursions = (
            2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
            17.5, 20.0, 22.5, 25.0, 27.5, 30.0
        )

        for index, excursion in enumerate(excursions, start=1):
            target = list(POSE)
            target[0] += index * .001
            self.kinematics.solution[3] = JOINTS[3] + excursion
            controller.send_target(target, index * .01)

        rejected = list(POSE)
        rejected[0] += .013
        self.kinematics.solution[3] = JOINTS[3] + 30.001

        accepted_target = list(controller.last_target)
        accepted_solution = list(controller.last_solution)

        result = controller.send_target(rejected, 0.13)

        self.assertEqual(result.outcome, MotionOutcome.HOLD)
        self.assertIn('excursion', result.reason)

        # 12 safe 2.5-degree increments were accepted.
        self.assertEqual(len(self.arm.moves), 12)

        # Rejected candidate must not modify accepted state.
        self.assertEqual(controller.last_target, accepted_target)
        self.assertEqual(controller.last_solution, accepted_solution)

    def test_joint_follow_uses_accepted_ik_solution(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        controller.send_target(POSE, 0.0)
        self.assertEqual(self.arm.moves, [JOINTS])
        self.assertTrue(controller.has_sent)
        self.assertEqual(controller.last_target, POSE)

    def test_nonzero_joint_follow_result_faults(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.arm.move_result = -4
        with self.assertRaisesRegex(LiveControlFault, '-4'):
            controller.send_target(POSE, 0.0)
        self.assertTrue(controller.has_sent)

    def test_recoverable_hold_keeps_state_and_reverse_can_exit(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        controller.send_target(POSE, 0.0)
        accepted_target = list(controller.last_target)
        accepted_solution = list(controller.last_solution)
        self.kinematics.solution = RecoverableLimit('singularity boundary')

        held = controller.send_target([POSE[0] + .01, *POSE[1:]], .01)

        self.assertEqual(held.outcome, MotionOutcome.HOLD)
        self.assertEqual(controller.last_target, accepted_target)
        self.assertEqual(controller.last_solution, accepted_solution)
        self.assertEqual(len(self.arm.moves), 1)

        self.kinematics.solution = list(JOINTS)
        recovered = controller.send_target([POSE[0] - .001, *POSE[1:]], .02)
        self.assertEqual(recovered.outcome, MotionOutcome.SENT)
        self.assertEqual(self.kinematics.solve_calls[-1][0], accepted_solution)
        self.assertEqual(len(self.arm.moves), 2)

    def test_first_and_repeated_recoverable_holds_send_nothing_then_reverse(self):
        controller = self.make_controller()
        controller.prepare(STATE)
        self.kinematics.solution = RecoverableLimit('joint limit boundary')

        first = controller.send_target([POSE[0] + .01, *POSE[1:]], 0.0)
        second = controller.send_target([POSE[0] + .02, *POSE[1:]], 0.01)

        self.assertEqual((first.outcome, second.outcome),
                         (MotionOutcome.HOLD, MotionOutcome.HOLD))
        self.assertFalse(controller.has_sent)
        self.assertIsNone(controller.last_target)
        self.assertEqual(controller.last_solution, JOINTS)
        self.assertEqual(self.arm.moves, [])

        self.kinematics.solution = list(JOINTS)
        recovered = controller.send_target([POSE[0] - .001, *POSE[1:]], .02)
        self.assertEqual(recovered.outcome, MotionOutcome.SENT)
        self.assertEqual(self.kinematics.solve_calls[-1][0], JOINTS)
        self.assertEqual(self.arm.moves, [JOINTS])

    def test_slow_ik_makes_target_stale_before_send(self):
        clock = [0.0]
        arm = FakeArm()
        kinematics = FakeKinematics()
        original_solve = kinematics.solve
        def slow_solve(seed, pose, angle):
            result = original_solve(seed, pose, angle)
            clock[0] = .25
            return result
        kinematics.solve = slow_solve
        controller = LiveMotionController(arm, lambda: STATE, kinematics,
                                          lambda: clock[0])
        controller.prepare(STATE)
        with self.assertRaisesRegex(LiveControlFault, 'stale'):
            controller.send_target(POSE, 0.0)
        self.assertEqual(arm.moves, [])
        self.assertFalse(controller.has_sent)

    def test_invalid_limits(self):
        for name in ('deadband', 'orientation_deadband', 'max_joint_step',
                     'max_joint_excursion',
                     'monitor_interval', 'max_position_error',
                     'max_orientation_error', 'max_joint_error',
                     'max_state_read_time', 'max_command_age'):
            for value in (0, -1, math.inf, math.nan, True):
                values = {name: value}
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LiveLimits(**values)


class KinematicsTests(unittest.TestCase):
    def test_sdk_adapter_validates_results(self):
        algo = FakeAlgo()
        kinematics = SdkRm75Kinematics(algo, FakeParams)
        self.assertEqual(kinematics.calculate_arm_angle(JOINTS), 179.5)
        self.assertEqual(kinematics.solve(JOINTS, POSE, 179.5), JOINTS)
        params = algo.ik_calls[0]
        self.assertEqual((params.seed, params.pose, params.flag),
                         (JOINTS, POSE, 1))

    def test_sdk_adapter_rejects_failures(self):
        algo = FakeAlgo()
        kinematics = SdkRm75Kinematics(algo, FakeParams)
        for result in ((-1, 0.0), (0, math.nan), (0, 'bad')):
            algo.angle_result = result
            with self.subTest(angle=result), self.assertRaises(LiveControlFault):
                kinematics.calculate_arm_angle(JOINTS)
        for result in ((-2, JOINTS), (0, [0.0] * 6),
                       (0, [0.0] * 6 + [math.inf])):
            algo.ik_result = result
            with self.subTest(ik=result), self.assertRaises(LiveControlFault):
                kinematics.solve(JOINTS, POSE, 179.5)

    def test_singularity_and_joint_boundaries_are_typed_recoverable_limits(self):
        algo = FakeAlgo()
        kinematics = SdkRm75Kinematics(algo, FakeParams)
        algo.singularity = -1
        with self.assertRaisesRegex(RecoverableLimit, 'singular'):
            kinematics.solve(JOINTS, POSE, 179.5)

        algo.singularity = 0
        algo.ik_result = (0, [-175.0, *JOINTS[1:]])
        with self.assertRaisesRegex(RecoverableLimit, 'minimum'):
            kinematics.solve(JOINTS, POSE, 179.5)


class MonitorAndStopTests(unittest.TestCase):
    def make_controller(self, states=None, clock_values=None):
        self.arm = FakeArm()
        self.kinematics = FakeKinematics()
        self.states = iter(states or [STATE])
        self.clock_values = [0.0, *(clock_values or [])]
        self.clock_index = 0
        def clock():
            value = self.clock_values[min(self.clock_index,
                                          len(self.clock_values) - 1)]
            self.clock_index += 1
            return value
        controller = LiveMotionController(
            self.arm, lambda: next(self.states), self.kinematics,
            clock)
        controller.prepare(STATE)
        controller.send_target(POSE, 0.0)
        return controller

    def test_monitor_not_due(self):
        controller = self.make_controller(states=[])
        self.assertIsNone(controller.monitor(0.199))

    def test_monitor_accepts_tracking_state(self):
        actual = {'joint': [value + 1.0 for value in JOINTS],
                  'pose': [POSE[0] + .001, *POSE[1:]]}
        controller = self.make_controller(states=[actual], clock_values=[1.0, 1.1])
        self.assertEqual(controller.monitor(.20), actual)
        self.assertIsNone(controller.monitor(.39))

    def test_position_error_fault(self):
        actual = {'joint': JOINTS, 'pose': [POSE[0] + .015001, *POSE[1:]]}
        controller = self.make_controller(states=[actual], clock_values=[1.0, 1.1])
        with self.assertRaisesRegex(LiveControlFault, 'position'):
            controller.monitor(.20)

    def test_orientation_error_fault(self):
        actual_pose = list(POSE)
        actual_pose[5] += math.radians(5.001)
        actual = {'joint': JOINTS, 'pose': actual_pose}
        controller = self.make_controller(states=[actual], clock_values=[1.0, 1.1])
        with self.assertRaisesRegex(LiveControlFault, 'orientation'):
            controller.monitor(.20)

    def test_joint_error_fault(self):
        actual = {'joint': [JOINTS[0] + 10.001, *JOINTS[1:]], 'pose': POSE}
        controller = self.make_controller(states=[actual], clock_values=[1.0, 1.1])
        with self.assertRaisesRegex(LiveControlFault, 'joint'):
            controller.monitor(.20)

    def test_slow_state_read_fault(self):
        controller = self.make_controller(states=[STATE], clock_values=[1.0, 1.251])
        with self.assertRaisesRegex(LiveControlFault, '0.25'):
            controller.monitor(.20)

    def test_state_read_at_time_limit_faults(self):
        controller = self.make_controller(states=[STATE], clock_values=[1.0, 1.25])
        with self.assertRaisesRegex(LiveControlFault, '0.25'):
            controller.monitor(.20)

    def test_release_stop_once(self):
        controller = self.make_controller()
        controller.stop('Grip released')
        controller.stop('Repeated')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 0))
        self.assertFalse(controller.prepared)
        self.assertFalse(controller.has_sent)
        self.assertEqual(controller.stop_reason, 'Grip released')
        self.assertIsNone(controller.last_target)
        self.assertIsNone(controller.last_solution)

    def test_stop_without_sent_motion_is_noop(self):
        arm = FakeArm()
        controller = LiveMotionController(arm, lambda: STATE,
                                          FakeKinematics(), lambda: 0.0)
        controller.prepare(STATE)
        controller.stop('No command')
        self.assertEqual((arm.slow_stops, arm.fast_stops), (0, 0))

    def test_slow_stop_failure_falls_back_to_fast_stop(self):
        controller = self.make_controller()
        self.arm.slow_result = -5
        controller.stop('Fault')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))

    def test_slow_stop_exception_falls_back(self):
        controller = self.make_controller()
        self.arm.slow_result = OSError('transport')
        controller.stop('Fault')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))

    def test_keyboard_interrupt_during_slow_stop_runs_fast_stop_then_propagates(self):
        controller = self.make_controller()
        self.arm.slow_result = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            controller.stop('Operator interrupt')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))
        controller.stop('Repeated')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))

    def test_interrupt_before_stop_call_leaves_transaction_resumable(self):
        controller = self.make_controller()
        stop_code = LiveMotionController.stop.__code__
        injected = [False]
        def trace(frame, event, arg):
            if (not injected[0] and frame.f_code is stop_code and event == 'line' and
                    frame.f_locals.get('should_stop') is True and
                    self.arm.slow_stops == 0):
                injected[0] = True
                raise KeyboardInterrupt
            return trace
        sys.settrace(trace)
        try:
            with self.assertRaises(KeyboardInterrupt):
                controller.stop('Operator interrupt')
        finally:
            sys.settrace(None)
        self.assertTrue(controller.has_sent)
        controller.stop('Resume cleanup')
        self.assertEqual(self.arm.slow_stops + self.arm.fast_stops, 1)
        self.assertFalse(controller.has_sent)

    def test_both_stops_fail_reports_fault(self):
        controller = self.make_controller()
        self.arm.slow_result = -5
        self.arm.fast_result = -6
        with self.assertRaisesRegex(LiveControlFault, '-6'):
            controller.stop('Fault')
        controller.stop('Again')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))

    def test_fast_stop_interrupt_reports_unconfirmed_stop(self):
        controller = self.make_controller()
        self.arm.slow_result = -5
        self.arm.fast_result = KeyboardInterrupt()
        with self.assertRaisesRegex(LiveControlFault, 'Fast stop raised'):
            controller.stop('Fault')
        self.assertEqual((self.arm.slow_stops, self.arm.fast_stops), (1, 1))
        self.assertFalse(controller.has_sent)

    def test_prepare_resets_stop_latch(self):
        controller = self.make_controller()
        controller.stop('Release')
        controller.prepare(STATE)
        controller.send_target(POSE, 1.0)
        controller.stop('Release again')
        self.assertEqual(self.arm.slow_stops, 2)


if __name__ == '__main__':
    unittest.main()
