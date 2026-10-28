RM75 + PICO 4 Ultra 6DoF Teleoperation
Stable Snapshot
Date: 2026-10-28

==============================
已验证主要参数
==============================

Robot:
- RealMan RM75
- 7 DoF
- 6-axis force version
- Robot IP: 192.168.1.18
- C API: 1.1.6

PICO mapping:
- axes = z, -x, y
- scale = 0.2

Control:
- Remote IK
- RM75 integrated 6F / ISF model
- dq_weight = 0.4

Safety:
- joint step = 3 deg
- joint excursion = 30 deg
- orientation radius = 10 deg
- orientation jump = 3 deg
- singularity threshold = 0.01
- position error = 0.015 m
- joint error = 10 deg
- packet timeout = 0.25 s

Important:
- DO NOT use experimental J4 soft-limit patch.
- rm_algo_set_joint_limit_angle experiment returned -1.
- Keep singularity guard at 0.01.
- Keep joint-step protection at 3 deg.

Known behavior:
- TCP target/actual tracking is generally sub-mm to ~1-2 mm.
- Larger orientation motion may approach J4≈-6 deg and trigger
  singularity protection.
- This is a known remaining issue for future redundancy / J4
  singularity avoidance work.
- Do not simply lower singularity threshold.

Last known test baseline:
- 159 pytest tests passed.

Recommended live command:

.venv/bin/python Assets/Scripts/rm75_target_preview.py \
  --live \
  --live-6dof \
  --robot-ip 192.168.1.18 \
  --control-log "Logs/rm75_stable_$(date +%Y%m%d_%H%M%S).jsonl"

