#!/usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float64
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np # Using numpy for easier vector math

# ---------- Config ----------
PUBLISH_RATE_HZ = 100.0
TRAJECTORY_DURATION_S = 5.0  # Duration FOR EACH SEGMENT

# Robot geometry (meters)
L1 = 0.135
L2 = 0.135
L3 = 0.0467

def clamp_acos_arg(v: float) -> float:
    return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)

def normalize_angle(a: float) -> float:
    a = (a + math.pi) % (2.0 * math.pi)
    return a - math.pi

def inverse_kinematics_with_orientation(Px: float, Py: float, phi: float) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Computes both Elbow-Down and Elbow-Up IK solutions.
    Returns (ok, q_elbow_down, q_elbow_up)
    """
    Wx = Px - L3 * math.cos(phi)
    Wy = Py - L3 * math.sin(phi)
    r2 = Wx*Wx + Wy*Wy
    r  = math.sqrt(r2)

    # Reachability
    if r > (L1+L2) or r < abs(L1 - L2) or r < 1e-6:
        return (False, None, None)

    # q3 (elbow)
    cos_q3 = clamp_acos_arg((r2-L1*L1-L2*L2) / (2.0*L1*L2))
    q3_down = -math.acos(cos_q3)
    q3_up   = math.acos(cos_q3)

    # q1
    beta  = math.atan2(Wy, Wx)
    cos_a = clamp_acos_arg((r2 + L1*L1 - L2*L2) / (2.0*r*L1))
    alpha = math.acos(cos_a)
    q1_down = beta + alpha
    q1_up   = beta - alpha

    # q5 for tool orientation
    q5_down = phi - q1_down - q3_down
    q5_up   = phi - q1_up   - q3_up

    sol_down = np.array([q1_down, q3_down, q5_down])
    sol_up   = np.array([q1_up,   q3_up,   q5_up])

    return (True, sol_down, sol_up)

def forward_kinematics(q1: float, q3: float, q5: float) -> Tuple[float, float, float]:
    """Computes FK for planar-3R.
    Returns (Px, Py, phi)
    """
    # Joint 1
    P1_x = L1 * math.cos(q1)
    P1_y = L1 * math.sin(q1)
    
    # Joint 2 (Wrist)
    P2_x = P1_x + L2 * math.cos(q1 + q3)
    P2_y = P1_y + L2 * math.sin(q1 + q3)
    
    # Orientation
    phi = q1 + q3 + q5
    
    # End-Effector
    P3_x = P2_x + L3 * math.cos(phi)
    P3_y = P2_y + L3 * math.sin(phi)
    
    return (P3_x, P3_y, phi)

def get_fk_points(q1: float, q3: float, q5: float) -> List[Tuple[float, float]]:
    """
    Computes the Cartesian (x,y) positions of all joints for plotting.
    Returns [P0, P1, P2, P3]
    """
    # P0 (Origin)
    P0 = (0.0, 0.0)
    
    # P1 (End of Link 1)
    P1_x = L1 * math.cos(q1)
    P1_y = L1 * math.sin(q1)
    P1 = (P1_x, P1_y)
    
    # P2 (End of Link 2 / Wrist)
    P2_x = P1_x + L2 * math.cos(q1 + q3)
    P2_y = P1_y + L2 * math.sin(q1 + q3)
    P2 = (P2_x, P2_y)
    
    # P3 (End of Link 3 / End-Effector)
    phi = q1 + q3 + q5
    P3_x = P2_x + L3 * math.cos(phi)
    P3_y = P2_y + L3 * math.sin(phi)
    P3 = (P3_x, P3_y)
    
    return [P0, P1, P2, P3]

def cubic_coeffs(q0: float, qf: float, dq0: float, dqf: float, T: float) -> Tuple[float, float, float, float]:
    """
    Computes cubic coefficients given position and velocity boundary conditions.
    """
    if T < 1e-9:
        return (q0, dq0, 0.0, 0.0) # Return start pos with start vel
    dq = qf - q0
    T2 = T*T
    T3 = T2*T
    a0 = q0
    a1 = dq0
    a2 = (3.0*dq / T2) - (2.0*dq0 / T) - (dqf / T)
    a3 = (-2.0*dq / T3) + ((dqf + dq0) / T2)
    return (a0, a1, a2, a3)

def eval_cubic(t: float, c: Tuple[float, float, float, float]) -> float:
    a0, a1, a2, a3 = c
    return a0 + a1*t + a2*t*t + a3*t*t*t

def generate_samples(c_list: List[Tuple[float,float,float,float]], T: float, rate_hz: float, t_offset: float = 0.0):
    """ Generates samples for a single segment """
    N = max(2, int(math.ceil(T*rate_hz)) + 1)
    ts = []
    xs = []
    for i in range(N):
        t_seg = min(i / rate_hz, T) # Time within this segment
        ts.append(t_seg + t_offset)
        xs.append(eval_cubic(t_seg, c_list))
    return ts, xs

def save_configuration_plot(joint_targets: List[np.ndarray], outpath: str):
    """
    Plots the robot's link configuration at the start and final points.
    """
    plt.figure()
    ax = plt.gca()

    # --- Plot Start Configuration (Home) ---
    q_start = joint_targets[0]
    points_start = get_fk_points(q_start[0], q_start[1], q_start[2])
    px_start = [p[0] for p in points_start]
    py_start = [p[1] for p in points_start]
    
    # Plot links as gray dashed lines
    ax.plot(px_start[:2], py_start[:2], 'k--', color='gray', label='Start Config (Link 1)')
    ax.plot(px_start[1:3], py_start[1:3], 'k--', color='gray', label='Start Config (Link 2)')
    ax.plot(px_start[2:], py_start[2:], 'k--', color='gray', label='Start Config (Link 3)')

    # --- Plot Final Configuration ---
    q_final = joint_targets[-1]
    points_final = get_fk_points(q_final[0], q_final[1], q_final[2])
    px_final = [p[0] for p in points_final]
    py_final = [p[1] for p in points_final]
    
    # Plot links as colored solid lines (matches your example image)
    ax.plot(px_final[:2], py_final[:2], 'r-', linewidth=2, label='Final Config (Link 1)')
    ax.plot(px_final[1:3], py_final[1:3], 'g-', linewidth=2, label='Final Config (Link 2)')
    ax.plot(px_final[2:], py_final[2:], 'b-', linewidth=2, label='Final Config (Link 3)')

    # Plot joints as markers
    ax.plot(px_final[0], py_final[0], 'ro', markersize=8) # P0 (Origin)
    ax.plot(px_final[1], py_final[1], 'go', markersize=8) # P1
    ax.plot(px_final[2], py_final[2], 'bo', markersize=8) # P2 (Wrist)
    ax.plot(px_final[3], py_final[3], 'mo', markersize=8) # P3 (EE)

    # Plot end-effector orientation arrow for the final config
    final_phi = q_final[0] + q_final[1] + q_final[2]
    arrow_len = 0.05 # 5cm arrow
    ax.arrow(px_final[3], py_final[3], 
             arrow_len * math.cos(final_phi), 
             arrow_len * math.sin(final_phi), 
             head_width=0.01, head_length=0.02, fc='m', ec='m')

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Manipulator Configuration (Start vs. Final)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_combined_plot(all_ts: List[float], all_q1s: List[float], all_q3s: List[float], all_q5s: List[float], 
                       joint_targets: List[np.ndarray], via_point_times: List[float], outpath: str):
    plt.figure()
    # Plot continuous trajectories
    plt.plot(all_ts, all_q1s, label='Joint1 (q1)')
    plt.plot(all_ts, all_q3s, label='Joint3 (q3)')
    plt.plot(all_ts, all_q5s, label='Joint5 (q5)')

    # Extract via points for plotting
    via_q1s = [q[0] for q in joint_targets]
    via_q3s = [q[1] for q in joint_targets]
    via_q5s = [q[2] for q in joint_targets]
    
    # Plot via points as black 'x' markers
    plt.plot(via_point_times, via_q1s, 'kx', markersize=8, markeredgewidth=2, label='Via Points')
    plt.plot(via_point_times, via_q3s, 'kx', markersize=8, markeredgewidth=2)
    plt.plot(via_point_times, via_q5s, 'kx', markersize=8, markeredgewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('Multi-Segment "Fly-By" Joint Trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_cartesian_plot(all_pxs: List[float], all_pys: List[float], 
                        cartesian_targets: List[List[float]], outpath: str):
    plt.figure()
    
    # Plot continuous Cartesian trajectory
    plt.plot(all_pxs, all_pys, 'b-', label='EE Path') # Blue line

    # Extract Cartesian via points for plotting
    via_pxs = [p[0] for p in cartesian_targets]
    via_pys = [p[1] for p in cartesian_targets]
    
    # Plot home position (origin of trajectory)
    # The FK of q_home [0,0,0] is (L1+L2+L3, 0)
    home_x = L1 + L2 + L3
    plt.plot(home_x, 0.0, 'go', markersize=10, label='Start (Home)') # Green circle
    
    # Plot Cartesian via points as red 'x' markers
    plt.plot(via_pxs, via_pys, 'rx', markersize=10, markeredgewidth=2, linestyle='None', label='Via Points (Target)')

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('End-Effector Cartesian Trajectory (Y vs X)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Use equal scaling for X and Y axes
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def unwrap_solution(q_raw: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    """
    Unwraps a new joint solution to be closest to the previous solution.
    """
    q_unwrapped = q_raw.copy()
    for j in range(3): # For q1, q3, q5
        # Calculate difference, then normalize it to be within [-pi, pi]
        diff = q_unwrapped[j] - q_prev[j]
        diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
        q_unwrapped[j] = q_prev[j] + diff_wrapped
    return q_unwrapped

def main():
    rospy.init_node('ik_cubic_merged_node_py')

    # Params: Get a list of targets. Each target is [x, y, phi]
    # Example: _targets:="[[0.1, 0.2, 1.0], [0.15, 0.1, 0.5]]"
    targets_list_cartesian = rospy.get_param('~targets', [])
    if not targets_list_cartesian:
        rospy.logfatal('No via points specified. Use _targets:="[[x1,y1,p1], [x2,y2,p2], ...]"')
        return

   

    topics = rospy.get_param('~topics', [
        '/robot/joint1_position_controller/command',
        '/robot/joint3_position_controller/command',
        '/robot/joint5_position_controller/command'
    ])
    if len(topics) != 3:
        rospy.logfatal('Need exactly 3 topics for q1,q3,q5')
        return

    out_file = rospy.get_param('~output_file', 'trajectory_generated.png')
    
    # Create separate filenames for joint and cartesian plots
    out_file_base = out_file
    if out_file.endswith(".png"):
        out_file_base = out_file[:-4]
    
    out_file_joint = out_file_base + "_joint.png"
    out_file_cartesian = out_file_base + "_cartesian.png"
    out_file_config = out_file_base + "_config.png"


    # --- PASS 1: PRE-COMPUTATION ---
    rospy.loginfo("--- Pass 1: Pre-computing IK and Velocities ---")
    
    # Start at home position (zero)
    q_home = np.array([0.0, 0.0, 0.0])
    
    # Convert all cartesian targets to joint-space targets
    joint_targets = [q_home] # Start with home position
    q_prev = q_home
    
    for i, target in enumerate(targets_list_cartesian):
        x, y, phi = target[0], target[1], target[2]
        ok, q_down_raw, q_up_raw = inverse_kinematics_with_orientation(x, y, phi)
        
        if not ok:
            rospy.logfatal(f'IK failed for target {i+1} [{x},{y},{phi}]. Aborting.')
            return
        
        # Unwrapped solutions for both elbow configurations
        q_down_unwrapped = unwrap_solution(q_down_raw, q_prev)
        q_up_unwrapped   = unwrap_solution(q_up_raw, q_prev)

        # Calculate squared Euclidean distance in joint space
        dist_down = np.sum((q_down_unwrapped - q_prev)**2)
        dist_up   = np.sum((q_up_unwrapped - q_prev)**2)

        # Choose the solution with the minimum distance
        if dist_down < dist_up:
            q_chosen = q_down_unwrapped
            rospy.loginfo(f"Target {i+1}: Chose Elbow-Down solution. Distance: {dist_down:.4f}")
        else:
            q_chosen = q_up_unwrapped
            rospy.loginfo(f"Target {i+1}: Chose Elbow-Up solution. Distance: {dist_up:.4f}")

        joint_targets.append(q_chosen)
        q_prev = q_chosen # Update q_prev for the next iteration
        rospy.loginfo(f"  -> (q1,q3,q5) final: ({q_chosen[0]:.3f}, {q_chosen[1]:.3f}, {q_chosen[2]:.3f})")

    num_segments = len(joint_targets) - 1
    if num_segments == 0:
        rospy.logwarn("No segments to execute.")
        return

    # Calculate joint velocities at each via point
    joint_velocities = []
    T = TRAJECTORY_DURATION_S # Assuming constant duration T for all segments

    # Start velocity is zero
    joint_velocities.append(np.array([0.0, 0.0, 0.0]))

    # Intermediate velocities (finite difference)
    for i in range(1, num_segments):
        q_prev = joint_targets[i-1]
        q_curr = joint_targets[i]
        q_next = joint_targets[i+1]
        
        # Finite difference: dq_i = ( (q_i+1 - q_i)/T + (q_i - q_i-1)/T ) / 2
        # Simplifies to:
        dq_via = (q_next - q_prev) / (2.0 * T)
        joint_velocities.append(dq_via)
        rospy.loginfo(f"Via-point {i} velocity (dq1,dq3,dq5): ({dq_via[0]:.3f}, {dq_via[1]:.3f}, {dq_via[2]:.3f})")

    # End velocity is zero
    joint_velocities.append(np.array([0.0, 0.0, 0.0]))

    # --- PASS 2: EXECUTION & PLOTTING ---
    rospy.loginfo(f"--- Pass 2: Executing {num_segments} segments ---")

    # 3) Prepare publishers
    pubs = [rospy.Publisher(t, Float64, queue_size=1) for t in topics]
    rate = rospy.Rate(PUBLISH_RATE_HZ)
    msg = Float64()

    # Lists to store data for the final combined plot
    all_ts, all_q1s, all_q3s, all_q5s = [], [], [], []
    current_time_offset = 0.0

    for i in range(num_segments):
        segment_num = i + 1
        rospy.loginfo(f"--- Executing Segment {segment_num}/{num_segments} ---")

        # Get positions and velocities for this segment
        q0  = joint_targets[i]
        qf  = joint_targets[i+1]
        dq0 = joint_velocities[i]
        dqf = joint_velocities[i+1]

        # 2) Coeffs using the new function
        c1 = cubic_coeffs(q0[0], qf[0], dq0[0], dqf[0], TRAJECTORY_DURATION_S)
        c3 = cubic_coeffs(q0[1], qf[1], dq0[1], dqf[1], TRAJECTORY_DURATION_S)
        c5 = cubic_coeffs(q0[2], qf[2], dq0[2], dqf[2], TRAJECTORY_DURATION_S)

        # 4) Precompute plot samples for this segment
        t_seg, q1_seg = generate_samples(c1, TRAJECTORY_DURATION_S, PUBLISH_RATE_HZ, current_time_offset)
        _    , q3_seg = generate_samples(c3, TRAJECTORY_DURATION_S, PUBLISH_RATE_HZ, current_time_offset)
        _    , q5_seg = generate_samples(c5, TRAJECTORY_DURATION_S, PUBLISH_RATE_HZ, current_time_offset)
        
        # To avoid duplicate points at segment boundaries, we skip the first point if it's not the very first segment
        if i > 0:
            t_seg = t_seg[1:]
            q1_seg = q1_seg[1:]
            q3_seg = q3_seg[1:]
            q5_seg = q5_seg[1:]

        all_ts.extend(t_seg)
        all_q1s.extend(q1_seg)
        all_q3s.extend(q3_seg)
        all_q5s.extend(q5_seg)
        current_time_offset += TRAJECTORY_DURATION_S

        # 5) Execute & publish this segment
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_time).to_sec()
            te = min(t, TRAJECTORY_DURATION_S) # Clamped time for this segment

            msg.data = eval_cubic(te, c1)
            pubs[0].publish(msg)
            msg.data = eval_cubic(te, c3)
            pubs[1].publish(msg)
            msg.data = eval_cubic(te, c5)
            pubs[2].publish(msg)

            if te >= TRAJECTORY_DURATION_S:
                rospy.loginfo(f"Segment {segment_num} finished.")
                break # Move to the next segment
            
            rate.sleep()
            
    via_point_times = [i * TRAJECTORY_DURATION_S for i in range(len(joint_targets))]

    # Generate Cartesian path from joint-space samples
    rospy.loginfo("--- Generating Cartesian path for plotting ---")
    all_pxs = []
    all_pys = []
    for q1, q3, q5 in zip(all_q1s, all_q3s, all_q5s):
        px, py, _ = forward_kinematics(q1, q3, q5)
        all_pxs.append(px)
        all_pys.append(py)

    # Save joint-space plot
    save_combined_plot(all_ts, all_q1s, all_q3s, all_q5s, joint_targets, via_point_times, out_file_joint)
    rospy.loginfo('Saved combined multi-segment joint plot: %s', out_file_joint)

    # Save Cartesian-space plot
    save_cartesian_plot(all_pxs, all_pys, targets_list_cartesian, out_file_cartesian)
    rospy.loginfo('Saved combined multi-segment Cartesian plot: %s', out_file_cartesian)

    # Save Configuration plot
    save_configuration_plot(joint_targets, out_file_config)
    rospy.loginfo('Saved configuration plot: %s', out_file_config)
    
    rospy.loginfo('Multi-segment trajectory finished. Holding final pose.')
    
    # Keep publishing the final pose indefinitely
    final_q = joint_targets[-1]
    msg.data = final_q[0]
    pubs[0].publish(msg)
    msg.data = final_q[1]
    pubs[1].publish(msg)
    msg.data = final_q[2]
    pubs[2].publish(msg)
    
    rospy.spin() # Keep node alive to hold pose

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

