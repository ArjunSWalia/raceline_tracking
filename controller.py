import numpy as np
from numpy.typing import ArrayLike
from racetrack import RaceTrack


def get_lookahead_pt(position: ArrayLike, path: ArrayLike, closest_idx: int, lookahead_dist: float) -> ArrayLike:
    n_points = len(path)
    acc_dist = 0.0
    idx = closest_idx
    
    while acc_dist < lookahead_dist:
        next_idx = (idx + 1) % n_points
        segment_dist = np.linalg.norm(path[next_idx] - path[idx])
        acc_dist += segment_dist
        idx = next_idx
        
        if idx == closest_idx:
            break
    
    return path[idx]


def compute_curvature(path: ArrayLike, idx: int, window: int = 5) -> float:
    n_points = len(path)
    
    p1 = path[(idx - window) % n_points]
    p2 = path[idx]
    p3 = path[(idx + window) % n_points]
    

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    

    if a < 0.000001 or b < 0.000001 or c < 0.000001:
        return 0.0
    
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                      (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    curvature = 4.0 * area / (a * b * c)
    return curvature


def pure_pursuit_steering(car_pos: ArrayLike, car_heading: float,lookahead_pt: ArrayLike, wheelbase: float,lookahead_dist: float) -> float:
    angle_to_goal = np.arctan2(lookahead_pt[1] - car_pos[1], lookahead_pt[0] - car_pos[0])
    

    alpha = angle_to_goal - car_heading
    

    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    
    
    steering_angle = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_dist)
    
    return steering_angle

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    car_pos = state[0:2]
    current_velocity = state[3]
    car_heading = state[4]
    
    wheelbase = parameters[0]
    max_steering = parameters[4]
    
    path = racetrack.centerline
    
    closest_idx = np.argmin(np.linalg.norm(path-car_pos,axis=1))
    

    lookahead_dist = np.clip(0.8 * abs(current_velocity) + 8.0, 8.0, 50.0) # constants that seem to work best based on running the simulation multiple times
    lookahead_pt = get_lookahead_pt(car_pos, path, closest_idx, lookahead_dist)
    
    delta_desired = np.clip(pure_pursuit_steering(car_pos, car_heading, lookahead_pt, wheelbase, lookahead_dist), -max_steering, max_steering)
    
    n_points = len(path)
    max_curvature = 0.0
    
    # after further testing, we need to also use the previous parts of the corner as well
    # to ensure that we don't accelerate too early, leading to over correction when entering the straights 
    for i in range(-10, 30):
        idx = (closest_idx + i) % n_points
        curvature = compute_curvature(path, idx)
        max_curvature = max(max_curvature, curvature)
    
    # get the velocity we want based on the curvature
    if max_curvature < 0.000001:
        v_des = 80.0
    else:
        v_des = np.sqrt(12.0 / max_curvature)
        v_des = np.clip(v_des, 15.0, 80.0)
    
    return np.array([delta_desired, v_des])


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    assert desired.shape == (2,)
    

    current_steering = state[2]
    current_velocity = state[3]
    
    delta_desired = desired[0]
    v_des = desired[1]
    
    min_steering_rate = parameters[7]
    min_acceleration = parameters[8]
    max_steering_rate = parameters[9]
    max_acceleration = parameters[10]
    

    steering_rate = np.clip(2.0 * (delta_desired - current_steering), min_steering_rate, max_steering_rate)
    acceleration = np.clip(3.0 * (v_des - current_velocity), min_acceleration, max_acceleration)
    
    return np.array([steering_rate, acceleration])
