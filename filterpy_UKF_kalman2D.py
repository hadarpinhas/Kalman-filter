from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import matplotlib.pyplot as plt
import json

def fx(x, dt):
    """State transition function for a constant velocity model."""
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x

def hx(x):
    """Measurement function."""
    return x

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    measurements = np.array([
        [d['x_position'], d['y_position'], d['x_velocity'], d['y_velocity']] for d in data
    ])
    return measurements

if __name__ == "__main__":
    dt = 1  # seconds

    # Define sigma points
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0)

    # Initialize UKF
    ukf = UKF(dim_x=4, dim_z=4, fx=fx, hx=hx, dt=dt, points=points)

    # Process noise covariance matrix (Q)
    ukf.Q = np.eye(4) * 0.1

    # Measurement noise covariance matrix (R)
    ukf.R = np.array([[100, 0, 0, 0],
                      [0, 100, 0, 0],
                      [0, 0, 25, 0],
                      [0, 0, 0, 25]])

    # Initial state covariance matrix (P)
    ukf.P = np.eye(4) * 100

    # Load measurements from JSON file
    measurements = load_data_from_json('measurements.json')
    U = np.zeros((measurements.shape[0], 2))

    # Use the first valid entry in the measurements as the initial state
    initial_x_v = measurements[0]
    ukf.x = initial_x_v

    # Run UKF
    xKalman = []
    for k in range(1, measurements.shape[0]):
        ukf.predict()  # No control input, so we use the predict method without arguments
        ukf.update(measurements[k])
        xKalman.append(ukf.x.copy())

    xKalman = np.array(xKalman)

    # Plotting y position as a function of x position
    posList_x = measurements[:, 0]
    posList_y = measurements[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(posList_x, posList_y, label='Measured Position')
    plt.scatter(xKalman[:, 0], xKalman[:, 1], label='Estimated Position')
    plt.plot(xKalman[:, 0], xKalman[:, 1], '--r', label='Kalman Filter Path')
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.legend()
    plt.title('2D Position Plot - The Unscented Kalman Filter')
    plt.grid(True)
    plt.show()
