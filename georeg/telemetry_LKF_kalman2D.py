import numpy as np
import matplotlib.pyplot as plt
import json

class LinearKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])  # State transition matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])  # Measurement matrix

    def f(self, x):
        # State transition function for constant velocity model
        return self.F @ x

    def h(self, x):
        # Measurement function for constant velocity model
        return self.H @ x

    def run(self, x, P, U, Y):
        xKalman = []
        XkList = []

        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            x = self.f(x)
            P = self.F @ P @ self.F.T + SigK

            ######################## Correction ####################################
            Yk = self.h(x)
            K = P @ self.H.T @ np.linalg.inv(self.H @ P @ self.H.T + SigV)
            x = x + K @ (Y[k] - Yk)
            P = (np.eye(len(P)) - K @ self.H) @ P

            XkList.append(Yk[:2])  # Store position only
            xKalman.append(x[:2])  # Store position only

        return XkList, xKalman

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    measurements = np.array([
        [d['lat'], d['lon']] for d in data
    ])
    return measurements

if __name__ == "__main__":
    dt = 1  # seconds

    # Process noise covariance matrix (SigK)
    SigK = np.eye(4) * 1e-2  # small process noise

    # Measurement noise covariance matrix (SigV)
    SigV = np.array([[0.1, 0],
                     [0, 0.1]])

    # Load measurements from JSON file
    measurements = load_data_from_json('telemetry_measurements.json')
    U = np.zeros((measurements.shape[0], 2))

    # Use the first valid entry in the measurements as the initial state
    initial_x_v = np.array([measurements[0][0], measurements[0][1], 0, 0])  # assuming the first entry is valid

    # Initial process covariance matrix (P)
    P = np.eye(4) * 100

    kf = LinearKalmanFilter(dt)
    XkList, xKalman = kf.run(initial_x_v, P, U, measurements)

    # Plotting lat as a function of lon
    latList = measurements[:, 0]
    lonList = measurements[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(latList, lonList, label='Measured Position')
    plt.scatter([p[0] for p in XkList], [p[1] for p in XkList], label='Estimated Position')
    plt.plot([p[0] for p in xKalman], [p[1] for p in xKalman], '--r', label='Kalman Filter Path')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.title('2D Position Plot - Linear Kalman Filter')
    plt.grid(True)
    plt.show()
