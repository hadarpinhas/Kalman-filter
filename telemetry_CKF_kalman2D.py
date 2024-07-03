import numpy as np
import matplotlib.pyplot as plt
import json

class CubatureKalmanFilter:
    def __init__(self, dt):
        self.dt = dt

    def f(self, x):
        # State transition function for constant velocity model
        x_new = np.array([
            x[0] + x[2] * self.dt,  # latitude
            x[1] + x[3] * self.dt,  # longitude
            x[2],                   # latitude velocity
            x[3]                    # longitude velocity
        ])
        return x_new

    def h(self, x):
        # Measurement function for constant velocity model
        return np.array([
            x[0],  # latitude
            x[1]   # longitude
        ])

    def cubature_points(self, n):
        points = np.sqrt(n) * np.vstack((np.eye(n), -np.eye(n)))
        return points

    def run(self, x, P, U, Y):
        n = x.shape[0]
        m = Y.shape[1]
        cubature_points = self.cubature_points(n)
        W = np.ones(2 * n) / (2 * n)

        xKalman = []
        XkList = []

        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            X = np.tile(x, (2 * n, 1)).T + np.linalg.cholesky(P) @ cubature_points.T
            X_pred = np.array([self.f(X[:, i]) for i in range(2 * n)]).T
            x_pred = X_pred @ W
            P_pred = (X_pred - np.tile(x_pred, (2 * n, 1)).T) @ np.diag(W) @ (X_pred - np.tile(x_pred, (2 * n, 1)).T).T + SigK

            ######################## Correction ####################################
            Y_pred = np.array([self.h(X_pred[:, i]) for i in range(2 * n)]).T
            y_pred = Y_pred @ W
            Pyy = (Y_pred - np.tile(y_pred, (2 * n, 1)).T) @ np.diag(W) @ (Y_pred - np.tile(y_pred, (2 * n, 1)).T).T + SigV
            Pxy = (X_pred - np.tile(x_pred, (2 * n, 1)).T) @ np.diag(W) @ (Y_pred - np.tile(y_pred, (2 * n, 1)).T).T
            K = Pxy @ np.linalg.inv(Pyy)
            x = x_pred + K @ (Y[k] - y_pred)
            P = P_pred - K @ Pyy @ K.T

            XkList.append(y_pred[:2])  # Store position only
            xKalman.append(x[:2])      # Store position only

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

    ckf = CubatureKalmanFilter(dt)
    XkList, xKalman = ckf.run(initial_x_v, P, U, measurements)

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
    plt.title('2D Position Plot - Cubature Kalman Filter')
    plt.grid(True)
    plt.show()
