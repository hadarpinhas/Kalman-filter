import numpy as np
import matplotlib.pyplot as plt
import json

class UnscentedKalmanFilter:
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

    def sigma_points(self, x, P, alpha=1e-3, beta=2, kappa=0):
        n = x.shape[0]
        lamb = alpha**2 * (n + kappa) - n
        points = np.zeros((2 * n + 1, n))
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        points[0] = x
        Wm[0] = lamb / (n + lamb)
        Wc[0] = Wm[0] + 1 - alpha**2 + beta
        U = np.linalg.cholesky((n + lamb) * P)
        for k in range(n):
            points[k + 1] = x + U[k]
            points[n + k + 1] = x - U[k]
            Wm[k + 1] = Wm[n + k + 1] = 1 / (2 * (n + lamb))
            Wc[k + 1] = Wc[n + k + 1] = 1 / (2 * (n + lamb))
        return points, Wm, Wc

    def unscented_transform(self, sigma_points, Wm, Wc, noise_cov):
        x = np.dot(Wm, sigma_points)
        y = sigma_points - x
        P = np.dot(Wc * y.T, y) + noise_cov
        return x, P

    def run(self, x, P, U, Y, alpha=1e-3, beta=2, kappa=0):
        n = x.shape[0]
        m = Y.shape[1]

        xKalman = []
        XkList = []

        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            sigma_points, Wm, Wc = self.sigma_points(x, P, alpha, beta, kappa)
            sigma_points_pred = np.array([self.f(sp) for sp in sigma_points])
            x_pred, P_pred = self.unscented_transform(sigma_points_pred, Wm, Wc, SigK)

            ######################## Correction ####################################
            sigma_points, Wm, Wc = self.sigma_points(x_pred, P_pred, alpha, beta, kappa)
            Y_pred = np.array([self.h(sp) for sp in sigma_points])
            y_pred, Pyy = self.unscented_transform(Y_pred, Wm, Wc, SigV)
            Pxy = np.dot(Wc * (sigma_points_pred - x_pred).T, (Y_pred - y_pred))
            K = np.dot(Pxy, np.linalg.inv(Pyy))
            x = x_pred + np.dot(K, (Y[k] - y_pred))
            P = P_pred - np.dot(K, Pyy).dot(K.T)

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
    SigV = np.array([[0.0001, 0],
                     [0, 0.0001]])

    # Load measurements from JSON file
    measurements = load_data_from_json('telemetry_measurements.json')
    U = np.zeros((measurements.shape[0], 2))

    # Use the first valid entry in the measurements as the initial state
    initial_x_v = np.array([measurements[0][0], measurements[0][1], 0, 0])  # assuming the first entry is valid

    # Initial process covariance matrix (P)
    P = np.eye(4) * 100

    ukf = UnscentedKalmanFilter(dt)
    XkList, xKalman = ukf.run(initial_x_v, P, U, measurements)

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
    plt.title('2D Position Plot - Unscented Kalman Filter')
    plt.grid(True)
    plt.show()
