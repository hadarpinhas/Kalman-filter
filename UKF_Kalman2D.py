import numpy as np
import matplotlib.pyplot as plt
import json

class UnscentedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt

    def f(self, x):
        # State transition function for non-linear system
        x_new = np.array([
            x[0] + x[2] * self.dt,  # latitude
            x[1] + x[3] * self.dt,  # longitude
            x[2],                   # latitude velocity
            x[3]                    # longitude velocity
        ])
        return x_new

    def h(self, x):
        # Measurement function for non-linear system
        return x

    def sigma_points(self, x, P, kappa):
        n = x.shape[0]
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x
        sqrt_P = np.linalg.cholesky((n + kappa) * P)
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]
        return sigma_points

    def unscented_transform(self, sigma_points, Wm, Wc, noise_cov):
        x = np.dot(Wm, sigma_points)
        y = sigma_points - x
        P = np.dot(Wc * y.T, y) + noise_cov
        return x, P

    def run(self, x, P, U, Y, alpha=1e-3, beta=2, kappa=0):
        n = x.shape[0]
        lamb = alpha**2 * (n + kappa) - n
        Wm = np.full(2 * n + 1, 0.5 / (n + lamb))
        Wm[0] = lamb / (n + lamb)
        Wc = Wm.copy()
        Wc[0] = Wm[0] + 1 - alpha**2 + beta

        xUKF = []
        XkList = []
        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            sigma_points = self.sigma_points(x, P, lamb)
            sigma_points_pred = np.array([self.f(sp) for sp in sigma_points])
            x, P = self.unscented_transform(sigma_points_pred, Wm, Wc, SigK)

            ######################## Correction ####################################
            sigma_points = self.sigma_points(x, P, lamb)
            sigma_points_meas = np.array([self.h(sp) for sp in sigma_points])
            y, Pyy = self.unscented_transform(sigma_points_meas, Wm, Wc, SigV)
            Pxy = np.dot(Wc * (sigma_points_pred - x).T, sigma_points_meas - y)
            K = np.dot(Pxy, np.linalg.inv(Pyy))
            x += np.dot(K, (Y[k] - y))
            P -= np.dot(K, Pyy).dot(K.T)

            XkList.append(y[:2])  # Store position only
            xUKF.append(x[:2])    # Store position only

        return XkList, xUKF

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    measurements = np.array([
        [d['x_position'], d['y_position'], d['x_velocity'], d['y_velocity']] for d in data
    ])
    return measurements

if __name__ == "__main__":
    dt = 1  # seconds

    # Process noise covariance matrix (SigK)
    SigK = np.eye(4) * 1e-2  # small process noise

    # Measurement noise covariance matrix (SigV)
    SigV = np.array([[100, 0, 0, 0],
                     [0, 100, 0, 0],
                     [0, 0, 25, 0],
                     [0, 0, 0, 25]])

    # Load measurements from JSON file
    measurements = load_data_from_json('measurements.json')
    U = np.zeros((measurements.shape[0], 2))

    # Use the first valid entry in the measurements as the initial state
    initial_x_v = measurements[0]  # assuming the first entry is valid

    # Initial process covariance matrix (P)
    P = np.array([[100, 0, 0, 0],
                  [0, 100, 0, 0],
                  [0, 0, 25, 0],
                  [0, 0, 0, 25]])

    ukf = UnscentedKalmanFilter(dt)
    XkList, xUKF = ukf.run(initial_x_v, P, U, measurements)

    # Plotting lat as a function of lon
    latList = measurements[:, 0]
    lonList = measurements[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(latList, lonList, label='Measured Position')
    plt.scatter([p[0] for p in XkList], [p[1] for p in XkList], label='Estimated Position')
    plt.plot([p[0] for p in xUKF], [p[1] for p in xUKF], '--r', label='Kalman Filter Path')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.title('2D Position Plot - The Unscented Kalman Filter')
    plt.grid(True)
    plt.show()
