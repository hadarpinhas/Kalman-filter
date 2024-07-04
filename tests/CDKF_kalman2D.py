import numpy as np
import matplotlib.pyplot as plt
import json

class CentralDifferenceKalmanFilter:
    def __init__(self, dt, state_dim, meas_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0

    def f(self, x, u):
        # State transition function for non-linear system
        x_new = np.array([
            x[0] + x[2] * self.dt,  # x position
            x[1] + x[3] * self.dt,  # y position
            x[2] + u[0] * self.dt,  # x velocity
            x[3] + u[1] * self.dt   # y velocity
        ])
        return x_new

    def h(self, x):
        # Measurement function for non-linear system
        return x

    def sigma_points(self, x, P):
        n = len(x)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = x
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        for i in range(n):
            sigma_pts[i + 1] = x + sqrt_P[i]
            sigma_pts[n + i + 1] = x - sqrt_P[i]
        return sigma_pts

    def run(self, x, P, U, Y):
        xKalman = []
        XkList = []

        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            sigma_pts = self.sigma_points(x, P)
            x_pred = np.mean([self.f(pt, U[k-1]) for pt in sigma_pts], axis=0)
            P_pred = np.cov(np.array([self.f(pt, U[k-1]) for pt in sigma_pts]).T) + SigK

            ######################## Correction ####################################
            sigma_pts = self.sigma_points(x_pred, P_pred)
            y_pred = np.mean([self.h(pt) for pt in sigma_pts], axis=0)
            Pyy = np.cov(np.array([self.h(pt) for pt in sigma_pts]).T) + SigV
            Pxy = np.mean([(sigma_pts[i] - x_pred).reshape(-1, 1) @ (self.h(sigma_pts[i]) - y_pred).reshape(1, -1)
                           for i in range(2 * self.state_dim + 1)], axis=0)

            K = Pxy @ np.linalg.inv(Pyy)
            x = x_pred + K @ (Y[k] - y_pred)
            P = P_pred - K @ Pyy @ K.T

            XkList.append(y_pred[:2])  # Store position only
            xKalman.append(x[:2])  # Store position only

        return XkList, xKalman

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

    cdkf = CentralDifferenceKalmanFilter(dt, state_dim=4, meas_dim=4)
    XkList, xKalman = cdkf.run(initial_x_v, P, U, measurements)

    # Plotting y position as a function of x position
    posList_x = measurements[:, 0]
    posList_y = measurements[:, 1]

    plt.figure(figsize=(8, 8))
    plt.scatter(posList_x, posList_y, label='Measured Position')
    plt.scatter([p[0] for p in XkList], [p[1] for p in XkList], label='Estimated Position')
    plt.plot([p[0] for p in xKalman], [p[1] for p in xKalman], '--r', label='Kalman Filter Path')
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.legend()
    plt.title('2D Position Plot')
    plt.grid(True)
    plt.show()
