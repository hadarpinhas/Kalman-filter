import numpy as np
import matplotlib.pyplot as plt
import json

class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt

    def f(self, x, u):
        # State transition function for non-linear system
        x_new = np.array([
            x[0] + x[2] * self.dt,
            x[1] + x[3] * self.dt,
            x[2] + u[0] * self.dt,
            x[3] + u[1] * self.dt
        ])
        return x_new

    def h(self, x):
        # Measurement function for non-linear system
        return x

    def jacobian_F(self, x, u):
        # Jacobian of the state transition function
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return F

    def jacobian_H(self, x):
        # Jacobian of the measurement function
        H = np.eye(4)
        return H

    def run(self, x, P, U, Y) -> np.array:
        xKalman = []
        XkList = []
        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            x = self.f(x, U[k-1])
            F = self.jacobian_F(x, U[k-1])
            P = F @ P @ F.T + SigK

            ######################## Correction ####################################
            H = self.jacobian_H(x)
            Yk = self.h(x)
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + SigV)
            x = x + K @ (Y[k] - Yk)
            P = (np.eye(len(P)) - K @ H) @ P

            XkList.append(Yk[:2])  # Store position only
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
    SigK = np.zeros((4, 4))

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

    ekf = ExtendedKalmanFilter(dt)
    XkList, xKalman = ekf.run(initial_x_v, P, U, measurements)

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
    plt.title('2D Position Plot - The Extended Kalman Filter')
    plt.grid(True)
    plt.show()
