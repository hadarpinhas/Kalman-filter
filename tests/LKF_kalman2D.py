import numpy as np
import matplotlib.pyplot as plt
import json

class KalmanFilter:
    def run(self, x, P, U, Y) -> np.array:
        xKalman = []
        XkList = []
        for k in range(1, Y.shape[0]):
            ####################### Prediction #####################################
            Xk = A @ x + B @ U[k-1] + Wk
            Pk = A @ P @ A.T + SigK
            Yk = C @ Xk + D @ U[k] + Vk

            ######################## Correction ####################################
            K = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + SigV)
            x = Xk + K @ (Y[k] - Yk)
            I = np.identity(len(Pk))
            P = (I - K @ C) @ Pk

            XkList.append(Xk[:2])  # Store position only
            xKalman.append(x[:2])  # Store position only

        return XkList, xKalman

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    measurements = np.array([
        [d['x_position'], d['y_position'], d['x_velocity'], d['y_velocity']] for d in data
    ])
    U = np.array([
        [d['control_input_x'], d['control_input_y']] for d in data
    ])
    return measurements, U

if __name__ == "__main__":
    kalmanFilterObj = KalmanFilter()

    dt = 1  # seconds

    # State transition matrix (A)
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Control input matrix (B)
    B = np.array([[0.5 * dt**2, 0],
                  [0, 0.5 * dt**2],
                  [dt, 0],
                  [0, dt]])

    # Observation matrix (C)
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Direct influence matrix (D)
    D = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0]])

    # Process noise
    Wk = np.zeros((4,))

    # Measurement noise
    Vk = np.zeros((4,))

    # Process noise covariance matrix (SigK)
    SigK = np.zeros((4, 4))

    # Measurement noise covariance matrix (SigV)
    SigV = np.array([[100, 0, 0, 0],
                     [0, 100, 0, 0],
                     [0, 0, 25, 0],
                     [0, 0, 0, 25]])

    # Load measurements and control inputs from JSON file
    measurements, U = load_data_from_json('measurements.json')
    U = np.zeros((U.shape[0], 2))

    # Use the first valid entry in the measurements as the initial state
    initial_x_v = measurements[0]  # assuming the first entry is valid

    # Initial process covariance matrix (P)
    P = np.array([[100, 0, 0, 0],
                  [0, 100, 0, 0],
                  [0, 0, 25, 0],
                  [0, 0, 0, 25]])

    XkList, xKalman = kalmanFilterObj.run(initial_x_v, P, U, measurements)

    # Plotting y position as a function of x position
    posList_x = np.concatenate((np.array([initial_x_v[0]]), measurements[1:, 0]), axis=0)
    posList_y = np.concatenate((np.array([initial_x_v[1]]), measurements[1:, 1]), axis=0)

    plt.figure(figsize=(8, 8))
    plt.scatter(posList_x, posList_y, label='Measured Position')
    plt.scatter([p[0] for p in XkList], [p[1] for p in XkList], label='Estimated Position')
    plt.plot([p[0] for p in xKalman], [p[1] for p in xKalman], '--r', label='Kalman Filter Path')
    plt.xlabel('x position [m]')
    plt.ylabel('y position [m]')
    plt.legend()
    plt.title('2D Position Plot - The Linear Kalman Filter')
    plt.grid(True)
    plt.show()
