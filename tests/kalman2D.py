import numpy as np
import matplotlib.pyplot as plt

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
    SigV = np.array([[625, 0, 0, 0],
                     [0, 625, 0, 0],
                     [0, 0, 36, 0],
                     [0, 0, 0, 36]])

    # Initial state (position and velocity in x and y)   
    initial_x_v = np.array([4000., 3000., 280., 200.]) # [x_position, y_position, x_velocity, y_velocity]

    # Initial process covariance matrix (P)
    P = np.array([[400, 0, 0, 0],
                  [0, 400, 0, 0],
                  [0, 0, 25, 0],
                  [0, 0, 0, 25]])

    # Measurements (positions and velocities in x and y)
    measurements = np.array([[np.NaN, np.NaN, np.NaN, np.NaN],
                             [4260., 3200., 282., 202.],
                             [4550., 3400., 285., 204.],
                             [4860., 3600., 286., 206.],
                             [5110., 3800., 290., 210.]])

    # Control input (acceleration in x and y)
    U = np.array([[2, 2],
                  [2, 2],
                  [2, 2],
                  [2, 2],
                  [2, 2]])  # [acceleration_x, acceleration_y]

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
    plt.title('2D Position Plot')
    plt.grid(True)
    plt.show()
