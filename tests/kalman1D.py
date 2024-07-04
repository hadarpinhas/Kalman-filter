
import numpy as np
import matplotlib.pyplot as plt
        
class KalmanFilter:
    def run(self, x, P, U, Y) -> np.array:
        # U- input like acceleration, Y measurements like location and speed
        xKalman = []
        XkList = []
        for k in range(1,Y.shape[0]):

            ####################### Prediction #####################################
            # Wk​ represents the process noise, at a specific time step, which accounts for the uncertainty or randomness in the state transition process.
            # It models the inaccuracies or uncertainties in the process model. This noise affects the state Xk​ as it evolves from one time step to the next.
            Xk = A @ x + B * U[k-1] + Wk # Step 1a: State estimate time update
            # print(f"{Xk=}")


            Pk = A @ P @ A.T + SigK # Step 1b: Error covariance time update

            # C⋅Xk: This part represents the estimated measurement based on the current state.
            # D⋅U: This term adds the influence of the control input U on the measurement.
            # If the control input affects the measured output directly, 𝐷 will capture this relationship
            # Vk: This adds the measurement noise, at a specific time step. It models the inaccuracies in the measurement process.
            Yk = C @ Xk + D * U[k] + Vk # Step 1c: Output estimate


            ######################## Correction ####################################

            # if SigV (measurement noise) -> 0, then k -> 1, ie, the measurement holds higher weight
            K = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + SigV) # update Kalman gain # Step 2a: Estimator gain matrix
            # print(f"{Pk @ C.T=}") # covariance estimation error


            # updating x for the next step = the old XK + K(measurement weight or certainty) * (measurement at k  -  measurement estimation at k)
            x = Xk + K @ (Y[k] - Yk) # Step 2b: State estimate measurement update
            # print(f"{Y[k] - Yk=}") # how much to add to the x, from the new measurement to the next

            
            I = np.identity(len(Pk))
            P = (I - K @ C) @ Pk # Step 2c: Error covariance measurement update
            # print(f"{P=}")
            print()

            XkList.append(Xk[0])
            xKalman.append(x[0])

        return XkList, xKalman

if __name__ == "__main__":

    kalmanFilterObj = KalmanFilter()

    dt = 1 # seconds
                  
    A = np.array([[1, dt],
                  [0, 1]]) # process model matrix linear transforamtion

    B = np.array([0.5 * dt**2, dt]) # like acceleration coefficient, for position and speed, x = x0 + dt*v + (0.5*dt**2) * a, v = v0 + (dt) * a 

    # C This is the observation model matrix that maps the state space into the measurement space. 
    # Essentially, it transforms the state Xk to the measurement Yk.
    C = np.array([[1, 0], # observation matrix or measurement matrix
                  [0, 1]]) # H matrix - transformation of vector state from state space to (measurement) sensor space
    # example: yk = C @ x (state vector) = [1, 1] @ [Xpos] = [Xpos + Xvel]
    #                                      [0, 1]   [Xvel]   [  Xvel    ]
    # This can be useful in situations where the measurements are inherently affected by multiple state variables:
    # Doppler Radar: Measures both the position and velocity of an object, where the velocity might influence the position measurement due to the Doppler effect.
    # Motion Blur in Cameras: In a tracking system, if a camera captures an image of a moving object, the velocity of the object could cause motion blur, effectively combining position and velocity information in the measured position.


    # D matrix represents the effect of the control input U on the output Yk. control like acceleration
    # In some systems, the control input can directly influence the measurements, and D captures this relationship.
    D = np.array([0, 0])

    # Wk: External Disturbances:: Environmental factors such as wind, temperature changes, or vibrations that affect the system. Unexpected forces acting on the system, like bumps or impacts.
    #     Unmodeled Dynamics: Aspects of the system’s behavior that are not included in the model, such as friction, drag, or other forces.
    #                           Changes in the system parameters over time that are not accounted for.
    # example: Navigation Systems:In an aircraft navigation system, process noise could come from atmospheric turbulence affecting the aircraft’s trajectory.
    Wk   = 0 # state/process noise Wk, at a specific time step
    Vk   = 0 # output/measurement noise Vk, at a specific time step

    # inherent noise and uncertainties in the process model.
    SigK = np.array([[0, 0], # denoted as Q sometimes
                     [0, 0]]) # covariance of the process noise (error)

    SigV = np.array([[625,  0],
                     [  0, 36]]) # covariance of the measurement noise (error)

    # initial data
    initial_x_v = [4000., 280.] # position and speed

    x = np.array([initial_x_v[0], initial_x_v[1]]) # the state is initialized 

    P = np.array([[400, 0 ],
                  [  0, 25]]) # Eest uncertainty - process covariance matrix, (different from Covariance of the Process Noise, SigK, but built on top on it)

    measurements = np.array([[4260., 282.], [4550., 285.], [4860., 286.], [5110., 290.]]) # measurements
    measurements = np.concatenate((np.array([[np.NaN, np.NaN]]), measurements[:]), axis=0) # to account for zero measurement
    print(f"{measurements=}")

    # U This is the control input vector, representing external influences on the system, such as acceleration or force applied to an object.
    U = np.array([2, 2, 2, 2, 2]) # control state, like acceleration
    print(f"{U=}")

    XkList, xKalman = kalmanFilterObj.run(x, P, U, measurements)
    print(f"{xKalman=}")

    print(f"{np.array([initial_x_v[0]])=}")
    posList = np.concatenate((np.array([initial_x_v[0]]), measurements[1:,0]), axis=0)
    posIndex = np.arange(len(posList))
    print(f"{len(posList)=}")
    print(f"{posIndex=}")
    plt.scatter(posIndex, posList, label='xMeasure')
    print(f"{len(XkList)=}")
    plt.scatter(posIndex[1:], XkList, label='xEstimate')
    plt.plot(posIndex[1:], xKalman, '--r', label='xKalman')
    plt.xlabel('pos index')
    plt.ylabel('position [m]')
    plt.legend()

    plt.show()