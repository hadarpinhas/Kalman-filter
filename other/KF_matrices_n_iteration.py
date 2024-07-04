
#Kalman filter
# Error in Estimation = P,  Estimation = X
# Error in Measurment = R,  Measurement = Y
''' 1) Xk = A*X(k-1) + B*Uk + Wk
    2) Pk-1
P(k-1) = Initial Error in estimation (state covariance matrix) done only once
    3) Pkp = A * P(k-1) * A(transpose) +Q
    4) KG = Pk * H(transpose)/(H * Pk * Htranspose +R)
        ( -> KG = ErrEst/(ErrEst + ErrMeas) )
    5) Yk = C*Yk +Zk (measurement)
    6) Xk = Xkp(predict) + KG*(Yk - H * Xk)
        (-> Est = Est + KG*(Meas[i]-Est))
    7) Pk = (I - K*H) * Pk
        (-> ErrEst = (1-KG)*(ErrEst) )

        Estimation prediction
        observation/measurement updating
    '''
# A - updating matrix
# X - state vector
# B - translation of accelaration 'a' into position 'x'
# Uk - control variable matrix - like acceleration which ...
# ...controls the speed and location
# Wk - nosie in the process
# Yk - position measurement at iteration k from measurement
# Zr - measurement noise
# P - state covariance matrix
# Q - Process noise measurement covariance matrix keeps the..
# ... state covariance matrix from becoming too small or going to 0
# R - measuemnt covariance matix (error in measurement)
# KG -  kalman gain
# A,H,C - transformation matrix for matrices size fitting

import numpy as np

def Kalman_filtering (Xk, Uk, Wk, Pk, R, Q, Yk, Zk, dt, x, v):

    for i in range(1,len(x)):
        Yk = np.matrix([[x[i]],
                    [v[i]]])

        # 1) The predicted/estimated step A - updating matrix
        A = np.matrix([ [1, dt],
                        [0, 1] ])

        #  B - translation of accelaration 'a' into position 'x'
        B = np.matrix([[0.5*dt**2],
                [dt]])

        Xk =A*Xk+B*Uk +Wk #  predicted position of iteration k
        print('Xk is ',Xk)

        # 2,3) The predicted process covariance matrix - only adding noise
        Pk = A * Pk * np.transpose(A) + Q
        Pk[0,1] = 0# zeros for simplicity
        Pk[1,0] = 0# zeros for simplicity
        print('Pk is ', Pk)
        # 4) Kalman gain with H as a unit matrix
        H = np.matrix([[1,0],
                        [0,1]])

        # print(Pk)
        # print(H)
        # print(H*Pk* np.transpose(H))
        # print(R)

        R[0,1] = 0# zeros for simplicity
        R[1,0] = 0# zeros for simplicity
        KG = Pk* np.transpose(H) / (H*Pk* np.transpose(H) +R)
        KG = np.nan_to_num(KG)
        print(KG)
        #5) The new observation with Zr errors correction, C as a unit matrix, Zr =0
        C = np.matrix([[1,0],
                        [0,1]])
        Yk = C * Yk +Zk
        print(Yk)
        # 6) calculate the current state Yk - measurement/observation
        #                               Xk - estimation/prediction
        Xk = Xk + KG * (Yk - H*Xk)
        print('Updated Xk is',Xk)

        # 7) Updating the process covariance matrix
        I = np.matrix([[1,0],
                        [0,1]])
        Pk = (I-KG*H)*Pk
        print('updated Pk is',Pk)

    return Xk


if __name__ == '__main__':
# Initial conditions
  # ax- acceleration, v - velocity, dt- time iterval, x- position interval

    dt = 1
    ax = 2
    x = [4000, 4260, 4550, 4860]#, 5110]# position observation/measuremnet
    v = [280, 282, 285, 286]#, 290]# velocity observation/measuremnet
    dv = 6 # Error in velocity measurement
    dx = 25 # Error in velocity measurement
    dPx = 20 # Error in velocity prediction/estimation
    dPv = 5 # Error in velocity prediction/estimation

    # initial Estimation/prediction (= first observation/measurement)
    Xk0 = np.matrix([ [x[0]],
                    [v[0]] ])
    Uk = ax
    Wk = 0 # no noise

    # Initial error in estimation/prediction (process covariance matrix)
    Pk0 = np.matrix([[dPx**2,     0],
                    [0    , dPv**2] ])
    Q = 0 # no noise

    # Initial measurment
    #Since we took the first measurement x[0] as our prediction Xk0, the... ...
    #... second measurment x[1] will be our initial measurment
    Yk0 = np.matrix([ [x[1]],
                    [v[1]] ])
    # print(Yk0)
    Zk = 0 # no noise in observation/measurement

    # Initial measurment error, R -measurement/observation error
    R = np.matrix([[dx**2,0],
                    [0, dv**2]]) # zeors for simplicity
    Final_state = Kalman_filtering(Xk0, Uk, Wk, Pk0, R, Q, Yk0, Zk, dt, x, v)
    print(Final_state)
