# here is a Kalman filter python code for Atom IDE
def predict(mu1, var1, mu2, var2):
    ''' after motion we predict the new location'''
    mu3 = mu1 + mu2
    var3 = var1 + var2
    return [mu3,var3]

def update_data(Est, ErrEst, Meas, ErrMeas):
    for i in range(len(Meas)):
        KG = ErrEst/(ErrEst + ErrMeas)
        Est = Est + KG*(Meas[i]-Est)
        ErrEst = (1-KG)*(ErrEst)
        print('Kalman Gain = ',"%.2f" % KG,\
            '\nEst = ',"%.2f" % Est,\
            '\nErrEst = ',"%.2f" % ErrEst,'\n')

if __name__ == "__main__":

    Meas = [75, 71, 70, 74]
    ErrMeas = 4
    Est = 68
    ErrEst = 2
    update_data (Est, ErrEst, Meas, ErrMeas)
