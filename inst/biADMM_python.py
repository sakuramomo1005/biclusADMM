

import numpy as np
from scipy import linalg

def elk_python(n,p):

    el1 = np.zeros([n,np.int(n*(n-1)/2)])
    el2 = np.zeros([n,np.int(n*(n-1)/2)])

    count = -1
    for i in list(range(n-1,0,-1)):
        temp = np.zeros([n,i])
        temp[n-i-1,:] = 1
        el1[:,(count+1):(count+i+1)] = temp
        el2[(n-i):n, (count + 1):(count + i + 1)] = np.eye(i)
        count = count+i

    ek1 = np.zeros([p,np.int(p*(p-1)/2)])
    ek2 = np.zeros([p,np.int(p*(p-1)/2)])

    count = -1
    for i in list(range(p-1,0,-1)):
        temp = np.zeros([p,i])
        temp[p-i-1,:] = 1
        ek1[:,(count+1):(count+i+1)] = temp
        ek2[(p-i):p, (count + 1):(count + i + 1)] = np.eye(i)
        count = count+i

    return(el1,el2,ek1,ek2)

def sign_c(px, x):
    res = x.copy()
    res[(px != abs(x)) & (x > 0)] = abs(px[(px != abs(x)) & (x > 0)])
    res[(px != abs(x)) & (x < 0)] = -abs(px[(px != abs(x)) & (x < 0)])
    return(res)

def project_to_simplex(x, z):
    mu = x.copy()
    mu.sort()
    mu = mu[::-1]
    cumsum = mu[0]
    for j in range(1, len(x)):
        cumsum = cumsum + mu[j]
        if(((j+1) * mu[j] - cumsum + z) <= 0):
            rho = j
            break
        rho = j + 1
    theta = (sum(mu[0:rho]) - z)/rho
    res = (x - theta)
    res[res < 0] = 0
    return(res)

def prox_L2(x, tau, par1, par2):
    px = 1 - tau/np.sqrt(np.sum(x**2,axis=0))
    px = np.where(px < 0,0,px)
    res = np.repeat(px,par1).reshape(par2,par1).T * x
    return(res)

def prox_L1(x, tau):
    px = np.abs(x) - tau
    px[px < 0] = 0
    x2 = x.copy()
    res = sign_c(px, x2)
    return(res)

def proj_L1(x, tau):
    x2 = x.copy()
    px = abs(x)
    for i in range(px.shape[1]):
        px[:,i] = project_to_simplex(px[:,i], tau[i])
    px = sign_c(px, x2)
    return(px)

def prox_Linf(x, tau):
    px = (1/tau) * x
    px = x - proj_L1(x, tau)
    return(px)

def biADMM_python(X, nu1, nu2, gamma_1, gamma_2, w_l, u_k, prox, niters, tol, output = 1):

    n = X.shape[0]
    p = X.shape[1]

    n2 = int(n*(n-1)/2)
    p2 = int(p*(p-1)/2)

    elks = elk_python(n,p)
    el1 = elks[0]
    el2 = elks[1]
    ek1 = elks[2]
    ek2 = elks[3]

    En = np.diag(list(range(n))) + np.diag(list(range(n-1,-1,-1))) - np.ones([n,n]) + np.eye(n)
    Ep = np.diag(list(range(p))) + np.diag(list(range(p-1,-1,-1))) - np.ones([p,p]) + np.eye(p)

    M = np.eye(n) + nu1 * En
    N = nu2 * Ep

    A = np.zeros(n*p).reshape(n,p)
    v = np.zeros(p*n2).reshape(p,n2)
    z = np.zeros(p2*n).reshape(n,p2)
    lambda_1 = v
    lambda_2 = z

    ## iterations
    for iters in range(int(niters)):

        A_old = A; v_old = v; z_old = z; lambda_1_old = lambda_1; lambda_2_old = lambda_2

        # update A

        lv = lambda_1 + nu1 * v
        lz = lambda_2 + nu2 * z
        C2 = 0 -np.dot((el2-el1),lv.T)
        C3 = np.dot(lz,(ek1-ek2).T)
        C = X +  C2 + C3

        A = linalg.solve_sylvester(M, N.T, C)

        al1 = np.dot(A.T,el1)
        al2 = np.dot(A.T,el2)
        ak1 = np.dot(A,ek1)
        ak2 = np.dot(A,ek2)

        # update vz
        if prox == 'l1':
          
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            temp2 = prox_L1(vtemp, sigma_1)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_L1(ztemp, sigma_2)
            z = temp4.copy()

        elif prox == 'l2':
            
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            # print(vtemp)
            temp2 = prox_L2(vtemp, sigma_1, p, n2)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_L2(ztemp, sigma_2, n, p2)
            z = temp4.copy()
            
        elif prox == 'l-inf':
            
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            temp2 = prox_Linf(vtemp, sigma_1)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_Linf(ztemp, sigma_2)
            z = temp4.copy()

        else:
            print('Error: please specify the norms of the proximal mapping')
            break

        # update lambda

        lambda_1 = lambda_1 + nu1 * (v - al1 + al2)
        lambda_2 = lambda_2 + nu2 * (z - ak1 + ak2)


        # output print

        if output == 1:
            print('iteration time: ' + str(iters))
            print('A_new - A_old: ' + str(round(np.mean(np.abs(A - A_old)),9)))
            print('v_new - v_old: ' + str(round(np.mean(np.abs(v - v_old)),9)))
            print('z_new - z_old: ' + str(round(np.mean(np.abs(z - z_old)),9)))
            print('lambda1_new - lambda1_old: ' + str(round(np.mean(np.abs(lambda_1 - lambda_1_old)),9)))
            print('lambda2_new - lambda2_old: ' + str(round(np.mean(np.abs(lambda_2 - lambda_2_old)),9)))

        # return

        conditions = ((np.mean(np.abs(A - A_old)) < tol) &
                      (np.mean(np.abs(v - v_old)) < tol) &
                      (np.mean(np.abs(z - z_old)) < tol) &
                      (np.mean(np.abs(lambda_1 - lambda_1_old))< tol) &
                      (np.mean(np.abs(lambda_2 - lambda_2_old))< tol))

        if conditions:
            return(A, v, z, lambda_1, lambda_2, iters)
            break

    return(A, v, z, lambda_1, lambda_2, iters)

def biADMM_python_compositional(X, nu1, nu2, nu3, gamma_1, gamma_2, w_l, u_k, prox, niters, tol, output = 1):

    n = X.shape[0]
    p = X.shape[1]

    n2 = int(n*(n-1)/2)
    p2 = int(p*(p-1)/2)

    elks = elk_python(n,p)
    el1 = elks[0]
    el2 = elks[1]
    ek1 = elks[2]
    ek2 = elks[3]

    En = np.diag(list(range(n))) + np.diag(list(range(n-1,-1,-1))) - np.ones([n,n]) + np.eye(n)
    Ep = np.diag(list(range(p))) + np.diag(list(range(p-1,-1,-1))) - np.ones([p,p]) + np.eye(p)

    M = np.eye(n) + nu1 * En
    N = nu2 * Ep + nu3 * np.ones(p*p).reshape(p,p)

    A = np.zeros(n*p).reshape(n,p)
    v = np.zeros(p*n2).reshape(p,n2)
    z = np.zeros(p2*n).reshape(n,p2)
    lambda_1 = v
    lambda_2 = z
    lambda_3 = np.zeros(n).reshape(n,1)

    ## iterations
    for iters in range(int(niters)):

        A_old = A; v_old = v; z_old = z;
        lambda_1_old = lambda_1; lambda_2_old = lambda_2; lambda_3_old = lambda_3

        # update A

        s = np.ones(n).reshape(n,1) + lambda_3/nu3

        lv = lambda_1 + nu1 * v
        lz = lambda_2 + nu2 * z
        C2 = 0 -np.dot((el2-el1),lv.T)
        C3 = np.dot(lz,(ek1-ek2).T)
        C4 = np.repeat(s,p).reshape(n,p) * nu3
        C = X +  C2 + C3 + C4

        A = linalg.solve_sylvester(M, N.T, C)

        al1 = np.dot(A.T,el1)
        al2 = np.dot(A.T,el2)
        ak1 = np.dot(A,ek1)
        ak2 = np.dot(A,ek2)

        # update vz

        if prox == 'l1':
          
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            temp2 = prox_L1(vtemp, sigma_1)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_L1(ztemp, sigma_2)
            z = temp4.copy()

        elif prox == 'l2':
            
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            print(vtemp)
            temp2 = prox_L2(vtemp, sigma_1, p, n2)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_L2(ztemp, sigma_2, n, p2)
            z = temp4.copy()
            
        elif prox == 'l-inf':
            
            sigma_1 = gamma_1 * w_l/nu1
            sigma_1 = sigma_1.flatten()
            vtemp = al1 - al2 - 1/nu1 * lambda_1
            temp2 = prox_Linf(vtemp, sigma_1)
            v = temp2.copy()

            sigma_2 = gamma_2 * u_k/nu2
            sigma_2 = sigma_2.flatten()
            ztemp = ak1 - ak2 - 1/nu2 * lambda_2
            temp4 = prox_Linf(ztemp, sigma_2)
            z = temp4.copy()
            
        else:
            print('Error: please specify the norms of the proximal mapping')
            break


        # update lambda

        lambda_1 = lambda_1 + nu1 * (v - al1 + al2)
        lambda_2 = lambda_2 + nu2 * (z - ak1 + ak2)
        lambda_3 = lambda_3 + nu3 * (np.ones(n).reshape(n,1) - np.dot(A, np.ones(p).reshape(p,1)))

        # output print

        if output == 1:
            print('iteration time: ' + str(iters))
            print('A_new - A_old: ' + str(round(np.mean(np.abs(A - A_old)),8)))
            print('v_new - v_old: ' + str(round(np.mean(np.abs(v - v_old)),8)))
            print('z_new - z_old: ' + str(round(np.mean(np.abs(z - z_old)),8)))
            print('lambda1_new - lambda1_old: ' + str(round(np.mean(np.abs(lambda_1 - lambda_1_old)),8)))
            print('lambda2_new - lambda2_old: ' + str(round(np.mean(np.abs(lambda_2 - lambda_2_old)),8)))
            print('lambda3_new - lambda3_old: ' + str(round(np.mean(np.abs(lambda_3 - lambda_3_old)),8)))

        # return

        conditions = ((np.mean(np.abs(A - A_old)) < tol) &
                      (np.mean(np.abs(v - v_old)) < tol) &
                      (np.mean(np.abs(z - z_old)) < tol) &
                      (np.mean(np.abs(lambda_1 - lambda_1_old))< tol) &
                      (np.mean(np.abs(lambda_2 - lambda_2_old))< tol) &
                      (np.mean(np.abs(lambda_3 - lambda_3_old))< tol))

        if conditions:
            return(A, v, z, lambda_1, lambda_2, lambda_3, iters)
            break

    return(A, v, z, lambda_1, lambda_2, lambda_3, iters)

