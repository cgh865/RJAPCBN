import numpy as np

import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader



def jpq(j, q, p, M):
    j_pq = j[q * M: (q + 1) * M, p * M: (p + 1) * M]
    return j_pq



def getHi(H, i, Q, M):
    Hi =(H[:, i * M * Q: (i + 1) * M * Q])
    return Hi



def getvi(V, i,Q, M):
    vi = (V[:, i]).reshape(Q*M,1)

    return vi



def getviq(V, i, q, M):
    viq = np.mat(V[q * M:(q + 1) * M, i])

    return viq


def covM(H, V, i, N, I, Q, M,device):
    CMi = torch.zeros((N, N)) + 1j * torch.zeros((N, N))
    CMi =CMi.to(device)
    for ic in range(I):
        Hi = (getHi(H, i, Q, M))
        Vj = (getvi(V, ic,Q, M))


        Hv = torch.matmul(Hi.to(device), Vj.to(device))
        Hv1=Hv.real-1j*Hv.imag

        CMi = CMi + (torch.matmul(Hv,Hv1.permute(1,0)))
    CovMi = CMi + (torch.eye(N).to(device)+1j*torch.torch.eye(N).to(device))
    return CovMi



def lassodic(Pmax, ue_act, q, Cik, J, I, M, lamda):
    cikq = np.mat(np.linalg.norm(Cik, axis=0))
    A_act = int(ue_act[q, :].sum(axis=0))
    mu_d = 0
    if A_act != 0:
        mu_u = (Pmax / (A_act)) ** -0.5 * cikq.max()
    else:
        mu_u = 0
    for imu in range(1000000):
        mu_qk = (mu_d + mu_u) / 2
        delta_d = np.zeros((1, I))
        delta_u = np.zeros((1, I))
        delta_ik = np.zeros((1, I))
        for idu in range(I):
            if ue_act[q, idu] != 0:
                Jtest = jpq(J, q, q, M)
                cctest = cikq[0, idu]
                ctest = cikq[0, idu] - 0.5 * lamda
                deltatese = Jtest / ctest
                delta_u[0, idu] = (np.linalg.norm(jpq(J, q, q, M), ord=2) + mu_u) / (cikq[0, idu] - 0.5 * lamda)

        for ik in range(I):
            if int(ue_act[q, ik]) != 0:
                for idelta in range(1000000):
                    delta_ik[0, ik] = (delta_d[0, ik] + delta_u[0, ik]) / 2
                    Bik = np.linalg.inv(
                        jpq(J, q, q, M) + (0.5 * delta_ik[0, ik] * lamda + mu_qk) * np.mat(np.eye(M))) * Cik[:, ik]
                    hik = delta_ik[0, ik] * np.linalg.norm(Bik)
                    if hik < 1:
                        delta_d[0, ik] = delta_ik[0, ik]
                    else:
                        delta_u[0, ik] = delta_ik[0, ik]
                    if abs(delta_u[0, ik] - delta_d[0, ik]) < 0.001:
                        break
        Pdelta = 0
        ccc = sum(delta_ik[0, :])
        if sum(delta_ik[0, :]) == 0:
            Pdelta = 0
        else:
            for ikdelta in range(I):
                if delta_ik[0, ikdelta] != 0:
                    Pdelta = Pdelta + (1 / delta_ik[0, ikdelta]) ** 2
        if Pdelta < Pmax:
            mu_u = mu_qk
        else:
            mu_d = mu_qk
        if abs(mu_u - mu_d) < 0.001:
            break
    return mu_qk, delta_ik



def lassomu(Pmax, ue_act, q, Cik, J, I, M):
    cikq = np.mat(np.linalg.norm(Cik, axis=0))
    A_act = int(ue_act[q, :].sum(axis=0))
    mu_d = 0
    if A_act != 0:
        mu_u = (Pmax / (A_act)) ** -0.5 * cikq.max()
    else:
        mu_u = 0
    for imu in range(1000000):
        mu_qk = (mu_d + mu_u) / 2
        hik = 0
        for ik in range(I):
            if int(ue_act[q, ik]) != 0:
                Bik = np.linalg.inv(
                    jpq(J, q, q, M) + mu_qk * np.mat(np.eye(M))) * Cik[:, ik]
                hik = hik + np.linalg.norm(Bik) ** 2

        if hik < Pmax:
            mu_u = mu_qk
        else:
            mu_d = mu_qk
        if abs(mu_u - mu_d) < 0.0001:
            break
    return mu_qk



def uF(H, V, I, N, Q, M,device):
    Rik = 0
    v_F=0
    for iuF in range(I):
        Hi = getHi(H, iuF, Q, M)
        Vi = getvi(V, iuF,Q, M)

        v_f=torch.norm(Vi, p=1)

        Hv = torch.matmul(Hi.to(device), Vi.to(device))
        Hv1= Hv.real-1j*Hv.imag

        E = torch.eye(N).to(device)+1j*torch.eye(N).to(device)

        Rik = Rik + torch.log2(torch.linalg.det(E + torch.matmul(Hv , Hv1.permute(1, 0)) * torch.inverse(covM(H, V, iuF, N, I, Q, M,device) - torch.matmul(Hv , Hv1.permute(1, 0)))))
        v_F=v_F+v_f
    return -(torch.real(Rik)),v_F



def uFspar(H, V, lamda, I, N, Q, M):
    Rs = 0
    Linorm = 0
    for i in range(I):
        Hi = getHi(H, i, Q, M)
        Vi = getvi(V, i)
        Hv = Hi * Vi
        E = np.mat(np.eye(N))
        for qs in range(Q):
            vnorm = np.linalg.norm(getviq(V, i, qs, M), ord=2)
            Linorm = Linorm + vnorm
        Rs = Rs + np.log2(np.linalg.det(E + Hv * Hv.H * (covM(H, V, i, N, I, Q, M) - Hv * Hv.H).I))
    Rs = np.real(Rs) - np.real(lamda * Linorm)
    return Rs



def compsare(V, Q, M, I):
    A = 0
    Act = np.zeros((Q, I))
    for icom in range(I):
        Vi = getvi(V, icom)
        for qcom in range(Q):
            vp = np.linalg.norm(Vi[qcom * M: (qcom + 1) * M, 0])
            if vp != 0:
                Act[qcom, icom] = 1
                A = A + 1
    Sp = A / I
    return Sp, Act


def getvq(V, q, I, M):
    vq = np.mat(np.zeros((M * I, 1)) + 1j * np.zeros((M * I, 1)))
    for ipr in range(I):
        vik = V[q * M:(q + 1) * M, ipr]
        vq[ipr * M:(ipr + 1) * M, :] = vik
    return vq



def proj(q, V, Pmax, M, I):
    Vpro = np.mat(np.zeros((M, I)) + 1j * np.zeros((M, I)))
    vq = getvq(V, q, I, M)
    Pq = np.linalg.norm(vq, ord=2)
    if Pq ** 2 > Pmax + 0.0001:
        v_norm = (np.sqrt(Pmax) / Pq) * vq
    else:
        v_norm = vq
    for ip in range(I):
        Vpro[:, ip] = v_norm[ip * M:(ip + 1) * M, :]
    return Vpro



def powerQ(V, Q, I, M):
    P = np.zeros((1, Q))
    for q in range(Q):
        vq = getvq(V, q, I, M)
        Pq = np.linalg.norm(vq, ord=2) ** 2
        P[0, q] = Pq
    return P



def sparseV(V, Pmin, Q, M, I):
    for i in range(I):
        for q in range(Q):
            viq = getviq(V, i, q, M)
            n_viq = np.linalg.norm(viq, ord=2) ** 2
            if n_viq <= Pmin:
                V[q * M:(q + 1) * M, i] = np.mat(np.zeros((M, 1)))
    return V



def preActs(H_norm, a, Q, I):
    A = np.zeros((Q, I))
    for i in range(I):
        for q in range(Q):
            hpi = max(H_norm[:, i])
            if H_norm[q, i] > a * hpi:
                A[q, i] = 1
    return A



def blocka(A1, i, Q, M):
    A = np.zeros((M * Q, M * Q))
    for q in range(Q):
        A[q * M: (q + 1) * M, q * M: (q + 1) * M] = A1[q, i] * np.eye(M)
    return np.mat(A)



def covMp(H, V, A1, i, N, I, Q, M):
    CMi = np.zeros((N, N)) + 1j * np.zeros((N, N))
    for ic in range(I):
        Hi = np.mat(getHi(H, i, Q, M))
        Aj = blocka(A1, ic, Q, M)
        Vj = Aj * np.mat(getvi(V, ic))
        Hv = Hi * Vj
        CMi = CMi + np.dot(np.mat(Hv), np.mat(Hv).H)
    CovMi = CMi + np.eye(N)
    return CovMi


def mu_pre(JC):
    pre_data = JC

    batch_size = 1
    learning_rate = 0.001
    num_epoches = 10


    pre_data1 = torch.tensor(pre_data)
    pre_data2 = torch.reshape(pre_data1, [1, 1, 16, 12])
    pre_data3 = abs(pre_data2)
    pre_data4 = DataLoader(pre_data3, batch_size=batch_size, shuffle=True)


    model = torch.load('model_mu.pkl')
    pre_result = []
    for p_data in pre_data4:
        img_2 = p_data
        img_2 = Variable(img_2)
        out = model(img_2)
        jj = 0
        pre_result.append((out[jj].data.item()) * (out[jj].data.item()) * (out[jj].data.item()))

    return pre_result


def delta_pre(J, C, I, q, ue_act):
    deltaik = np.mat(np.zeros((1, I)))
    for id in range(I):
        if int(ue_act[q, id]) != 0:
            JC_all = np.hstack((J, C[:, id]))
            JC_real = JC_all.real
            JC_imag = JC_all.imag
            JC = np.vstack((JC_real, JC_imag))
            pre_data = JC

            batch_size = 1
            learning_rate = 0.01
            num_epoches = 10

            # 预处理
            pre_data1 = torch.tensor(pre_data)
            pre_data2 = torch.reshape(pre_data1, [1, 1, 8, 5])
            pre_data3 = abs(pre_data2)
            pre_data4 = DataLoader(pre_data3, batch_size=batch_size, shuffle=True)


            model = torch.load('model_delta.pkl')
            pre_result = []

            for p_data in pre_data4:
                img_2 = p_data
                img_2 = Variable(img_2)
                out = model(img_2)
                jj = 0
                pre_result = (1 / out[jj].data.item()) * (1 / out[jj].data.item()) * (1 / out[jj].data.item()) * (
                        1 / out[jj].data.item()) * (1 / out[jj].data.item()) * (1 / out[jj].data.item())
            deltaik[0, id] = pre_result
    return deltaik
