import numpy as np
import quaternion

from visnav.algo import tools


if __name__ == '__main__':
    N = 10000

    qr0 = quaternion.from_float_array(np.random.normal(0, 1, (N, 4)))
    qd0 = 0.02 * np.random.normal(0, 1, (N, 4))
    qn0 = np.zeros_like(qr0)
    for i, q in enumerate(qr0):
        qr0[i] = q.normalized()
        qn0[i] = np.quaternion(*(quaternion.as_float_array(qr0[i]) + qd0[i, :])).normalized()

    qr1 = quaternion.from_float_array(np.random.normal(0, 1, (N, 4)))
    qd1 = 0.005 * np.random.normal(0, 1, (N, 4))
    qn1 = np.zeros_like(qr0)
    for i, q in enumerate(qr1):
        qr1[i] = q.normalized()
        qn1[i] = np.quaternion(*(quaternion.as_float_array(qr1[i]) + qd1[i, :])).normalized()

    q_rel = qr1 * qr0.conj()
    qn_rel = qn1 * qn0.conj()
    diff = tools.angle_between_q_arr(q_rel, qn_rel)

    print('mean deviation: %.6f deg\n' % (
        np.sqrt(np.mean(diff**2)) * 180 / np.pi
    ))