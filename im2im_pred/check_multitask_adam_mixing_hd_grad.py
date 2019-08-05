import numpy as np


def u(g_t, m_prev, v_prev, t, alpha=1., b1=0.9, b2=0.99, eps=1e-9):
    m_t = b1 * m_prev + (1-b1) * g_t
    v_t = b2 * v_prev + (1-b2) * g_t**2
    m_t_corr = m_t / (1-b1**t)
    v_t_corr = v_t / (1-b2**t)
    u_t = -alpha * m_t_corr / (np.sqrt(v_t_corr) + eps)
    return u_t, v_t_corr, m_t_corr, v_t, m_t


def normalized_mixing(mf):
    mixing = np.array(mf)
    mixing = mixing / mixing.sum()
    mixing *= mixing.size
    return mixing


if __name__ == '__main__':
    mixing = normalized_mixing(np.random.randn(3))
    g_t = np.array(np.random.randn(3))
    m_prev, v_prev, t = np.random.randn(1), np.random.randn(1), 100

    alpha = 1.
    b1 = 0.9
    b2 = 0.99
    eps = 1e-12

    u_t, v_t_corr, m_t_corr, v_t, m_t = u(np.sum(mixing * g_t), m_prev, v_prev, t, alpha=alpha, b1=b1, b2=b2, eps=eps)

    # grad_u_t
    bias_correction1 = 1 - b1 ** (t)
    bias_correction2 = 1 - b2 ** (t)

    beta_1_ratio = (1-b1)/bias_correction1
    beta_2_ratio = (1-b2)/bias_correction2

    def grad_mf(i):
        return -alpha * ((beta_1_ratio * g_t[i]*(np.sqrt(v_t_corr) + eps)
                          - m_t_corr * beta_2_ratio * 1/(np.sqrt(v_t_corr)+eps) * np.sum(mixing * g_t) * g_t[i])
                         /
                         (np.sqrt(v_t_corr) + eps)**2)

    # identical
    # def grad_mf2(i):
    #     return -alpha * ((beta_1_ratio * (np.sqrt(v_t_corr)+eps)-m_t_corr*beta_2_ratio*(1/(np.sqrt(v_t_corr)+eps))*np.sum(mixing * g_t))/(np.sqrt(v_t_corr)+eps)**2) * g_t[i]

    grad_mf0 = grad_mf(0)
    # grad_mf2_0 = grad_mf2(0)

    # finite differences
    mixing_plus_eps = mixing.copy()
    mixing_plus_eps[0] += eps
    mixing_minus_eps = mixing.copy()
    mixing_minus_eps[0] -= eps


    grad_mf0_fd = u(np.sum(mixing_plus_eps * g_t), m_prev, v_prev, t, alpha=alpha, b1=b1, b2=b2, eps=eps)[0] - \
                  u(np.sum(mixing_minus_eps * g_t), m_prev, v_prev, t, alpha=alpha, b1=b1, b2=b2, eps=eps)[0]
    grad_mf0_fd /= 2*eps

    pass
