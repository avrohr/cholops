"""Independent Decimal arithmetic for local adjoints and complete GP gradients."""

from decimal import Decimal, localcontext

import numpy as np


def decimal_matrix(a):
    return [[Decimal.from_float(float(v)) for v in row] for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


def matmul(a, b):
    return [
        [sum((x * y for x, y in zip(row, col)), Decimal(0)) for col in zip(*b)]
        for row in a
    ]


def sym(a):
    return [[(a[i][j] + a[j][i]) / 2 for j in range(len(a))] for i in range(len(a))]


def solve_lower(L, B, *, transpose_factor=False):
    """Solve a matrix right-hand side using Decimal scalar substitutions."""
    n, r = len(L), len(B[0])
    X = [[Decimal(0)] * r for _ in range(n)]
    order = range(n - 1, -1, -1) if transpose_factor else range(n)
    for i in order:
        for j in range(r):
            terms = (
                (L[k][i] * X[k][j] for k in range(i + 1, n))
                if transpose_factor
                else (L[i][k] * X[k][j] for k in range(i))
            )
            X[i][j] = (B[i][j] - sum(terms, Decimal(0))) / L[i][i]
    return X


def pullback(L, G, digits=80):
    """Exact-input local Cholesky adjoint; all operations use Decimal precision."""
    with localcontext() as ctx:
        ctx.prec = digits
        L, G = decimal_matrix(L), decimal_matrix(G)
        n = len(L)
        G = [[G[i][j] if i >= j else Decimal(0) for j in range(n)] for i in range(n)]
        M = matmul(transpose(L), G)
        P = [
            [
                M[i][j] if i > j else M[i][j] / 2 if i == j else Decimal(0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        Z = solve_lower(L, sym(P), transpose_factor=True)
        Q = transpose(solve_lower(L, transpose(Z), transpose_factor=True))
        return np.asarray(Q, dtype=np.float64)


def gp(x, y, theta, scenario, digits=80):
    """GP NLL and x/y/log-parameter gradients for exactly supplied binary inputs."""
    with localcontext() as ctx:
        ctx.prec = digits

        def D(v):
            return Decimal.from_float(float(v))

        x, y, theta = list(map(D, x)), list(map(D, y)), list(map(D, theta))
        n = len(x)
        variance, length, noise = [v.exp() for v in theta]
        zero, one = Decimal(0), Decimal(1)
        A = [[zero] * n for _ in range(n)]
        derivatives = [[[zero] * n for _ in range(n)] for _ in range(3)]
        dx = [[zero] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                distance = x[i] - x[j]
                d = distance / length
                if scenario == "se_clustered":
                    signal = variance * (-d * d / 2).exp()
                    dl = signal * d * d
                    dx[i][j] = -signal * distance / (length * length)
                else:
                    z = Decimal(3).sqrt() * abs(d)
                    base = variance * (-z).exp()
                    signal = base * (one + z)
                    dl = base * z * z
                    dx[i][j] = -3 * base * distance / (length * length)
                A[i][j] = signal + (noise if i == j else zero)
                derivatives[0][i][j] = signal
                derivatives[1][i][j] = dl
                derivatives[2][i][j] = noise if i == j else zero
        L = [[zero] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                v = A[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), zero)
                L[i][j] = v.sqrt() if i == j else v / L[j][j]
        identity = [[one if i == j else zero for j in range(n)] for i in range(n)]
        alpha = solve_lower(L, solve_lower(L, [[v] for v in y]), transpose_factor=True)
        alpha = [v[0] for v in alpha]
        inverse = solve_lower(L, solve_lower(L, identity), transpose_factor=True)
        Q = [[inverse[i][j] - alpha[i] * alpha[j] for j in range(n)] for i in range(n)]
        value = (
            sum((L[i][i].ln() for i in range(n)), zero)
            + sum((y[i] * alpha[i] for i in range(n)), zero) / 2
        ) / n
        theta_gradient = [
            sum(
                (Q[i][j] * derivatives[k][i][j] for i in range(n) for j in range(n)),
                zero,
            )
            / (2 * n)
            for k in range(3)
        ]
        x_gradient = [
            sum((Q[i][j] * dx[i][j] for j in range(n)), zero) / n for i in range(n)
        ]
        y_gradient = [v / n for v in alpha]
        return float(value), tuple(
            np.asarray(v, dtype=np.float64)
            for v in (x_gradient, y_gradient, theta_gradient)
        )


def sequence(case, args, digits=80):
    """Accumulate independent GP derivatives through caller-owned index histories."""
    x0, incoming, y0, new_y, theta = map(np.asarray, args)
    x = np.concatenate((x0, incoming.ravel()))
    y = np.concatenate((y0, new_y.ravel()))
    gx, gy, gt = np.zeros(x.size), np.zeros(y.size), np.zeros(3)
    indices = np.arange(case.n)
    value = 0.0
    for t in range(case.steps):
        keep = (
            slice(case.block, None)
            if case.pattern == "fifo"
            else slice(None, case.n - case.block)
        )
        new = case.n + t * case.block + np.arange(case.block)
        indices = np.concatenate((indices[keep], new))
        v, (gxi, gyi, gti) = gp(x[indices], y[indices], theta, case.scenario, digits)
        w = (1 + t / case.steps) / case.steps
        value += w * v
        gx[indices] += w * gxi
        gy[indices] += w * gyi
        gt += w * gti
    return value, (
        gx[: case.n],
        gx[case.n :].reshape(incoming.shape),
        gy[: case.n],
        gy[case.n :].reshape(new_y.shape),
        gt,
    )


def factor_and_tangent(A, dA):
    """Scalar Cholesky and its differentiated recurrence on Decimal matrices.

    The caller owns the Decimal context and constructs the target covariance
    and its perturbation in that context, avoiding a rounded intermediate.
    """
    n = len(A)
    L = [[Decimal(0)] * n for _ in range(n)]
    dL = [[Decimal(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            tangent = dA[i][j] - sum(
                dL[i][k] * L[j][k] + L[i][k] * dL[j][k] for k in range(j)
            )
            if i == j:
                L[i][j] = value.sqrt()
                dL[i][j] = tangent / (2 * L[i][j])
            else:
                L[i][j] = value / L[j][j]
                dL[i][j] = (tangent - L[i][j] * dL[j][j]) / L[j][j]
    return np.asarray(L, dtype=np.float64), np.asarray(dL, dtype=np.float64)
