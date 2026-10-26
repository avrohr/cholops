"""Independent Decimal covariance and hyperparameter derivatives."""

from decimal import Decimal, localcontext


def reference(x, y, theta, scenario, digits=80):
    """NLL, log-hyperparameter gradient and factor for exactly supplied inputs."""
    with localcontext() as ctx:
        ctx.prec = digits

        def D(value):
            return Decimal.from_float(float(value))

        x, y, theta = list(map(D, x)), list(map(D, y)), list(map(D, theta))
        n = len(x)
        variance, length, noise = [v.exp() for v in theta]
        zero, one = Decimal(0), Decimal(1)
        A = [[zero for _ in x] for _ in x]
        derivatives = [[[zero for _ in x] for _ in x] for _ in range(3)]
        for i in range(n):
            for j in range(n):
                d = (x[i] - x[j]) / length
                if scenario == "se_clustered":
                    signal = variance * (-d * d / 2).exp()
                    dlength = signal * d * d
                else:
                    z = Decimal(3).sqrt() * abs(d)
                    base = variance * (-z).exp()
                    signal = base * (one + z)
                    dlength = base * z * z
                A[i][j] = signal + (noise if i == j else zero)
                derivatives[0][i][j] = signal
                derivatives[1][i][j] = dlength
                derivatives[2][i][j] = noise if i == j else zero
        L = [[zero for _ in x] for _ in x]
        for i in range(n):
            for j in range(i + 1):
                v = A[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), zero)
                L[i][j] = v.sqrt() if i == j else v / L[j][j]

        def solve(b):
            z = [zero] * n
            for i in range(n):
                z[i] = (b[i] - sum((L[i][j] * z[j] for j in range(i)), zero)) / L[i][i]
            out = [zero] * n
            for i in reversed(range(n)):
                out[i] = (
                    z[i] - sum((L[j][i] * out[j] for j in range(i + 1, n)), zero)
                ) / L[i][i]
            return out

        alpha = solve(y)
        columns = [solve([one if i == j else zero for i in range(n)]) for j in range(n)]
        Q = [[columns[j][i] - alpha[i] * alpha[j] for j in range(n)] for i in range(n)]
        score = (
            sum((L[i][i].ln() for i in range(n)), zero)
            + sum((y[i] * alpha[i] for i in range(n)), zero) / 2
        ) / n
        gradient = [
            sum(
                (Q[i][j] * derivatives[k][i][j] for i in range(n) for j in range(n)),
                zero,
            )
            / (2 * n)
            for k in range(3)
        ]
        return (
            float(score),
            list(map(float, gradient)),
            [[float(v) for v in row] for row in L],
        )
