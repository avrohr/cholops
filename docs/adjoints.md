# Update, deletion and replacement adjoints

This note explains the formulas in [rank_one.py](../src/cholops/tensors/rank_one.py),
[deletion.py](../src/cholops/tensors/deletion.py)
and [replacement.py](../src/cholops/tensors/replacement.py). It concerns real lower factors
with positive diagonals and positive-definite targets. Indices are fixed during
differentiation; factor perturbations are lower triangular. Bars denote
cotangents under the Euclidean or Frobenius inner product.

## Rank-one factorization

Let $Y Y^T = L L^T + s u u^T$, where $s=1$ for a positive update and $s=-1$
for the internal downdate. For $v=L^{-1}u$, define

$$
\rho_j^2 = 1+s\sum_{k=0}^{j}v_k^2,\qquad
\rho_{-1}=1,\qquad
c_j=\frac{\rho_{j-1}}{\rho_j},\qquad
b_j=\frac{s v_j}{\rho_j\rho_{j-1}}.
$$

The positive roots exist on the stated domain. The lower factor $T$ of
$I+s v v^T$ has diagonal $T_{jj}=1/c_j$ and entries $T_{ij}=v_i b_j$ for
$i>j$. To verify this, use
$b_j^2=s(\rho_{j-1}^{-2}-\rho_j^{-2})$ to telescope the products in $T T^T$.
Consequently $Y=LT$. Substituting $Lv=u$ and
$1/c_j=c_j+v_j b_j$ gives the implemented assembly:

$$
Y_{ij}=L_{ij}c_j+
\left(u_i-\sum_{k<j}L_{ik}v_k\right)b_j\quad(i\geq j),\qquad
Y_{jj}=L_{jj}/c_j.
$$

The separate diagonal expression avoids repeating a residual subtraction.

The code evaluates $\rho_j$ differently for updates and downdates. For a
positive update, $\rho_j$ is the Euclidean norm of $(1,v_0,\ldots,v_j)$.
The code combines these entries with rescaling, avoiding overflow from
directly squaring large values of $v_k$.

For a downdate, it uses the equivalent expression

$$
\rho_j^2 = \left(1-\|v\|^2\right)+\sum_{k>j}v_k^2.
$$

It computes $1-\|v\|^2$ once, then adds the sum of squares after each index
$j$. This avoids a separate subtraction from one at every index. The initial
subtraction can still lose accuracy when $\|v\|^2$ is close to one, where
the downdated matrix approaches singularity.

## Rank-one reverse pass

For $G=\operatorname{tril}(\bar Y)$, define the exclusive suffix sum $B$ and
strictly lower residual $E$ by

$$
B_{ij}=\sum_{k>j}G_{ik}b_k,\qquad
E_{ij}=\frac{s}{\rho_j^2}
\left(u_i-\sum_{k\leq j}L_{ik}v_k\right)\quad(i>j),
$$

with $E_{ij}=0$ otherwise. The implemented reverse pass is

$$
q_j=\sum_i\left(G_{ij}E_{ij}-Y_{ij}B_{ij}\right),\qquad
Y^T z=q,\qquad
\bar u=z+Gb,\qquad
\bar L=\operatorname{tril}\left(
G\operatorname{diag}(c)-(B+z\mathbf{1}^T)\operatorname{diag}(v)
\right).
$$

To see the role of the updated factor, first differentiate $Y=LT$. Let $h$
be the cotangent of $v$ obtained from $T(v)$ with $L$ fixed, and put $a=Gb$.
Differentiating $Lv=u$ gives
$\bar u=L^{-T}h$ and $\bar L=\operatorname{tril}(GT^T-\bar u v^T)$.
The entries of $T$ above give

$$
GT^T=G\operatorname{diag}(c)+(a\mathbf{1}^T-B)\operatorname{diag}(v),\qquad
q=T^T h-Y^T a.
$$

The second identity follows by differentiating the prefix norms and collecting
the resulting suffix sums; its component form is the expression for $q$ above.
Since $Y=LT$, writing $\bar u=a+z$ turns the equation $L^T\bar u=h$ into
$Y^Tz=q$. Substitution yields the stated $\bar L$ and $\bar u$ without a
transpose solve against the original factor. The implementation evaluates
$q$ directly; it does not form $T$, $h$, or an inverse. Scans, elementwise
products and one triangular solve keep work and storage quadratic.

The same expressions with $s=-1$ give the internal downdate adjoint.
They do not describe the separate signed-weight recurrence used by
`chol_rank1`, whose weight can also be zero.

## Index-preserving replacement

Fix the replaced index $p$. Removing row and column $p$ from the old factor
gives a lower matrix $D$. Let $w$ be the removed factor column restricted to
survivors, with entries before $p$ set to zero. The surviving covariance is
$DD^T+ww^T$, so the positive-update rule gives its factor $R$.

Let $P=R_{:p,:p}$. The inserted row $a$, diagonal $d$, and column $t$ satisfy

$$
P a_{:p}=k_{:p},\qquad a_{p:}=0,\qquad
d=\sqrt{\kappa-a^Ta},\qquad
t_{:p}=0,\qquad t_{p:}=(k-Ra)_{p:}/d.
$$

Downdating $R$ by $t$ gives $W W^T=RR^T-tt^T$. Inserting row $a$, column $t$
and diagonal $d$ into $W$ recovers the desired covariance: the surviving block
is $WW^T+tt^T=RR^T$, the cross-covariance is $Wa+dt=Ra+dt=k$, and the inserted
diagonal is $a^Ta+d^2=\kappa$. Here $W$ and $R$ agree in their first $p$
columns. The implementation explicitly retains those columns from $R$.

The integrated reverse pass first reverses this assembly and the downdate.
It then accumulates the bordering contributions using

$$
\mathrm{d}a_{:p}=P^{-1}
\left(\mathrm{d}k_{:p}-(\mathrm{d}P)a_{:p}\right),\qquad
\mathrm{d}d=\frac{\mathrm{d}\kappa-2a^T\mathrm{d}a}{2d},
$$

$$
\mathrm{d}t_{p:}=
\frac{\left(\mathrm{d}k-(\mathrm{d}R)a-R\,\mathrm{d}a\right)_{p:}}{d}
-\frac{t_{p:}}{d}\,\mathrm{d}d.
$$

Finally, the positive-update adjoint maps the accumulated cotangent of $R$
back to $D$ and $w$, and insertion maps those cotangents to the old factor.
The saved index controls masks and assembly only; the returned cotangents are
for $L$, $k$ and $\kappa$. Sharing $R$ between the two rank-one rules avoids
saving it twice. Last-position replacement uses bordering alone, and the
singleton case is a scalar square root.

The [update tests](../tests/test_updates.py),
[replacement tests](../tests/test_replace.py) and
[forward-mode tests](../tests/test_forward_ad.py) compare these rules with
independent dense or high-precision references. The
[numerical limits](numerics.md#conditioning-and-precision) still apply.

## Block deletion

Deletion uses orthogonal rotations of the retained factor and the deleted-column
block. `_delete_pushforward` differentiates each rotation, carrying the tangent
of that block alongside its value. It reuses the saved rotation coefficients
and output factor.

The reverse pass starts with the final, zeroed deleted-column block. It undoes
each rotation to recover the block before that step, then uses `jax.vjp` on the
single rotation to propagate its cotangents. The retained factor, initial column
block, output factor and rotation coefficients are saved, rather than all
intermediate blocks or factors. Work is quadratic in the retained size and linear in the deletion count
in both directions. Saved storage is quadratic in the retained size plus the
size of the deleted-column block.
