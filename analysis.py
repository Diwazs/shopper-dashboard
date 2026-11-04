"""
Traffic Source Quality and Visitor Segmentation for an Online Retailer
Mathematical Foundation of Business Analytics - WAS Group-9

Reproducible analysis: run `python analysis.py <path_to_csv> <k_segments>`.
Sections follow the course modules:
  A. Probability   - channel purchase probability: CLT/Wilson/Hoeffding intervals,
                     Beta-Binomial Bayes, MLE for an empirical-Bayes prior, chi-square test
  B. Linear algebra - PCA computed by hand via the SVD
  C. Clustering    - k-means (Voronoi cells) vs Gaussian mixture fitted by EM
  D. Regression    - polynomial regression, 10-fold CV, gradient descent / SGD vs closed form
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize, special
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "online_shoppers_intention.csv"
K_SEG = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
SEED = 42
np.random.seed(SEED)
Z95 = stats.norm.ppf(0.975)
R = {}  # results collector

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.family": "DejaVu Sans"})
C1, C2, C3, C4 = "#1f4e79", "#c0504d", "#7f7f7f", "#e69f00"

df = pd.read_csv(CSV)
df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)
N = len(df)
R["n_sessions"] = N
R["n_buyers"] = int(df.Revenue.sum())
R["overall_rate"] = float(df.Revenue.mean())
R["n_duplicates"] = int(df.duplicated().sum())


def wilson(k, n, z=Z95):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


# ---------------------------------------------------------------- A. CHANNELS
g = df.groupby("TrafficType")["Revenue"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "k"})
g["rate"] = g.k / g.n
g[["wil_lo", "wil_hi"]] = np.array([wilson(k, n) for k, n in zip(g.k, g.n)])
g["clt_half"] = Z95 * np.sqrt(g.rate * (1 - g.rate) / g.n)           # CLT / Wald half-width
g["hoeff_half"] = np.sqrt(np.log(2 / 0.05) / (2 * g.n))               # Hoeffding half-width
# Bayesian, flat Beta(1,1) prior
g["post_mean"] = (1 + g.k) / (2 + g.n)
g["cred_lo"] = stats.beta.ppf(0.025, 1 + g.k, 1 + g.n - g.k)
g["cred_hi"] = stats.beta.ppf(0.975, 1 + g.k, 1 + g.n - g.k)


# Empirical-Bayes prior by MLE of the Beta-Binomial marginal likelihood
def negll(theta):
    a, b = np.exp(theta)
    k, n = g.k.values, g.n.values
    ll = (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
          + special.betaln(k + a, n - k + b) - special.betaln(a, b))
    return -ll.sum()


opt = optimize.minimize(negll, x0=np.log([1.0, 5.0]), method="Nelder-Mead")
a_eb, b_eb = np.exp(opt.x)
R["eb_alpha"], R["eb_beta"] = float(a_eb), float(b_eb)
R["eb_prior_mean"] = float(a_eb / (a_eb + b_eb))
R["eb_prior_strength"] = float(a_eb + b_eb)
g["eb_mean"] = (a_eb + g.k) / (a_eb + b_eb + g.n)
g["eb_lo"] = stats.beta.ppf(0.025, a_eb + g.k, b_eb + g.n - g.k)
g["eb_hi"] = stats.beta.ppf(0.975, a_eb + g.k, b_eb + g.n - g.k)
g.round(4).to_csv(f"{OUT}/channel_scorecard_all.csv")

# pooled version for tests / models: channels with fewer than 100 sessions -> "Rare"
RARE_CUT = 100
rare = g.index[g.n < RARE_CUT].tolist()
R["rare_types"] = rare
R["rare_sessions"] = int(g.loc[rare, "n"].sum())
df["Channel"] = np.where(df.TrafficType.isin(rare), "Rare", "T" + df.TrafficType.astype(str))
ct = pd.crosstab(df.Channel, df.Revenue)
chi2, p, dof, _ = stats.chi2_contingency(ct)
R["chi2_channel"] = dict(chi2=float(chi2), dof=int(dof), p=float(p),
                         cramers_v=float(np.sqrt(chi2 / (N * 1))))
# Monte-Carlo posterior comparisons (flat prior)
rng = np.random.default_rng(SEED)
big = g[g.n >= RARE_CUT].copy()
draws = {t: rng.beta(1 + r.k, 1 + r.n - r.k, 200000) for t, r in big.iterrows()}
best = big.post_mean.idxmax()
worst = big.post_mean.idxmin()
R["best_channel"] = int(best)
R["worst_channel"] = int(worst)
R["p_best_gt_worst"] = float((draws[best] > draws[worst]).mean())
R["diff_best_worst_mean"] = float((draws[best] - draws[worst]).mean())
R["diff_best_worst_ci"] = [float(x) for x in np.percentile(draws[best] - draws[worst], [2.5, 97.5])]
# probability each big channel beats the overall rate
R["p_beats_overall"] = {int(t): float((d > R["overall_rate"]).mean()) for t, d in draws.items()}

# forest plot
gg = g.sort_values("post_mean")
fig, ax = plt.subplots(figsize=(6.6, 5.6))
y = np.arange(len(gg))
ax.hlines(y, gg.wil_lo, gg.wil_hi, color=C3, lw=1.6, label="Wilson 95% interval (raw rate)")
ax.scatter(gg.rate, y, color=C3, s=18, zorder=3, label="Raw rate")
ax.scatter(gg.eb_mean, y, color=C2, marker="D", s=22, zorder=4, label="Shrunk estimate (empirical Bayes)")
ax.axvline(R["overall_rate"], color=C1, ls="--", lw=1, label=f"Overall rate {R['overall_rate']:.1%}")
ax.set_yticks(y)
ax.set_yticklabels([f"Type {t}  (n={n})" for t, n in zip(gg.index, gg.n)])
ax.set_xlabel("Purchase probability")
ax.set_xlim(-0.02, 1.02)
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
ax.set_title("Channel scorecard: raw rates vs shrunk estimates", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_channel_scorecard.png")
plt.close()

# ---------------------------------------------------------------- B. PCA via SVD
beh = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
       "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates"]
X = df[beh].copy()
for c in beh[:6]:
    X[c] = np.log1p(X[c])                       # tame heavy right skew
Xv = X.values.astype(float)
mu, sd = Xv.mean(0), Xv.std(0, ddof=1)
Xs = (Xv - mu) / sd                            # centre + scale
U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
eigvals = S**2 / (N - 1)                       # eigenvalues of the correlation matrix
evr = eigvals / eigvals.sum()
cum = np.cumsum(evr)
scores = Xs @ Vt.T                             # = U * S
R["rank_Xs"] = int(np.linalg.matrix_rank(Xs))
R["eigvals"] = [float(x) for x in eigvals]
R["evr"] = [float(x) for x in evr]
R["cum_evr"] = [float(x) for x in cum]
# check versus sklearn and versus eigen-decomposition of the correlation matrix
pca_sk = PCA().fit(Xs)
R["pca_check_evr_maxdiff"] = float(np.max(np.abs(pca_sk.explained_variance_ratio_ - evr)))
ev_corr = np.sort(np.linalg.eigvalsh(np.corrcoef(Xs.T)))[::-1]
R["pca_check_eig_maxdiff"] = float(np.max(np.abs(ev_corr - eigvals)))
K_PCA = int(np.argmax(cum >= 0.80) + 1)
R["k_pca_80"] = K_PCA
R["n_eig_gt1"] = int((eigvals > 1).sum())
# rank-k approximation error (Eckart-Young)
R["rankk_rel_error"] = {int(k): float(np.sqrt((S[k:] ** 2).sum() / (S**2).sum())) for k in range(1, 9)}
load = pd.DataFrame(Vt[:4].T, index=beh, columns=[f"PC{i+1}" for i in range(4)])
load.round(3).to_csv(f"{OUT}/pca_loadings.csv")
corr = pd.DataFrame(np.corrcoef(Xs.T), index=beh, columns=beh)
corr.round(2).to_csv(f"{OUT}/behaviour_correlations.csv")
R["corr_bounce_exit"] = float(corr.loc["BounceRates", "ExitRates"])
R["corr_prod_count_dur"] = float(corr.loc["ProductRelated", "ProductRelated_Duration"])

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax[0].bar(range(1, 9), eigvals, color=C1)
ax[0].axhline(1, color=C2, ls="--", lw=1)
ax[0].set_xlabel("Component")
ax[0].set_ylabel("Eigenvalue")
ax[0].set_title("Scree plot", fontsize=9)
ax[1].plot(range(1, 9), cum, "-o", color=C1, ms=4)
ax[1].axhline(0.8, color=C2, ls="--", lw=1)
ax[1].set_xlabel("Number of components")
ax[1].set_ylabel("Cumulative variance")
ax[1].set_title("Variance explained", fontsize=9)
ax[1].set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_pca_scree.png")
plt.close()

fig, ax = plt.subplots(figsize=(5.6, 3.4))
im = ax.imshow(load.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
ax.set_xticks(range(4))
ax.set_xticklabels(load.columns)
ax.set_yticks(range(8))
ax.set_yticklabels(beh, fontsize=8)
for i in range(8):
    for j in range(4):
        ax.text(j, i, f"{load.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
ax.grid(False)
ax.set_title("PCA loadings (first four components)", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_pca_loadings.png")
plt.close()

# ---------------------------------------------------------------- C. CLUSTERING
Z = scores[:, :K_PCA]
ks = range(2, 9)
inertia, sil, bic, ari = {}, {}, {}, {}
for k in ks:
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    inertia[k] = float(km.inertia_)
    sil[k] = float(silhouette_score(Z, km.labels_, sample_size=5000, random_state=SEED))
for k in range(1, 9):
    gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(Z)
    bic[k] = float(gm.bic(Z))
R["kmeans_inertia"], R["kmeans_silhouette"], R["gmm_bic"] = inertia, sil, bic
R["bic_best_k"] = int(min(bic, key=bic.get))
R["sil_best_k"] = int(max(sil, key=sil.get))

fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.7))
ax[0].plot(list(inertia), list(inertia.values()), "-o", color=C1, ms=4)
ax[0].set_title("k-means inertia (elbow)", fontsize=9)
ax[1].plot(list(sil), list(sil.values()), "-o", color=C1, ms=4)
ax[1].set_title("k-means silhouette", fontsize=9)
ax[2].plot(list(bic), list(bic.values()), "-o", color=C2, ms=4)
ax[2].set_title("Gaussian mixture BIC", fontsize=9)
for a_ in ax:
    a_.set_xlabel("k")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_choose_k.png")
plt.close()

km = KMeans(K_SEG, n_init=50, random_state=SEED).fit(Z)
gm = GaussianMixture(K_SEG, covariance_type="full", n_init=10, random_state=SEED).fit(Z)
lab_km, lab_gm = km.labels_, gm.predict(Z)
R["ari_kmeans_gmm"] = float(adjusted_rand_score(lab_km, lab_gm))
R["gmm_converged"] = bool(gm.converged_)
R["gmm_n_iter"] = int(gm.n_iter_)
R["gmm_weights"] = [float(w) for w in gm.weights_]
R["kmeans_silhouette_final"] = float(silhouette_score(Z, lab_km, sample_size=5000, random_state=SEED))
# order segments by median product pages (ascending) so labels are stable & interpretable
order = (pd.Series(df.ProductRelated.values).groupby(lab_km).median().sort_values().index.tolist())
remap = {old: new for new, old in enumerate(order)}
df["Seg"] = pd.Series(lab_km).map(remap).values
# align GMM labels to k-means segments by majority overlap for comparison
tab = pd.crosstab(df.Seg, lab_gm)
R["kmeans_vs_gmm_crosstab"] = tab.values.tolist()
df["SegGMM"] = lab_gm

prof = df.groupby("Seg").agg(
    n=("Revenue", "size"), buyers=("Revenue", "sum"),
    pages_prod_med=("ProductRelated", "median"), dur_prod_med=("ProductRelated_Duration", "median"),
    admin_mean=("Administrative", "mean"), info_mean=("Informational", "mean"),
    bounce_mean=("BounceRates", "mean"), exit_mean=("ExitRates", "mean"),
    pagevalue_mean=("PageValues", "mean"), new_share=("VisitorType", lambda s: (s == "New_Visitor").mean()),
    weekend_share=("Weekend", "mean"))
prof["share_sessions"] = prof.n / N
prof["conv"] = prof.buyers / prof.n
prof[["conv_lo", "conv_hi"]] = np.array([wilson(k, n) for k, n in zip(prof.buyers, prof.n)])
prof["share_buyers"] = prof.buyers / prof.buyers.sum()
prof["lift"] = prof.conv / R["overall_rate"]
prof.round(4).to_csv(f"{OUT}/segment_profile.csv")
R["seg_profile"] = prof.round(4).reset_index().to_dict(orient="records")

# do channels feed segments unevenly?
ctab = pd.crosstab(df.Channel, df.Seg)
chi2s, ps, dofs, _ = stats.chi2_contingency(ctab)
R["chi2_channel_segment"] = dict(chi2=float(chi2s), dof=int(dofs), p=float(ps),
                                 cramers_v=float(np.sqrt(chi2s / (N * (min(ctab.shape) - 1)))))
mix = (ctab.div(ctab.sum(1), axis=0)).round(3)
mix["n"] = ctab.sum(1)
mix.to_csv(f"{OUT}/channel_segment_mix.csv")
R["channel_segment_mix"] = mix.reset_index().to_dict(orient="records")
# is conversion driven by segment or by channel? within-segment channel conversion (big channels)
w = df[df.Channel != "Rare"].groupby(["Seg", "Channel"])["Revenue"].agg(["size", "mean"]).reset_index()
w.round(4).to_csv(f"{OUT}/conv_by_segment_channel.csv", index=False)
vt = pd.crosstab(df.VisitorType, df.Seg)
R["visitor_by_segment"] = (vt.div(vt.sum(0), axis=1)).round(3).reset_index().to_dict(orient="records")

# scatter + conversion bars
cols = [C1, C2, C4, "#2e8b57", "#8064a2", "#4bacc6"]
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
idx = rng.choice(N, 4000, replace=False)
for s in range(K_SEG):
    m = idx[df.Seg.values[idx] == s]
    ax[0].scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.45, color=cols[s], label=f"Segment {s+1}")
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[0].legend(fontsize=7, frameon=False, markerscale=2)
ax[0].set_title("Segments in PCA space (4,000-session sample)", fontsize=9)
xs = np.arange(K_SEG)
ax[1].bar(xs, prof.conv, color=cols[:K_SEG], yerr=[prof.conv - prof.conv_lo, prof.conv_hi - prof.conv],
          capsize=3)
ax[1].axhline(R["overall_rate"], color=C3, ls="--", lw=1)
ax[1].set_xticks(xs)
SEG_NAMES = ["Bouncers", "Light\nbrowsers", "Account-\nengaged", "Deep\nresearchers"]
ax[1].set_xticklabels([f"Seg {i+1}\n{SEG_NAMES[i] if K_SEG == 4 else ''}\n{prof.share_sessions.iloc[i]:.0%} of sessions" for i in range(K_SEG)], fontsize=7)
ax[1].set_ylabel("Purchase rate (95% interval)")
ax[1].set_title("Conversion by segment", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_segments.png")
plt.close()

# ---------------------------------------------------------------- D. REGRESSION
# Target: log(1 + ProductRelated_Duration). Numeric predictor: log(1 + ProductRelated) (polynomial term).
# Controls: channel (pooled), visitor type, weekend, month, special-day proximity.
y = np.log1p(df.ProductRelated_Duration.values)
xnum = np.log1p(df.ProductRelated.values)
xz = (xnum - xnum.mean()) / xnum.std()
ctrl = pd.get_dummies(df[["Channel", "VisitorType", "Month"]], drop_first=False).astype(float)
for drop in ["Channel_T2", "VisitorType_Returning_Visitor", "Month_May"]:   # reference groups
    ctrl = ctrl.drop(columns=drop)
ctrl["Weekend"] = df.Weekend.values.astype(float)
ctrl["SpecialDay"] = df.SpecialDay.values
C = ctrl.values


def design(deg, idx=None):
    P = np.column_stack([xz**d for d in range(1, deg + 1)])
    M = np.column_stack([np.ones(N), P, C])
    return M if idx is None else M[idx]


def lstsq_fit(M, t):
    b, *_ = np.linalg.lstsq(M, t, rcond=None)
    return b


kf = KFold(10, shuffle=True, random_state=SEED)
cv = {}
for deg in range(1, 9):
    M = design(deg)
    tr_err, te_err = [], []
    for tr, te in kf.split(M):
        b = lstsq_fit(M[tr], y[tr])
        tr_err.append(np.sqrt(np.mean((y[tr] - M[tr] @ b) ** 2)))
        te_err.append(np.sqrt(np.mean((y[te] - M[te] @ b) ** 2)))
    cv[deg] = dict(train=float(np.mean(tr_err)), cv=float(np.mean(te_err)),
                   cv_se=float(np.std(te_err, ddof=1) / np.sqrt(10)))
R["cv"] = cv
best_deg = min(cv, key=lambda d: cv[d]["cv"])
thr = cv[best_deg]["cv"] + cv[best_deg]["cv_se"]
chosen = min(d for d in cv if cv[d]["cv"] <= thr)               # one-standard-error rule
R["cv_best_deg"], R["cv_chosen_deg"] = int(best_deg), int(chosen)

# baseline: controls only (no polynomial term) to show how much the numeric predictor adds
Mb = np.column_stack([np.ones(N), C])
base_te = []
for tr, te in kf.split(Mb):
    b = lstsq_fit(Mb[tr], y[tr])
    base_te.append(np.sqrt(np.mean((y[te] - Mb[te] @ b) ** 2)))
R["cv_baseline_controls_only"] = float(np.mean(base_te))
R["y_sd"] = float(y.std())

fig, ax = plt.subplots(figsize=(5.2, 3.2))
d_ = sorted(cv)
ax.plot(d_, [cv[d]["train"] for d in d_], "-o", color=C3, ms=4, label="Training RMSE")
ax.errorbar(d_, [cv[d]["cv"] for d in d_], yerr=[cv[d]["cv_se"] for d in d_], fmt="-o", color=C2,
            ms=4, capsize=2, label="10-fold CV RMSE")
ax.axvline(chosen, color=C1, ls="--", lw=1, label=f"Chosen degree = {chosen}")
ax.set_xlabel("Polynomial degree")
ax.set_ylabel("RMSE (log seconds)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Choosing the polynomial degree by cross-validation", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_cv_degree.png")
plt.close()

# final closed-form fit with standard errors
M = design(chosen)
names = ["Intercept"] + [f"x^{d}" for d in range(1, chosen + 1)] + list(ctrl.columns)
b_cf = lstsq_fit(M, y)
res = y - M @ b_cf
dof_r = N - M.shape[1]
s2 = res @ res / dof_r
cov = s2 * np.linalg.inv(M.T @ M)
se = np.sqrt(np.diag(cov))
tvals = b_cf / se
pvals = 2 * stats.t.sf(np.abs(tvals), dof_r)
ss_tot = ((y - y.mean()) ** 2).sum()
R["r2_in_sample"] = float(1 - (res @ res) / ss_tot)
R["rmse_in_sample"] = float(np.sqrt(np.mean(res**2)))
coef = pd.DataFrame(dict(coef=b_cf, se=se, t=tvals, p=pvals,
                         pct_effect=(np.exp(b_cf) - 1) * 100), index=names)
coef.round(4).to_csv(f"{OUT}/regression_coefficients.csv")
R["coef"] = coef.round(4).reset_index().rename(columns={"index": "term"}).to_dict(orient="records")

# gradient descent from scratch on standardised columns (compare with closed form)
Mc = M[:, 1:]
cm, cs = Mc.mean(0), Mc.std(0)
cs[cs == 0] = 1
Ms = np.column_stack([np.ones(N), (Mc - cm) / cs])
b_ls = lstsq_fit(Ms, y)
H = Ms.T @ Ms / N
lam_max = np.linalg.eigvalsh(H).max()
lr = 1.0 / lam_max
w = np.zeros(Ms.shape[1])
loss_hist = []
for it in range(60000):
    r_ = Ms @ w - y
    loss_hist.append(0.5 * np.mean(r_**2))
    w -= lr * (Ms.T @ r_) / N
loss_star = 0.5 * np.mean((Ms @ b_ls - y) ** 2)
R["gd"] = dict(lr=float(lr), iters=60000, final_loss=float(loss_hist[-1]), optimal_loss=float(loss_star),
               max_coef_diff=float(np.max(np.abs(w - b_ls))),
               max_pred_diff=float(np.max(np.abs(Ms @ w - Ms @ b_ls))),
               cond_number=float(np.linalg.cond(H)),
               iters_to_1e6=int(next((i for i, l in enumerate(loss_hist) if l - loss_star < 1e-6), -1)))
# minibatch SGD with decaying step
rng2 = np.random.default_rng(SEED)
w2 = np.zeros(Ms.shape[1])
t_ = 0
for ep in range(30):
    perm = rng2.permutation(N)
    for i in range(0, N, 64):
        bi = perm[i:i + 64]
        t_ += 1
        w2 -= (0.05 / (1 + 0.002 * t_)) * (Ms[bi].T @ (Ms[bi] @ w2 - y[bi])) / len(bi)
R["sgd"] = dict(epochs=30, batch=64, final_loss=float(0.5 * np.mean((Ms @ w2 - y) ** 2)),
                max_pred_diff=float(np.max(np.abs(Ms @ w2 - Ms @ b_ls))))

fig, ax = plt.subplots(figsize=(5.2, 3.1))
ax.semilogy(np.maximum(np.array(loss_hist) - loss_star, 1e-16), color=C1)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss gap to closed-form optimum")
ax.set_title("Gradient descent convergence", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_gd_convergence.png")
plt.close()

# ---------------------------------------------------------------- E. EXPECTED VALUE
# Reallocating 1,000 sessions from the weakest to the strongest large channel (flat-prior posteriors).
dd = draws[best] - draws[worst]
R["ev_per_1000_mean"] = float(1000 * dd.mean())
R["ev_per_1000_ci"] = [float(1000 * x) for x in np.percentile(dd, [2.5, 97.5])]
R["big_channels"] = big[["n", "k", "rate", "post_mean", "cred_lo", "cred_hi"]].round(4).reset_index().to_dict(orient="records")

df[["Seg", "SegGMM"]].to_csv(f"{OUT}/segment_labels.csv", index=False)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print("done; K_SEG =", K_SEG)
