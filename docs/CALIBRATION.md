# Calibration methods

One section per method, in the order they are reviewed. Code in `fp/calibration/`.

Notation. A detector turns an episode into a score trajectory $s(1),\dots,s(T)$ (higher means more likely to fail). A calibrator turns the trajectories of calibration *successes* into a per-step threshold $b_\alpha(t)$; the detector alarms at the first step with $s(t) \ge b_\alpha(t)$. $\alpha$ is the nominal false-alarm level. A false alarm is any alarm on a success.

## Which method gets which calibrator (`protocol.py`)

| method | calibrator | note |
|---|---|---|
| SAFE-MLP, SAFE-LSTM | SAFE band (SAFE-faithful quantile) | their own |
| FailureSpot | SAFE-style band | |
| our probes (all heads and label variants) | max-conformal | the band may be reported as a secondary table; LTT separately |
| STAC (original, single) | max-conformal | STAC accumulates non-negative per-step scores, so the maximum over steps is the final value of the accumulated score; checked on the saved STAC scores: no negative per-step value in 500 episodes for each of pi0.5 and GR00T LIBERO-10 (both variants) and in 4,360 LIBERO-X episodes |
| ActProbe | max-conformal | as in the ActProbe paper |
| Mahalanobis, Euclidean (k-NN), PCA-KMeans | SAFE band | as SAFE itself evaluates them; every one of their 26 grid configurations gets the same calibrator. Their score can be negative, so the maximum over steps is not the final value (unlike STAC) |

The table is one dictionary in `protocol.py`; the registry only builds scorers. Probe names follow `<lstm|mlp>[_mha]_<plain|modes|clip|modes_clip>__<label variant>` (for example `lstm_mha_modes_clip__onset_pre`); an unknown name is an error, never a default.

## 1. SAFE functional conformal band (`safe_band.py`)

Source: SAFE (Gu et al., NeurIPS 2025, arXiv 2506.09937). SAFE uses the functional conformal band of "The Importance of Being a Band: Finite-Sample Exact Distribution-Free Prediction Sets for Functional Data" (arXiv 2102.06746, as cited in SAFE's source) with the *Tfunc* modulation and the *mean* regression, in a one-sided (upper) form. The code is a port of `FunctionalPredictor.get_one_sided_prediction_band`, checked against SAFE's own class by a test (alpha 0.01, 0.1, 0.3).

### Step 0: data preparation (`SafeBand.fit`)

The $n$ calibration successes have different lengths. Let $L=\max_i T_i$. Each trajectory is extended to length $L$ by repeating its last value (edge padding):

$$\tilde s_i(t) = s_i(\min(t, T_i)), \qquad t = 1,\dots,L .$$

The padded trajectories are shuffled with a seeded generator and split into a *fit* part $\mathcal F$ of size $n_1=\lfloor 0.3\,n\rfloor$ and a *conformal* part $\mathcal C$ of size $m=n-n_1$. At least $n_1\ge 2$ and $m\ge 1$ are required (so $n \ge 7$).

### Step 1: mean trajectory (from $\mathcal F$ only)

$$\mu(t) = \frac{1}{n_1}\sum_{i\in\mathcal F} \tilde s_i(t).$$

### Step 2: Tfunc modulation (from $\mathcal F$ only, depends on $\alpha$)

Deviation of each fit trajectory and its worst point over time:

$$d_i(t)=\lvert \tilde s_i(t)-\mu(t)\rvert, \qquad D_i=\max_t d_i(t).$$

Let $k=\lceil (n_1+1)(1-\alpha)\rceil$. If $k>n_1$, keep every fit trajectory. Otherwise let $\gamma$ be the $k$-th smallest $D_i$ and keep $\mathcal K=\{i: D_i\le\gamma\}$, the $k$ best-behaved trajectories. The modulation is the pointwise worst deviation among the kept ones:

$$\sigma_\alpha(t)=\max_{i\in\mathcal K} d_i(t)+\varepsilon, \qquad \varepsilon=10^{-8}.$$

$\sigma_\alpha(t)$ is the band's local width scale: wide where successes vary, narrow where they agree.

### Step 3: conformity scores (from $\mathcal C$)

One-sided (upper) and sup-normalised, one number per conformal trajectory:

$$r_j=\max_{t}\ \frac{\tilde s_j(t)-\mu(t)}{\sigma_\alpha(t)}, \qquad j\in\mathcal C .$$

### Step 4: band width and band

SAFE's rule (`finite_sample=False`, the default, reproduces SAFE):

$$w_\alpha=\operatorname{Quantile}_{1-\alpha}\big(\{r_j\}_{j\in\mathcal C}\big)\quad\text{(linear interpolation, as }\texttt{np.quantile}).$$

Finite-sample rule (`finite_sample=True`): with $r_{(1)}\le\dots\le r_{(m)}$ the sorted scores,

$$w_\alpha^{\rm fs}=r_{(\lceil (m+1)(1-\alpha)\rceil)}\quad(\,=+\infty \text{ if the rank exceeds } m).$$

Two rank computations live in this file on purpose: the finite-sample width uses `conformal_rank` (rounded, see section 2), while the Tfunc $k=\lceil (n_1+1)(1-\alpha)\rceil$ in step 2 keeps SAFE's plain floating-point ceiling so that the port stays identical to SAFE's code.

$$\boxed{\,b_\alpha(t)=\mu(t)+w_\alpha\,\sigma_\alpha(t)\,}$$

The detector alarms at $\tau=\min\{t:\ s(t)\ge b_\alpha(t)\}$.

### Step 5: episodes longer than $L$ (`SafeBand.threshold`)

For $t>L$ the band is $b_\alpha(L)$. Beyond the longest fit episode $\mu$ and $\sigma_\alpha$ are constant, so this is exactly the band that padding the calibration data further would give, and each $r_j$ is already the supremum over *all* steps. The guarantee below therefore also covers episodes longer than $L$, as long as they are exchangeable with the calibration successes.

### What is guaranteed

The finite-sample rule makes the split-conformal statement valid: for a new success exchangeable with the calibration successes (up to ties),

$$\Pr\big[\exists t:\ s_{\rm new}(t)\ge b_\alpha(t)\big]\le\alpha .$$

SAFE's interpolated quantile sits at rank $(m-1)(1-\alpha)+1$ instead of $\lceil (m+1)(1-\alpha)\rceil$, so for continuous scores the realised false-alarm probability is

$$\Pr[\text{false alarm}]\ \approx\ \alpha+\frac{1-2\alpha}{m+1},$$

exceeding the nominal level by $(1-2\alpha)/(m+1)$. Simulation through the real code path (`SafeBand.fit`, `threshold`, `alarm_step`; exchangeable score trajectories; mean over 300 calibration draws of 300 test episodes; variable-length = lengths uniform on 20..40, so edge padding matters). Columns: formula | SAFE rule, fixed length | SAFE rule, variable length | finite-sample rule, variable length.

| $n$ ($m$) | $\alpha=0.05$ | 0.10 | 0.20 | 0.30 |
|--:|--:|--:|--:|--:|
| 20 (14) | .110 \| .096 \| .099 \| **.000** | .153 \| .147 \| .153 \| .067 | .240 \| .226 \| .244 \| .187 | .327 \| .312 \| .321 \| .270 |
| 26 (19) | .095 \| .089 \| .088 \| .046 | .140 \| .135 \| .141 \| .100 | .230 \| .224 \| .224 \| .203 | .320 \| .309 \| .314 \| .297 |
| 66 (47) | .069 \| .063 \| .068 \| .040 | .117 \| .113 \| .113 \| .079 | .213 \| .210 \| .206 \| .185 | .308 \| .304 \| .309 \| .283 |
| 200 (140) | .056 \| .058 \| .055 \| .049 | .106 \| .105 \| .108 \| .101 | .204 \| .212 \| .201 \| .198 | .303 \| .302 \| .299 \| .305 |

Reading: with the 20-26 calibration successes of LIBERO-10, SAFE's rule gives about 9-10% false alarms at a nominal 5%; the finite-sample rule stays at or below $\alpha$ (at $n=20$, $\alpha=0.05$ its rank exceeds $m$, so it never alarms). Variable lengths change the realised rate very little. With the 66 or more successes of LIBERO-X the excess is within 2 points. Script: `checks/check_safe_band_false_alarm_rate.py`.

Other limits:
- The guarantee is marginal over the calibration sample and holds for successes *exchangeable with the calibration successes* (the held-out half of the seen tasks), not for unseen tasks.
- $\mu$ and $\sigma_\alpha$ come from only $30\%$ of the data ($n_1$ as low as 6), which makes the band shape noisy but does not affect validity.
- The result depends on the shuffle seed; `SafeBand(seed)` makes it reproducible.

### Behaviour on real scores

Script: `checks/check_safe_band_on_real_scores.py` (saved scores of earlier models, band seed = split seed, the old 40-point alpha grid 0.005..0.5, unseen tasks). Runs: LIBERO-X, 14 per model; LIBERO-10, 3 seeds per policy and method.

**The smallest modulation is far from $\varepsilon$.** Over all runs the minimum $\sigma_\alpha(t)$ is $2\times10^{-3}$ (LIBERO-X LSTM onset) to $0.09$ (SAFE-LSTM); LIBERO-10 SAFE-MLP reaches $5\times10^{-3}$ (GR00T) and its smallest-to-largest ratio is $3\times10^{-4}$. So the $10^{-8}$ blow-up is not reached, but $\sigma$ spans up to four orders of magnitude within a run.

**The band is not monotone in $\alpha$, and the only possible cause is the Tfunc step.** If $\sigma$ were held fixed, the conformity scores $r_j$ would be fixed, $w_\alpha=\operatorname{Quantile}_{1-\alpha}(\{r_j\})$ would be non-increasing in $\alpha$, and since $\sigma>0$ the band $\mu+w_\alpha\sigma$ would be pointwise non-increasing. So any rise comes from $\sigma_\alpha$ depending on $\alpha$. The mechanism: raising $\alpha$ lowers $k$; each time $k$ drops, the worst remaining fit trajectory is excluded, $\sigma_\alpha$ shrinks, the conformity scores grow, and $w_\alpha$ jumps. Example (LIBERO-X, LSTM onset, task split seed 0; fit part 21, conformal part 51):

| $\alpha$ | kept $k$ | $\max\sigma$ | $w_\alpha$ | $\max_t b_\alpha$ |
|--:|--:|--:|--:|--:|
| 0.132 | 20 | 0.713 | 2.54 | 2.05 |
| 0.145 | 19 | 0.559 | 3.64 | 2.30 |

The band went *up* when $\alpha$ went up. In this run the band's maximum is 1.9-3.6 (over $\alpha=0.005$-$0.195$) while the successes' scores stay below 0.95, so at the steps where the band exceeds the score range the detector cannot alarm at all. Over the key-plot runs the band rose with $\alpha$ in all 14 runs for every LIBERO-X model (largest rise: 1.4 SAFE-LSTM, 26 LSTM onset, 11 onset_pre, 12 onset_traj), and in 2-3 of 3 runs per LIBERO-10 policy and method (largest 4.5 SAFE-MLP, at most 0.13 SAFE-LSTM).

**Measured FPR is not monotone in $\alpha$** (on the unseen tasks): in 9-13 of 14 LIBERO-X runs per model the FPR falls somewhere along the grid, by up to 15 (SAFE-LSTM) to 31 (onset_traj) points; in LIBERO-10, 0-2 of 3 runs, up to 8 points. This is consistent with the jumps above. Consequences: (i) the curve is an envelope: `area_to_fpr` sorts by FPR and takes the running maximum, which is the best TPR early reachable at each FPR over the whole grid (an oracle choice of $\alpha$); (ii) `sweep`'s rule "stop at the first row with FPR $\ge 40$" can stop before a later, lower-FPR, higher-TPR row. Running the full grid always avoids (ii). Measured effect on the band areas below (bracketed columns of the script output): at most 0.5 points (LSTM onset, skill, 23.0 to 22.5), usually under 0.2; for max-conformal, whose FPR is monotone in $\alpha$, none.

**Area to FPR 40 of TPR early** (LIBERO-X, unseen tasks, mean over runs; no run stopped short of 40% FPR). SAFE band, SAFE's interpolated rule | finite-sample rule; the last four rows, for comparison, are the package's max-conformal rule (LIBERO-10: a GR00T SAFE-MLP run does stop short with the finite-sample rule, 1 of 3):

| model | task (3) | signature (3) | skill (8) |
|---|--:|--:|--:|
| SAFE-LSTM, band | 44.3 \| 43.9 | 45.6 \| 46.1 | 29.3 \| 28.5 |
| LSTM onset, band | 37.0 \| 37.5 | 31.6 \| 32.3 | 23.0 \| 22.4 |
| LSTM onset_pre, band | 33.0 \| 32.6 | 36.7 \| 36.3 | 29.5 \| 30.2 |
| LSTM onset_traj, band | 34.6 \| 34.1 | 36.6 \| 37.1 | 26.1 \| 23.8 |
| SAFE-LSTM, max-conformal | 45.3 | 37.9 | 25.5 |
| LSTM onset, max-conformal | 37.6 | 33.4 | 23.0 |
| LSTM onset_pre, max-conformal | 41.5 | 46.5 | 40.3 |
| LSTM onset_traj, max-conformal | 39.6 | 36.4 | 20.2 |

The quantile rule moves the areas by at most 2.3 points (mostly under 1). The calibration method moves them much more: with the band, `LSTM onset_pre` loses 8.5-11 points on all three splits relative to max-conformal and is no longer ahead of SAFE-LSTM on signature or skill (36.7 vs 45.6 and 29.5 vs 29.3), while SAFE-LSTM itself gains on signature and skill. Old models, one run per fold, no error bars.

### Code map

| formula | code |
|---|---|
| edge padding, shuffle, 30/70 split | `SafeBand.fit` |
| $\mu(t)$ | `upper_band`: `fit.mean(axis=0)` |
| $\sigma_\alpha(t)$, $k$, $\gamma$, $\mathcal K$ | `tfunc_modulation` |
| $r_j$, $w_\alpha$, $w^{\rm fs}_\alpha$, $b_\alpha(t)$ | `upper_band` |
| extension past $L$ | `SafeBand.threshold` |

### Tests (`tests/test_safe_band.py`, 14)
Reproducibility and seed dependence; band above the mean trajectory; smaller $\alpha$ gives a higher band *on average* (not pointwise, see above); edge padding, extension and truncation; too few successes; equality with SAFE's own `FunctionalPredictor` for three alphas; the finite-sample width is the $\lceil (m+1)(1-\alpha)\rceil$-th conformity score and is infinite when the rank exceeds $m$; the finite-sample rule keeps the false-alarm rate at most $\alpha$ (n = 26, alpha 0.1 and 0.2); SAFE's rule at n = 26, alpha = 0.05 gives 0.07-0.115 (formula 0.095).

Optimism of the band curves: the area is the running-maximum envelope over a non-monotone sweep, i.e. the best TPR early reachable at each FPR when $\alpha$ is chosen after seeing the evaluation data. For the band this envelope is looser than for max-conformal (whose sweep is monotone), so band and max-conformal areas are not like-for-like. How much looser it is, measured: see "Decided" 2 below (about 0.1 points on average on the probe runs, up to 12 in single runs). The band is also time-varying and a constant threshold is not, which differs from the envelope effect.

### Decided
1. **Quantile rule: SAFE-faithful** (`finite_sample=False`, the default). The finite-sample rule stays available for any claim about the false-alarm level.
2. **Sweep: the alphas 0.005 ... 0.40, all of them** (`stop_fpr=None` is the default of `sweep`; the early stop is optional). Effect on the band areas versus stopping at 40% FPR was at most 0.5 points on the earlier models; on the 1,280 probe runs the envelope adds 0.10 on average over a stop at the first row with FPR 40 or more (largest single run 12.3; max-conformal 0.01, largest 0.9), and the band's FPR falls somewhere along the alpha grid in 92% of those runs (largest fall 37.5 points), max-conformal's in none.
   **Why not beyond 0.40:** a calibrator at level $\alpha$ promises FPR $\le\alpha$ (for episodes like the calibration ones). Only $\alpha \le 0.40$ promises an FPR inside the plotted range 0 to 40%, so an operating point from a larger $\alpha$ is not used to fill in the area below 40%, however low its FPR happens to be on the evaluation episodes (decided 2026-10-21; the full grid 0.005 ... 0.995 is `ALPHAS_FULL`, kept for diagnostics). Measured on the probe runs, unseen sets: mean area change -0.46 (max-conformal) and -0.24 (band); 62 of 2,560 calibrated evaluations lose more than 5 points, all of them curves that stay below 40% FPR for every $\alpha \le 0.40$ (small calibration sets, or detectors that never alarm on successes), which are held flat to 40 and flagged `extended`. The report lists how many there are. The stronger alternative, crediting a point with at least its guaranteed FPR, i.e. $x=\max(\text{FPR},\alpha)$, costs 4.6 (max-conformal) and 2.2 (band) points on average; it was not adopted because it also penalises a calibrator that is merely conservative. The guarantee itself holds for exchangeable episodes, which unseen tasks are not.
3. **Protocol for the main results: SAFE models are calibrated with their own band, our probes with max-conformal.** Consequence, with areas already computed (LIBERO-X, unseen tasks; SAFE-LSTM with the band, `LSTM onset_pre` with max-conformal): task 44.3 vs 41.5, signature 45.6 vs 46.5, skill 29.3 vs 40.3. The signature-split lead of `onset_pre` over SAFE-LSTM that max-conformal-for-both showed (37.9 vs 46.5) becomes a tie; the skill-split lead stays (about 11 points); on the task split SAFE-LSTM is ahead. These comparisons are across calibrators, so they are not like-for-like (the band's area is an optimistic envelope over a non-monotone sweep); the same-calibrator comparison (max-conformal for both) remains a secondary table.

## 2. Max-conformal (`max_conformal.py`, `ranks.py`)

Split-conformal prediction applied to one number per episode: the highest score the episode reaches. It is the calibration used for our probes in the main table (next to the SAFE band) and for STAC. Lineage: the STAC authors' code thresholds at the plain quantile `np.quantile(calib, 1 - alpha)`; the previous pipeline added the finite-sample level $\lceil (n+1)(1-\alpha)\rceil/n$ but still interpolated; this package uses the exact order statistic below. So it is a corrected version of the authors' rule, not the rule itself.

### Steps

**Step 1, one number per calibration success.** For each of the $n$ calibration successes, the maximum over its own steps:

$$M_i=\max_{t\le T_i}\ s_i(t),\qquad i=1,\dots,n .$$

Sorted: $M_{(1)}\le\dots\le M_{(n)}$. No padding is needed. Non-finite scores are rejected (a NaN maximum would silently disable the alarm).

**Step 2, rank.**

$$k(\alpha)=\big\lceil (n+1)(1-\alpha)\big\rceil .$$

The product is rounded to 9 decimals before the ceiling (`conformal_rank`): in floating point $150\times0.82=123.00000000000001$ and a plain ceiling returns 124. On the 0.005-step alpha grid the plain float ceiling was one rank too high in 289 of the (n, alpha) pairs with $n<400$ that I tested against exact arithmetic.

**Step 3, constant threshold.**

$$b_\alpha(t)=\begin{cases}M_{(k(\alpha))} & k(\alpha)\le n\\ +\infty & k(\alpha)>n\ \ (\alpha<1/(n+1))\end{cases}$$

The same value for every step. The detector alarms at $\tau=\min\{t: s(t)\ge b_\alpha\}$, i.e. as soon as an episode's running maximum reaches $b_\alpha$.

### What is guaranteed

For a new success exchangeable with the calibration successes, with continuous scores (no ties), its maximum $M_{\rm new}$ has a uniformly distributed rank among the $n+1$ values, so

$$\Pr\big[M_{\rm new}\ge M_{(k)}\big]=\frac{n+1-k}{n+1}\in\Big(\alpha-\frac1{n+1},\ \alpha\Big].$$

An alarm on a success happens exactly when its maximum reaches the threshold, so this is the false-alarm probability: at most $\alpha$, and conservative by less than $1/(n+1)$. The statement is **marginal over the calibration draw**. For one fixed calibration set, the realised false-alarm probability is random:

$$\Pr\big[\text{false alarm}\mid\text{calibration}\big]\ \sim\ \mathrm{Beta}\big(n+1-k,\ k\big).$$

Its mean is the value above, but it is far from constant for small $n$. For $n=26$, $\alpha=0.1$: $k=25$, $\mathrm{Beta}(2,25)$, mean $7.4\%$, standard deviation $4.9$ points, central 90% interval $1.4\%$ to $17.0\%$. For $n=76$ (typical of LIBERO-X), $\alpha=0.1$: $k=70$, $\mathrm{Beta}(7,70)$, mean $9.1\%$, standard deviation $3.3$ points, interval $4.4\%$ to $15.0\%$. A single LIBERO-10 run at nominal 10% can therefore land anywhere in roughly 1-17% before any evaluation noise from its 20-27 held-out successes.

### Monotone in $\alpha$ (unlike the band)

$k(\alpha)$ is non-increasing in $\alpha$ and $M_{(k)}$ is non-decreasing in $k$, so $b_\alpha$ is non-increasing in $\alpha$. A lower constant threshold alarms on a superset of episodes, so on any fixed set of episodes the FPR and the TPR early are non-decreasing in $\alpha$, exactly. An early stop at the first FPR $\ge40$ would therefore be safe for max-conformal (the package runs the full grid anyway), and the curve needs no running-maximum envelope (one is still applied by `area_to_fpr` and changes nothing).

### Limits
- **Exchangeability** holds between the calibration half and the held-out half of the seen tasks (stratified halves), not for unseen tasks. Only the false-alarm side is controlled; there is no guarantee on the TPR.
- **Ties.** With ties at the threshold the rule "$\ge$" can exceed $\alpha$ (the rule "$>$" would stay valid): if every calibration maximum saturates at the same value, every saturating new success alarms. On the saved scores no run had any tied calibration maxima (0 of 56 LIBERO-X and 0 of 12 LIBERO-10 runs), so the two rules give identical numbers; a test documents the limitation.
- $\alpha<1/(n+1)$ gives no alarm: with 20-26 calibration successes (LIBERO-10) that is $\alpha<0.036$-$0.048$.

### Measured on the saved scores (`checks/check_max_conformal_on_real_scores.py`)
Mean over runs of the FPR on the held-out seen half (the half exchangeable with the calibration half) against $\alpha$; theory is $(n+1-k)/(n+1)$ ($n=76$: 3.9, 9.1, 19.5, 29.9%).

| | $\alpha=0.05$ | 0.10 | 0.20 | 0.30 |
|---|--:|--:|--:|--:|
| theory, $n=76$ | 3.9 | 9.1 | 19.5 | 29.9 |
| LIBERO-X, four models (14 runs each; 66-90 calibration and 66-91 held-out successes; SE 0.6-2.3) | 3.7-4.6 | 9.2-10.0 | 16.7-19.6 | 26.9-28.7 |
| LIBERO-10, SAFE-MLP / SAFE-LSTM, pi0.5 and GR00T (3 runs each; 20-26 calibration and 20-27 held-out successes; SE 0-5.7) | 0.0-8.0 | 2.6-12.9 | 20.6-23.8 | 25.0-34.8 |

LIBERO-X: at $\alpha=0.05$ and $0.1$ all eight cells are within one standard error of theory. At $\alpha=0.2$ and $0.3$ seven of the eight are below theory (the furthest, SAFE-LSTM at $\alpha=0.2$, 16.7 against 19.5 with SE 1.5, is about 1.9 standard errors low; the other cell is 0.1 above), so there is a consistent lean toward conservative of up to about 2-3 points, not noise around theory. The four models share the same 14 splits, so they are not four independent confirmations. LIBERO-10: 3 runs of 20-27 held-out successes each cannot separate $\alpha$ from the run-to-run spread above; all cells lie within about 2 standard errors of theory. The FPR on the unseen tasks was non-decreasing along the whole 199-point alpha grid in every run, as proved.

### Code map

| formula | code |
|---|---|
| $M_i$, sorting, finiteness check | `MaxConformal.fit` |
| $k(\alpha)$ | `conformal_rank` in `ranks.py`, `MaxConformal.rank` |
| $b_\alpha(t)$ | `MaxConformal.threshold` |

### Tests (`test_max_conformal.py` 12, `test_ranks.py` 3)
$k$-th order statistic on an unsorted example and never an interpolation; $+\infty$ when $k>n$; constant over the requested length; per-episode maximum (not the final score); false-alarm rate averaged over 400 calibration sets of 50 within $[\alpha-0.02,\alpha+0.005]$ at four levels; threshold non-increasing and FPR non-decreasing over the 199-point grid; $k$ equals the exact rank where the float ceiling is off by one ($n=149$, $\alpha=0.18$); exact-arithmetic agreement for every $n<400$ on the whole grid; saturated ties exceed $\alpha$ (documented); NaN or infinite scores and bad $\alpha$ raise.

### Decided
1. **Alarm rule at the threshold: keep "$\ge$"** (the same convention as the band and LTT; no ties observed).

## 3. Learn-then-Test (`ltt.py`)

Unlike the other two, LTT needs calibration *failures* (with onsets) as well as successes, returns one constant threshold $\tau^\star$ (not a curve in $\alpha$), and certifies two targets at once with confidence $1-\delta$: a TPR early of at least $1-\alpha_{\rm e}$ and an FPR of at most $\alpha_{\rm f}$. It is reported as a separate evaluation, not swept.

### Steps

**Grid.** $\tau_j=\operatorname{sigmoid}(z_j)$, $z_j$ equally spaced on $[-9,9]$, $j=1,\dots,181$, fixed in advance (data-independent, as the method requires). The scores must be probabilities in $(0,1)$; this excludes SAFE-MLP, whose scores are cumulative sums.

**Step 1, first crossing.** For an episode with scores $s$:

$$c(\tau)=\min\{t:\ \max_{u\le t}s(u)\ge\tau\}\quad(=T\text{ if never}),$$

the same step as the evaluation's `alarm_step` with threshold $\tau$ (a test checks this).

**Step 2, violation counts** on the $n_{\rm f}$ calibration failures (onsets $o_i$) and $n_{\rm s}$ calibration successes:

$$K_{\rm e}(\tau)=\#\{i\in\text{failures}:\ c_i(\tau)>o_i\},\qquad K_{\rm f}(\tau)=\#\{i\in\text{successes}:\ c_i(\tau)<T_i\}.$$

$K_{\rm e}$ counts failures not flagged at or before their onset; $K_{\rm f}$ counts successes with any alarm.

**Step 3, p-value per threshold.** Test $H_\tau$: "$\tau$ violates a target", i.e. $R_{\rm e}(\tau)>\alpha_{\rm e}$ or $R_{\rm f}(\tau)>\alpha_{\rm f}$, where $R_{\rm e},R_{\rm f}$ are the true miss and false-alarm probabilities:

$$p_\tau=\max\Big\{\Pr\big[\mathrm{Bin}(n_{\rm f},\alpha_{\rm e})\le K_{\rm e}(\tau)\big],\ \Pr\big[\mathrm{Bin}(n_{\rm s},\alpha_{\rm f})\le K_{\rm f}(\tau)\big]\Big\}.$$

Each binomial tail is a valid p-value for its own null; the maximum is a valid p-value for the union of the two nulls (intersection-union test).

**Step 4, fixed-sequence testing with $S=5$ starts.** Starting indices $j_1<\dots<j_S$ are the rounded, equally spaced points of $\{1,\dots,181\}$ (indices 1, 46, 91, 136, 181). From each start, walk to higher $\tau$ and accept $\tau_j$ while $p_{\tau_j}\le\delta/S$; stop that walk at its first failure. The accepted set $\mathcal A$ is the union over the walks.

**Step 5, selection** among $\mathcal A$: highest calibration TPR early; ties by lowest calibration FPR, then lowest $\tau$. If $\mathcal A$ is empty the run is *not certified*.

### What is guaranteed

Each walk is a fixed-sequence test at level $\delta/S$, so the probability that it accepts any truly invalid threshold is at most $\delta/S$ (to accept an invalid threshold the walk must reject, in particular, the first true null on its path, which has probability at most $\delta/S$). A union bound over the $S$ walks gives

$$\Pr\Big[\forall\tau\in\mathcal A:\ R_{\rm e}(\tau)\le\alpha_{\rm e}\ \text{and}\ R_{\rm f}(\tau)\le\alpha_{\rm f}\Big]\ge1-\delta .$$

The statement is unconditional over calibration draws, so choosing $\tau^\star$ inside $\mathcal A$ with the calibration data is allowed. It is *not* a statement conditional on certification: when certification is rare, the rate of invalid selections among certified draws can exceed $\delta$ (simulation below: up to 17% among 6 certified draws, 13% among 61), while the unconditional rate stays under $\delta$.

Limits: the calibration and test episodes must be exchangeable, so the guarantee is for seen tasks and fails on unseen tasks (below); "early" is relative to Gemini onset labels; episodes of one task are treated as independent.

### Measured

Scripts: `checks/check_ltt_on_real_scores.py` (saved scores of earlier models) and `checks/check_ltt_validity_simulation.py` (known true risks).

**Sample sizes bound what can be certified.** A single binomial test with zero violations reaches $\alpha_{\min}=1-L^{1/n}$ at level $L$ (for the five starts $L=\delta/5=0.02$):

| | calibration failures | calibration successes | smallest $\alpha_{\rm e}$ | smallest $\alpha_{\rm f}$ |
|---|--:|--:|--:|--:|
| LIBERO-X (14 runs) | 92-120 | 66-90 | 0.03-0.04 | 0.04-0.06 |
| LIBERO-10 pi0.5 (3 seeds) | 9 | 26 | 0.35 | 0.14 |
| LIBERO-10 GR00T (3 seeds) | 12-15 | 20-24 | 0.23-0.28 | 0.15-0.18 |

So on LIBERO-10 LTT can only certify weak targets (TPR early of at most 65-77%, FPR of at least 14-18%), however good the detector is.

**The five-start walk often fails to certify when valid thresholds exist.** The risks are monotone in $\tau$ (a higher $\tau$ alarms on fewer episodes: $K_{\rm f}$ falls and $K_{\rm e}$ rises), so the set where both p-values are small is an interval; a walk accepts only the part of it to the right of a start that lies inside it, and the five starts are 45 grid points (4.5 logit units) apart. LIBERO-X, certified runs of 14 at $(\alpha_{\rm e},\alpha_{\rm f})$, five starts | two-walk prototype:

| model | (0.7, 0.2) | (0.7, 0.3) | (0.6, 0.3) | (0.5, 0.3) |
|---|--:|--:|--:|--:|
| SAFE-LSTM | 0 \| 13 | 0 \| 14 | 0 \| 13 | 0 \| 11 |
| LSTM onset | 0 \| 9 | 0 \| 14 | 0 \| 12 | 0 \| 4 |
| LSTM onset_pre | 8 \| 14 | 14 \| 14 | 11 \| 14 | 8 \| 14 |
| LSTM onset_traj | 4 \| 11 | 11 \| 14 | 5 \| 14 | 3 \| 10 |

**The five-start procedure's selected threshold is one of its start points.** Because the acceptable region is an interval, a walk accepts from its start to the upper end of the interval, and the selection (highest TPR early) takes the lowest accepted $\tau$, i.e. the lowest start inside the interval. The starts are grid indices 0, 45, 90, 135, 180. Measured: for LSTM onset_pre the selected $\tau$ is index 135 ($\tau=\operatorname{sigmoid}(4.5)=0.989$) in every certified run, at all four target pairs tried (8, 14, 11 and 8 runs); for onset_traj it is 135, 136 or 180 (the last grid point, 0.99988); SAFE-LSTM and LSTM onset never certify. So the five-start "LTT threshold" is a fixed grid value with a certificate attached, not a calibrated operating point, which is also why its held-out FPR sits far below the target (8.7% at $\alpha_{\rm f}=0.2$).

The two-walk prototype is not in the package: FPR hypotheses are walked from the highest $\tau$ downwards and TPR-early hypotheses from the lowest upwards, each fixed-sequence test at level $\delta/2$, and the accepted set is the intersection. It is valid for the same reason (a fixed-sequence test is valid for any fixed order; a union bound over the two risks), and monotonicity only makes it powerful. Its selected thresholds spread over the grid (for onset_pre at $(0.7,0.3)$: indices 103 to 135, 12 distinct values in 14 runs).

**Both procedures are valid in simulation** (600 calibration draws per row; synthetic episodes with a large reference sample for the true risks; limit $\delta=0.10$):

| failures = successes, $(\alpha_{\rm e},\alpha_{\rm f})$ | invalid $\tau$ accepted, five starts \| two walks | certified, five starts \| two walks |
|---|--:|--:|
| 60, (0.6, 0.2) | 0.8% \| 5.0% | 23.2% \| 58.7% |
| 60, (0.5, 0.3) | 0.8% \| 4.0% | 1.0% \| 16.8% |
| 120, (0.6, 0.2) | 0.7% \| 5.0% | 53.8% \| 87.3% |
| 120, (0.5, 0.15) | 1.5% \| 4.5% | 1.7% \| 10.2% |
| 30, (0.7, 0.3) | 1.5% \| 5.8% | 52.2% \| 81.5% |

The five-start procedure is very conservative (about 1% against the 10% allowed).

**Held-out behaviour on the real scores, both procedures.** LIBERO-X, 14 runs per model; per cell: certified runs, then mean TPR early / FPR (%) of the selected threshold over the certified runs on the held-out seen half (in brackets: runs in which both targets hold there) and on the unseen tasks.

$(\alpha_{\rm e},\alpha_{\rm f})=(0.7,\,0.2)$:

| model | five starts | two walks |
|---|---|---|
| SAFE-LSTM | 0/14 | 13/14; seen 54.6 / 12.4 (12/13); unseen 44.4 / 32.0 |
| LSTM onset | 0/14 | 9/14; seen 46.1 / 14.2 (8/9); unseen 28.5 / 22.7 |
| LSTM onset_pre | 8/14; seen 51.1 / 8.7 (8/8); unseen 37.9 / 17.3 | 14/14; seen 64.7 / 13.3 (13/14); unseen 49.3 / 25.2 |
| LSTM onset_traj | 4/14; seen 46.3 / 8.2 (4/4); unseen 43.2 / 42.7 | 11/14; seen 54.3 / 12.6 (11/11); unseen 42.5 / 35.2 |

$(\alpha_{\rm e},\alpha_{\rm f})=(0.7,\,0.3)$:

| model | five starts | two walks |
|---|---|---|
| SAFE-LSTM | 0/14 | 14/14; seen 68.3 / 18.3 (13/14); unseen 58.1 / 47.3 |
| LSTM onset | 0/14 | 14/14; seen 54.1 / 19.4 (14/14); unseen 40.5 / 32.5 |
| LSTM onset_pre | 14/14; seen 62.2 / 13.4 (14/14); unseen 47.9 / 24.8 | 14/14; seen 77.1 / 20.4 (13/14); unseen 66.5 / 42.2 |
| LSTM onset_traj | 11/14; seen 52.1 / 12.8 (11/11); unseen 42.0 / 36.3 | 14/14; seen 66.6 / 19.7 (13/14); unseen 57.0 / 52.7 |

Reading: the two-walk thresholds are less conservative (higher TPR early, FPR nearer the target) and hold the targets on the seen half in 8 of 9 to 14 of 14 runs per cell, i.e. one miss in at most 14 runs, consistent with $\delta=0.10$; the "hold" count also includes the sampling noise of an empirical rate on 66-91 held-out successes, so it overstates true violations. On unseen tasks there is no guarantee and the FPR exceeds the target in most cells (for example onset_pre two-walk 42.2% against 30%), so an LTT threshold cannot be advertised as controlling the false-alarm rate under task shift.

### Code map

| formula | code |
|---|---|
| grid, first crossing $c(\tau)$ | `TAUS`, `first_crossing` |
| $K_{\rm e},K_{\rm f}$ | `risk_counts` (returns `RiskCounts`) |
| $p_\tau$ | `joint_p_values` |
| walks, accepted set | `fixed_sequence_test` |
| $\tau^\star$ | `select_threshold` |
| all of it | `calibrate_ltt` |

### Tests (`test_ltt.py`, 21)
Grid; first crossing with running max and `T` when never reached; first crossing equals `alarm_step`; hand-computed risk counts; joint p-value is the larger tail; fixed-sequence walk stops at the first failure and multi-start can accept after a failure; separable problem certified with $\tau$ between the classes; non-rising and late failures not certified, an alarm exactly at the onset counts as early; the FPR target and the TPR-early target each can block certification; looser targets never hurt; selection prefers TPR early, then FPR, then lowest $\tau$; agreement with an independent plain-loop implementation on 40 random problems; family-wise error below $\delta$ in a simulation with known true risks; input validation.

### Decided
1. **Procedure: keep the five-start walk.** Known consequence, measured above: its selected threshold is always one of the start points (grid index 135 in every certified `onset_pre` run), and SAFE-LSTM and `LSTM onset` never certify at the targets tried. Any LTT result must be reported with this caveat and with the number of certified runs. The two-walk prototype stays in `checks/` only.
2. **Targets $\alpha_{\rm e}$, $\alpha_{\rm f}$, $\delta$: to be reviewed after the final data and models exist.** They have no defaults.
3. **Selection rule: highest TPR early** (as implemented).
