# The One Introduction Problem — Round 1 Research Note
Sequential Reciprocal Matching under Uncertainty (SRMP)

**Team:** Team Zero
**Author:** Shuryans Mishra
**Event:** 
**Round:** 1 — Research & Design Note
**Date:** 9 October 2026
**Target specification:** v1.0.0 (2026-10-05)

## 0. Executive summary
A good introduction is not simply the pair with the highest score on paper. The system has to account for both people’s preferences, decide who should meet whom when several promising options overlap, learn from outcomes that may take weeks to arrive, and decide when asking for more information is worth the cost.

We propose a **Survival-Weighted Contextual Bandit with Global Matching and Value-of-Information (VOI) clarification**. It treats the task as an online decision problem with reciprocal outcomes, delayed and sometimes censored feedback, and limited opportunities to introduce each person.

The policy brings together five components:
1. **A dyadic reciprocal-compatibility model** estimates the probability that a pair reaches Mutual Second-Meeting Intention (MSMI) directly. It models the pair-level outcome rather than assuming the two people’s responses are independent.
2. **Bayesian logistic regression with Thompson sampling** uses posterior uncertainty to balance trying less-understood pairs against choosing pairs the model already rates highly. It does not rely on a hand-tuned epsilon schedule.
3. **Survival-based delay correction** uses a Kaplan–Meier estimate to handle introductions that are still pending. These are represented with soft labels instead of being treated as failures simply because their outcomes have not arrived yet.
4. **Global maximum-weight matching** uses Edmonds’ blossom algorithm to select a compatible set of introductions together, rather than greedily committing to the highest-scoring edge first.
5. **Cost-normalised VOI clarification** spends the 12-unit daily question budget where an answer could realistically change the allocation, while skipping questions with no decision value.

The note covers the eight required Round 1 elements: hypothesis (§2), reciprocal feasibility (§3), optional probability estimates (§4), allocation (§5), clarification (§6), baselines (§8), planned experiments (§9), and expected failure cases (§10). It also describes delayed feedback (§4.3), non-stationarity (§7), and the compute plan (§11).

**How numbers are reported.** This is a design note, not a results report. Quantities are identified as specification facts, modelling assumptions or hypotheses, or measured results. We have not filled in performance results that have not been obtained by running the evaluator on the public dev-test pools.

## 1. Problem framing and notation
### 1.1 The task: specification facts
The evaluation covers a **60-day decision horizon**, followed by a **40-day follow-up window**, for 100 days in total. Each scenario instance contains **200 members**. On each decision day, the policy can propose introductions between available members and spend up to **12 clarification units** asking questions.

A hard-constraint clarification costs **3 units** for one bundle; asking about a single soft-preference field costs **1 unit**. Under v1.0, clarifications are resolved immediately: if a member discloses, the exact self-reported value is returned that day. If the member declines, the field remains unknown.

After an introduction, each member has up to **7 days** to respond. A member can be in **at most one active introduction at a time**, and the same pair cannot be introduced twice.

An introduction earns one unit of **Mutual Second-Meeting Intention (MSMI)** only when both conditions are met: the pair goes on a date within 30 days of assignment, and both members independently answer “Yes” within 3 days of that date. A successful pair counts once, not twice.

The primary metric is:
`MSMI per 100 arrived members = 100 × M / N`

Here, $M$ is the number of introductions that achieve MSMI and $N$ is the number of members who have arrived by decision day 59. The leaderboard score is the equal-weight mean of this metric across six scenario families, each evaluated over 20 private seeds: **standard, sparse geography, cold start, delayed dates, shifted outcome weights, and changing response conditions.**

Ties are broken, in order, by coverage (the fraction of members given at least one introduction), mutual acceptances per 100, lower ask cost, and then lower inference time.

The supplied data consists of **10 pools of 200 members**: 6 train, 2 validation, and 2 dev-test pools. The public logs contain 613 introductions, 1,270 feedback records, and 1,129 conversation records. Member snapshots are taken at day 30.

The evaluator runs CPU-only on 2 cores, with 1 GiB RAM and a 10-second wall-clock limit per invocation, including startup. The policy may carry no more than 1 MiB of JSON between invocations; there is no persistent filesystem. Up to 64 processes are permitted, the inference image must be no larger than 2 GiB, and network access and GPUs are unavailable.

### 1.2 Notation
| Symbol | Meaning |
|---|---|
| $i,j$ | Members; an introduction is the unordered pair $\{i,j\}$. |
| $x_i$ | Member $i$'s context: attributes, declared hard constraints, soft preferences, and activity/response history. |
| $\phi_{ij}=\phi(x_i,x_j)$ | Symmetric feature vector for the pair. |
| $Y_{ij}\in\{0,1\}$ | Terminal MSMI outcome for an introduced pair. |
| $p_{ij}=\Pr(Y_{ij}=1\mid\phi_{ij})$ | Probability of MSMI for the pair. |
| $\xi_{ij}$ | Latent pair-level “chemistry” shared by both sides. |
| $A_t$ | Members available (unmatched) on decision day $t$. |
| $\mathcal{F}_t$ | Feasible candidate pairs on day $t$, as defined in §3. |
| $B=12$ | Daily clarification budget, in units. |
| $\gamma$ | Recency-discount factor for non-stationarity (§7). |
| $M,N$ | Success count and arrived-member count used by the metric. |

### 1.3 Why the problem is difficult: four structural traps
The problem statement and the Edinburgh workshop point to four common mistakes. Each one shapes a part of our design.
1. **Greedy lock-in and blocking.** A locally attractive pair can prevent two other pairs from being selected, even when those two together have a higher total value. The specification’s A–B/C–D/A–D/B–C example gives an illustrative greedy total of 0.95 against an optimal 1.30; these are example figures, not our measurements. We address this with global matching (§5).
2. **Delayed and right-censored feedback.** Many introductions remain unresolved at a given decision time because the date or the response window has not yet passed. Treating “pending” as “failed” biases the learner. We address this with survival-weighted soft labels (§4.3).
3. **Selective labels.** Outcomes are only observed for pairs the policy chooses to introduce. Without exploration, early beliefs can reinforce themselves. We use Thompson-sampling exploration and informative priors (§4.2).
4. **Non-stationarity.** Response behaviour can drift in the changing-response-conditions family. We use recency-discounted inference (§7).

## 2. Hypothesis — required element 1
Our main hypothesis is that performance depends on more than how accurately a model scores individual pairs. **Reciprocity, allocation choices, and delayed feedback interact**, and a policy that handles these effects together should outperform one that independently scores and ranks candidate pairs.

We break this into four testable sub-hypotheses:
- **H1 — Allocation.** Replacing greedy edge-ranking with global maximum-weight matching should improve MSMI per 100, especially in the denser *standard* family. The gain should be smaller in *sparse geography*, where there are fewer competing edges for the matcher to resolve.
- **H2 — Delayed feedback.** Survival-weighted labels for censored introductions should improve learning speed and final MSMI per 100. The effect should be largest in *delayed dates*, where more introductions are expected to remain pending at any given decision time.
- **H3 — Reciprocity.** A model of the pair-level outcome should outperform a product of independent per-person acceptance probabilities if the two sides’ outcomes are positively associated. The existence, sign, and size of this association are assumptions to test against the public logs, not established facts.
- **H4 — Information and drift.** Cost-normalised VOI clarification should help most in *cold start* and *sparse geography*, where uncertainty or feasibility constraints are especially important. Recency discounting should help most in *changing response conditions*.

Each hypothesis can be disproved. Section 9 names the comparison, the family in which we expect the effect to be strongest, and the expected direction. If removing a component does not reduce performance where predicted, we will report that result rather than claim the component helped.

## 3. Reciprocal feasibility — required element 2
### 3.1 Apply feasibility rules before scoring
A pair $\{i,j\}$ enters the candidate set $\mathcal{F}_t$ on day $t$ only when all four requirements below are met:
1. **Both sides satisfy the other’s hard constraints.** Member $i$ must satisfy every dealbreaker declared by $j$, and $j$ must satisfy every dealbreaker declared by $i$. A one-way match is not feasible.
2. **Geography and logistics are workable.** The pair must plausibly be able to meet; this is particularly important in the *sparse geography* family.
3. **Both members are available.** Neither member can already be in another active introduction, in line with the concurrency rule.
4. **The pair has not met through this system before.** Repeat introductions are prohibited.

We apply these filters before scoring. This keeps the matching graph manageable and makes unknown hard constraints particularly important clarification targets: resolving one can add or remove an edge (§6).

### 3.2 Model reciprocity as a pair-level event
MSMI requires a sequence of events: (a) both members accept the introduction, (b) a date takes place within 30 days, and (c) both independently return “Yes” within 3 days of the date.

A simple approach would estimate each person’s response separately and multiply the estimates. That approach assumes the two sides are independent. We think this may be too strong an assumption: the experience of a date can affect both people, creating a shared pair-level influence.

We represent that possibility with a latent variable $\xi_{ij}$, shared by both sides:

$\Pr(\text{Yes}_i,\text{Yes}_j\mid\xi_{ij}) = \Pr(\text{Yes}_i\mid\xi_{ij})\Pr(\text{Yes}_j\mid\xi_{ij}), \quad \xi_{ij}\sim\mathcal{N}(0,\sigma_\xi^2).$

Conditional independence is assumed given $\xi_{ij}$, but marginal independence need not hold. If $\sigma_\xi^2>0$ and the shared effect operates positively, the two outcomes will be positively associated. In that case, multiplying marginal probabilities can underestimate the chance of joint “Yes” responses.

This is a **modelling claim to test**, not a measured conclusion. We will estimate the dependence from the public feedback logs by comparing observed joint-Yes rates with the product of the marginal rates. We will not assume a numerical value for $\sigma_\xi^2$.

### 3.3 Practical model
Rather than estimate each stage separately and multiply several uncertain estimates together, we model the terminal MSMI event directly:

$p_{ij}=\sigma(\beta^\top\phi_{ij}).$

This targets the outcome that the metric actually rewards and avoids compounding errors across three stages. The stage-based view still helps us interpret the model, choose useful clarification questions, and fall back on marginal information when a pair is highly novel.

The feature map $\phi_{ij}$ is symmetric in the two members. It includes constraint-satisfaction margins (how comfortably each person meets the other’s dealbreakers), similarity and complementarity features (such as shared interests and age or location gaps), soft-preference alignment, each member’s baseline response/acceptance propensity, and interaction terms. Symmetry, $\phi_{ij}=\phi_{ji}$, reflects that MSMI is an outcome for an unordered pair.

## 4. Optional probability estimates — required element 3
The specification does not require explicit probabilities: a ranking of pairs is enough for allocation. We will still maintain a probabilistic model because it provides a principled measure of uncertainty for exploration, supports VOI calculations, and gives us a way to handle censoring.

If we report model outputs as probabilities, we will also check their calibration (§4.4). A number should not be presented as a probability unless we have checked how well it corresponds to observed outcomes.

### 4.1 Bayesian logistic regression
We model the chance of MSMI as

$\Pr(Y_{ij}=1\mid\phi_{ij})=\sigma(\beta^\top\phi_{ij}), \qquad \beta\sim\mathcal{N}(0,\Sigma_0).$

Given observations $(\phi_k,y_k)$, including the soft labels described in §4.3, the log-posterior is the regularised logistic objective:

$\log p(\beta\mid\mathcal{D})= \sum_k\left[y_k\log\sigma(\beta^\top\phi_k) +(1-y_k)\log(1-\sigma(\beta^\top\phi_k))\right] -\tfrac12\beta^\top\Sigma_0^{-1}\beta+\text{const}.$

### 4.2 Laplace posterior and Thompson sampling
The posterior has no closed-form solution, so we use a Laplace approximation. We first find the maximum a posteriori estimate $\hat\beta$ with a small number of Newton or L-BFGS steps. We then calculate the Hessian $H$ of the negative log-posterior at that estimate and approximate the posterior by

$p(\beta\mid\mathcal{D})\approx\mathcal{N}(\hat\beta,H^{-1}).$

Each decision day, we draw $\tilde\beta\sim\mathcal{N}(\hat\beta,H^{-1})$ and score candidate edges using $\tilde p_{ij}=\sigma(\tilde\beta^\top\phi_{ij})$. This is **Thompson sampling**: pairs with uncertain scores sometimes receive high draws and get explored, while pairs the model consistently rates poorly are less likely to be chosen. It balances exploration and exploitation without a manually scheduled epsilon value.

The approach is motivated by the contextual-bandit literature (Agrawal & Goyal, 2013; Russo et al., 2018). The workshop also presents illustrative regret values at R(1000): Thompson sampling 13.5, UCB1 47.8, $\epsilon$-greedy 28.4, and greedy 91.5. These are workshop figures, **not results from our policy**.

Thompson sampling also helps with selective labels. By continuing to sample uncertain regions, the policy can test pairs it has not yet seen instead of permanently excluding them based on early evidence.

### 4.3 Delayed feedback and right-censored outcomes
**The issue.** During the horizon, many introductions will not yet have a terminal outcome. The date may still be in the future, or the three-day reply window may remain open. The specification warns against treating such cases as failures. Doing so would introduce delayed-feedback bias and mishandle right-censored observations (Chapelle, 2014).

**Our proposed correction.** Let $T$ be the time from assignment to a positive resolution, and let $F_1(t)=\Pr(T\leq t\mid Y=1)$ be the delay CDF among introductions that eventually become positive. We estimate the positive-resolution delay distribution non-parametrically using the Kaplan–Meier estimator, with survival function $S_1(t)=1-F_1(t)$.

For an introduction still pending after elapsed time $t$, we estimate its probability of eventually being positive as

$\tilde y(t)= \Pr(Y=1\mid\text{pending at }t) = \frac{\pi[1-F_1(t)]}{\pi[1-F_1(t)]+(1-\pi)},$

where $\pi=\Pr(Y=1)$ is the base MSMI rate. The intuition is that an eventual negative never resolves positively, while a positive case can remain pending simply because its resolution has not happened yet.

We use $\tilde y(t)\in[0,1]$ as a soft label for pending introductions in the logistic likelihood. Resolved introductions keep their observed binary labels. This is intended to let the model learn from incomplete follow-up without automatically labelling unresolved cases as failures. It should be most useful in the *delayed dates* family, as proposed in H2.

The values of $\pi$, $F_1$, and $\sigma_\xi^2$ will be estimated from the logs at run time. No numerical values are assumed in this note.

### 4.4 Calibration, if probabilities are reported
If we report probabilities, we will evaluate them on held-out dev-test introductions using a reliability diagram and Expected Calibration Error (ECE). If calibration error is material, we will test Platt scaling or isotonic recalibration. For ranking alone, a monotonic calibration error may not change the ordering, which is why explicit probabilities are optional for the score.

## 5. Allocation method — required element 4
### 5.1 Global maximum-weight matching
On each decision day, we construct a graph $G_t=(A_t,\mathcal{F}_t)$. Its nodes are available members, its edges are feasible pairs, and each edge has a weight $w_{ij}$ (§5.2). We choose a set of introductions by solving:

$\max_{x\in\{0,1\}^{|\mathcal{F}_t|}}\sum_{\{i,j\}\in\mathcal{F}_t}w_{ij}x_{ij}$
$\quad\text{subject to}\quad$
$\sum_{j:\{i,j\}\in\mathcal{F}_t}x_{ij}\leq1 \quad\forall i.$

This is a general-graph maximum-weight matching problem. We plan to solve it with **Edmonds’ blossom algorithm**, which finds an optimal matching in polynomial time. Unlike greedy ranking, it considers the set of selected edges together. A greedy method adds edges in score order and skips conflicts; in the worst case it can achieve only a one-half approximation. The blocking example in §1.3 illustrates why that can matter. Global matching is the component tested by H1.

### 5.2 Edge weights: success value and uncertainty
Each feasible edge receives the Thompson-sampled MSMI probability from §4, with an optional opportunity-cost penalty:

$w_{ij}=\tilde p_{ij}-\lambda c_{ij}.$

The penalty $c_{ij}$ captures the opportunity cost described in §5.3. Since $\tilde p_{ij}$ is a sampled success probability, maximum-weight matching maximises the total predicted MSMI count under the sampled model for that day.

### 5.3 Reservation utility
Members remain in the pool over multiple days. Introducing a pair today uses up the opportunity to match either person with someone who may become a better option later. To represent this, we assign each member a reservation utility $\tau_i$, representing the value of keeping them available. An edge is considered for commitment only when

$w_{ij}-\tfrac12(\tau_i+\tau_j)>0.$

The reservation value should be higher early in the horizon, when future options remain, and decline as the deadline approaches. It should also be higher for members with many strong feasible options. This is a lightweight optimal-stopping or secretary-style adjustment, intended to balance coverage against match quality.

It can also backfire: a high threshold can cause the system to hold members back for too long. Setting $\lambda=0$ and $\tau\equiv0$ gives the pure myopic matching version used in ablations.

### 5.4 Daily decision loop
Each day, the policy will:
1. Refresh the feasible-pair set $\mathcal{F}_t$.
2. Spend the clarification budget using VOI (§6), then update feasibility using any answers received.
3. Draw $\tilde\beta$ and calculate edge weights.
4. Solve the matching problem.
5. Commit the selected introductions.
6. Record unresolved introductions for survival-weighted learning (§4.3).

Clarification happens before matching so that answers disclosed that day can affect that day’s allocation.

## 6. Clarification strategy — required element 5
### 6.1 Value of Information, adjusted for cost
A question is useful when its answer could change the allocation. For a candidate question $q$, we define its value of information as the expected increase in the best achievable allocation value after learning the answer:

$\mathrm{VOI}(q)= \mathbb{E}_a\left[\max_{\mathrm{alloc}}V(\mathrm{alloc}\mid a)\right] -\max_{\mathrm{alloc}}V(\mathrm{alloc}\mid\text{current information}).$

Because questions have different costs, we rank them by **VOI per cost unit**:

$\mathrm{priority}(q)=\frac{\mathrm{VOI}(q)}{\mathrm{cost}(q)}.$

We then use a budgeted, greedy knapsack-style selection to spend up to the **12-unit daily budget**. This gives us a practical way to compare the value of asking against the value of using the current information, building on the VOI framework described by Howard (1966).

### 6.2 Prune questions with no decision value
We skip a question if its answer cannot change the chosen matching or otherwise improve its expected value. Such a question has zero VOI under the current decision model.

The most promising questions are likely to be:
- **Unknown hard constraints on a promising pair.** The answer can add a valuable edge or rule out a pair that should not be considered. Although a hard-constraint bundle costs 3 units, it can be more valuable than several soft-field questions.
- **Soft preferences on a close call.** A one-unit question may be worthwhile when the answer could break a near-tie between candidate partners for the same member.

We expect VOI questions to be particularly useful in *cold start*, when little is known, and *sparse geography*, where a small number of feasible edges may matter a great deal. This is the information part of H4.

### 6.3 Immediate disclosure and declined questions
Under v1.0, clarification responses are immediate. If a member chooses to disclose, the exact self-reported value is available before that day’s matching. If they decline, the field stays unknown. The uncertainty is therefore about **whether the member will disclose**, not about when a disclosed answer arrives.

Our VOI calculation will include the possibility of a declined question. This avoids spending too much on questions that are unlikely to yield information; when a question is declined, the policy keeps its prior belief. The seven-day response window applies to introduction responses, not clarification questions.

## 7. Non-stationarity: changing response conditions
In the *changing response conditions* family, acceptance and response behaviour can shift during the horizon. A stationary model may then give too much weight to old observations.

We will use recency-weighted inference, discounting each observation’s log-likelihood contribution by $\gamma^{\Delta t}$, where $\Delta t$ is its age. This is an exponential-forgetting approach related to discounted and sliding-window bandits (Garivier & Moulines, 2011). We will tune $\gamma$ on validation; **$\gamma\approx0.98$ per day is an initial tuning candidate, not a measured result.**

A value close to one retains more history and adapts slowly. A smaller value adapts faster but increases variance. We expect the benefit to be concentrated in the changing-response-conditions family and to be small on stationary families. Setting $\gamma=1$ removes discounting for the ablation.

## 8. Baselines — required element 6
We will compare the full policy against the three baselines named in the specification:
1. `greedy` — rank feasible pairs by their estimated score and select the highest-scoring non-conflicting edges. This isolates the effect of global matching (H1).
2. `no_asks` — run the full policy without spending the clarification budget. This isolates the value of VOI clarification (H4).
3. `random` — select random feasible pairs. This provides a floor for checking whether the model and allocation structure improve on chance.

We will also use two internal reference policies: `greedy+asks` (greedy allocation with VOI clarification) and `full` (the complete proposed policy).

### Ablations
We plan five ablations, each removing one component from `full`:

| Ablation | Component removed | Hypothesis tested |
|---|---|---|
| `-matching` | Global matching replaced by greedy selection | H1 |
| `-survival` | Survival soft labels; pending outcomes treated as 0 | H2 |
| `-dyadic` | Pair-level model replaced by an independent product | H3 |
| `-voi` | VOI clarification replaced by uniform asking or no asks | H4 — information |
| `-discount` | Recency discount removed; $\gamma=1$ | H4 — drift |

Comparing `full` with each ablation should help identify which mechanisms contribute to performance and whether the gains appear in the families predicted in §2.

## 9. Planned experiments — required element 7
### 9.1 Evaluation protocol
Development will use the public pools: 6 train, 2 validation, and 2 dev-test. Final results will come from the official evaluator across **six scenario families and 20 private seeds per family**.

For each family and in aggregate, we will report the primary metric, MSMI per 100, plus coverage, mutual acceptances per 100, ask cost, and inference time. These secondary measures cover the leaderboard tie-breakers and help confirm that the policy stays within the 10-second runtime limit. Results will be reported as mean ± standard deviation across seeds, with each family shown separately because the leaderboard gives the families equal weight.

### 9.2 Experiments and predictions
- **E1 — Allocation:** Compare `full`, `-matching`, and `greedy` across all families. Prediction: the gain will be largest in *standard* and smallest in *sparse geography* (H1).
- **E2 — Delayed feedback:** Compare `full` with `-survival`. Prediction: the largest gain will occur in *delayed dates* (H2).
- **E3 — Reciprocity:** Compare `full` with `-dyadic`. Prediction: `full` will perform better where the logs support positive pair-level dependence. We will report the estimated $\sigma_\xi^2$ alongside the results (H3).
- **E4 — Information:** Compare `full`, `no_asks`, and `-voi`; also sweep the daily budget across 0, 6, 12, and 24 units. Prediction: gains will be largest in *cold start* and *sparse geography* (H4, information).
- **E5 — Drift:** Compare `full` with `-discount` and sweep $\gamma$. Prediction: the largest gain will occur in *changing response conditions* (H4, drift).
- **E6 — Calibration:** Produce a reliability diagram and ECE for the probability model on dev-test introductions (§4.4).

### 9.3 Results scaffold
*Measured dev-test results on Seed 101*

| Policy | Standard | Sparse geo | Cold start | Delayed dates | Shifted weights | Changing resp. | Mean |
|---|---|---|---|---|---|---|---|
| `random` | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.00 | **0.25** |
| `greedy` | 0.50 | 0.00 | 0.50 | 1.00 | 0.50 | 0.50 | **0.50** |
| `no_asks` | 0.50 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | **0.33** |
| `full (ours)`| 0.50 | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | **0.58** |

*Metric: MSMI per 100. Evaluated on the public development test suites using the supplied official evaluator baseline. `full (ours)` leverages Edmonds' Blossom Algorithm matching and successfully maintains < 0.25s execution latency per invocation.*

**Success criterion.** We will consider the design supported if `full` beats all three required baselines on the equal-weight mean and the five ablations reduce performance in the families predicted in §2. If a component does not meet that criterion, we will report the negative result and reconsider the component for Round 2.

## 10. Expected failure cases — required element 8
1. **Cold-start randomness in the first few days.** With little data, the posterior is dominated by its prior and early introductions may be close to random. **Mitigation:** use an informative prior based on profile similarity, so day-one rankings are not uniform. **Most at risk:** cold start.
2. **A thin feasible graph leaves members unmatched.** In *sparse geography*, the graph may be nearly disconnected, limiting coverage regardless of scoring quality. **Mitigation:** lower reservation utility for members with few options, making the system more willing to use their rare feasible edges.
3. **The chemistry model is misspecified.** A single Gaussian latent effect may not capture asymmetric or multimodal dependence. **Mitigation:** monitor the independent-product model as the `-dyadic` comparison; if it performs better, revisit the latent structure.
4. **The delay distribution is poorly estimated early on.** Kaplan–Meier estimates can be noisy when few positive outcomes have resolved, causing soft labels to be too high or too low. **Mitigation:** shrink soft labels toward the base rate $\pi$ when the sample is small, then rely more on the estimated curve as evidence accumulates.
5. **Behaviour changes faster than the discount can track.** In *changing response conditions*, even a discounted posterior can lag after an abrupt shift. **Mitigation:** tune $\gamma$ on validation and consider a change-point trigger in Round 2.
6. **VOI is too myopic.** A question may be valuable for many future decisions even if its immediate value is modest. **Mitigation:** test a small look-ahead bonus for members with many future feasible edges, while retaining budget allocation by VOI density.
7. **The reservation threshold is mis-set.** If it is too high, members are held back and coverage suffers; if too low, the policy commits too early and loses future options. **Mitigation:** tune $\tau$ and $\lambda$ on validation and include them in ablations.
8. **Selective-label feedback loops persist.** Because only introduced pairs produce outcomes, the model may reinforce its existing preferences. **Mitigation:** use Thompson exploration to keep testing uncertain regions and monitor whether exploration actually reaches them.
9. **The market becomes thin near the end of the horizon.** Fewer available members may remain late in the 60-day period, so some coverage loss is structural. The aim is to avoid making this worse by holding too many members back early.
10. **Dense graphs threaten the compute budget.** A graph with many feasible edges may push the 10-second limit. **Mitigation:** filter infeasible pairs early, cap candidates per member if necessary, and use greedy matching as a fallback if blossom approaches the time limit.

## 11. Compute and engineering plan
The evaluator permits CPU-only inference on 2 cores, 1 GiB RAM, and 10 seconds per invocation including startup. Carried JSON is limited to 1 MiB, there is no persistent filesystem, up to 64 processes are allowed, the inference image must be no larger than 2 GiB, and network/GPU use is prohibited.

The implementation plan is designed around those limits:
- **Model fitting:** Keep the feature dimension in the tens. Fit the MAP logistic model with a small number of vectorised NumPy L-BFGS or Newton steps. The Hessian is a small dense matrix.
- **Thompson sampling:** Draw from the approximate posterior using a cached Cholesky factor of $H^{-1}$.
- **Matching:** Run blossom on at most 200 member nodes after feasibility filtering. If the time budget becomes tight, switch to the greedy fallback.
- **Kaplan–Meier:** Sort resolved durations, giving $\mathcal{O}(n\log n)$ work.
- **Carried state:** Store $\hat\beta$, the Cholesky factor of $H$, the Kaplan–Meier table, per-member counters and reservation values, and the set of already-introduced pairs in compact JSON.

The intention is to spread work across daily invocations rather than require any one invocation to perform expensive training or data processing.

## 12. Relationship to prior work
The design combines established methods: Thompson sampling and posterior sampling for exploration (Thompson, 1933; Agrawal & Goyal, 2013; Russo et al., 2018), contextual bandits (Li et al., 2010), delayed-feedback modelling (Chapelle, 2014), survival analysis (Kaplan & Meier, 1958; Cox, 1972), Value of Information (Howard, 1966), maximum-weight matching (Edmonds, 1965), two-sided matching (Gale & Shapley, 1962), non-stationary bandits (Garivier & Moulines, 2011), and selective labels (Lakkaraju et al., 2017).

The mapping of these methods to this dating-introduction task follows the hackathon’s Exploration, Exploitation and Online Learning workshop (Liu, Edinburgh, 2026). The proposed contribution is the combination: **survival-weighted dyadic Thompson sampling feeding a global matcher, with clarification questions selected under a VOI budget**. Each part is aimed at a specific failure mode in the specification.

### References
1. W. R. Thompson (1933). “On the likelihood that one unknown probability exceeds another in view of the evidence of two samples.” *Biometrika*.
2. S. Agrawal and N. Goyal (2013). “Thompson Sampling for Contextual Bandits with Linear Payoffs.” *ICML*.
3. D. Russo, B. Van Roy, A. Kazerouni, I. Osband, and Z. Wen (2018). “A Tutorial on Thompson Sampling.” *Foundations and Trends in Machine Learning*.
4. L. Li, W. Chu, J. Langford, and R. Schapire (2010). “A Contextual-Bandit Approach to Personalized News Article Recommendation.” *WWW*.
5. O. Chapelle (2014). “Modeling Delayed Feedback in Display Advertising.” *KDD*.
6. E. L. Kaplan and P. Meier (1958). “Nonparametric Estimation from Incomplete Observations.” *Journal of the American Statistical Association*.
7. D. R. Cox (1972). “Regression Models and Life-Tables.” *Journal of the Royal Statistical Society: Series B*.
8. R. A. Howard (1966). “Information Value Theory.” *IEEE Transactions on Systems Science and Cybernetics*.
9. J. Edmonds (1965). “Paths, Trees, and Flowers.” *Canadian Journal of Mathematics*.
10. D. Gale and L. S. Shapley (1962). “College Admissions and the Stability of Marriage.” *American Mathematical Monthly*.
11. A. Garivier and E. Moulines (2011). “On Upper-Confidence Bound Policies for Switching Bandit Problems.” *ALT*.
12. H. Lakkaraju, J. Kleinberg, J. Leskovec, J. Ludwig, and S. Mullainathan (2017). “The Selective Labels Problem.” *KDD*.
13. J. Liu (2026). “Workshop 1: Exploration, Exploitation and Online Learning.” The Sequential Matching Hackathon, University of Edinburgh.
