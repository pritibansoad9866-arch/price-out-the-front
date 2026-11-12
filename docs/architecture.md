# Architecture

**Game theory is the architecture, statistics are the mechanics, technology is the enabler.**

## Players

| Player | Who | Moves |
|---|---|---|
| PayU, the leader | commits one policy for everyone | envelope, verified programme, graded ladder, fleet linking, threshold rotation, vesting |
| Legitimate business | does what it declared; auxiliary or adjacent lines count | stays; may opt in to the programme |
| Front | real documents; the money is from an undeclared trade (betting, mule pass-through) | big fronts, fleets, probing, fake trust, leave |

The win: a front's cost per Rs through PayU exceeds its outside option (`c_alt` ~ 2-3%, win line 1.5 x `c_alt` ~ 4.5%),
normal businesses' friction stays ~0, and PayU stays profitable after expected losses.

## Code map

```
src/potf/
  config.py          ladder rungs, error budgets, economics, envelope, programme
  nodes.py           node specs (YAML in nodes/): one core, business types plug in
  panel.py           merchants x days x features, the shape every source becomes
  simulate.py        calibrated population: honest (with growth / adjacent lines), fronts, a syndicate fleet
  stats/
    peers.py         hierarchical peer baselines (geo x industry x scale), noise model, recursive residuals
    flag.py          1 Anything different?   F-test-esque first flag
    mmd.py           2 What moved?           kernel MMD per channel
    explain.py       3 Which explanation?    Bayes factors: growth / adjacent vs betting / mule
    fleet.py         4 Is it a fleet?        payer-overlap graph, lambda(N) = lambda0 (1 + N / N0)
    evalue.py        running e-values; benign directions projected out; Ville bound
  policy/
    ladder.py        5 Which rung?           evidence x inclusion rule >= 1/alpha, cost ratio, human for heavy rungs
    envelope.py      volume envelope per business type and scale
    programme.py     opt-in verified programme; slow vesting; no-exploit G(tau) < F(tau)
    economics.py     front cost per Rs: c = (A + k)/(V T) + h rho + o + envelope friction
  game.py            repeated game: PayU leads, syndicate best-responds, PayU retunes
  pipeline.py        end to end with rollout modes (shadow, light, full) and human review
  loaders/           real datasets -> the same shapes
  cli.py             `potf` command
app/streamlit_app.py dashboard
```

## Why the error budgets hold

For an honest merchant, each day's deviation from peers is standardised by a fitted noise model
and turned into a one-step-ahead residual against its own running level. These are independent
N(0, 1) when nothing changes. Before testing, the residual is projected onto the directions no
benign explanation (growth, decline, an adjacent line) can produce, so a benign change of any
shape adds no evidence. The running product of likelihood ratios is then a test martingale and
by Ville's inequality

    P(E_t ever >= 1/alpha) <= alpha,   alpha = budget / 1e6.

The inclusion rule multiplies the evidence by the likelihood ratio of (not) joining the programme.
Its mean over honest merchants is exactly 1, so the product is still an e-value: the rule has a
higher impact but is never a conviction on its own. Fleet corroboration multiplies members'
e-values (independent data). Shadow mode measures realised crossing rates on live traffic before
any rung is applied, and the `inflation` safety margin in `peers.py` is set from it.

## The repeated game (default parameters)

| Round | PayU moves | Syndicate responds |
|---|---|---|
| 0 | status quo: KYC once, score each payment | a few big fronts |
| 1 | cap volume with an envelope, start the ladder | a fleet of small fronts inside the envelope |
| 2 | link fleets through shared payers | probe the thresholds, sit just under |
| 3 | rotate thresholds at random | join the verified programme, behave, then exploit |
| 4 | vest rewards slowly; scrutiny never fully drops | leave PayU |

## Limits

- Simulated data by default; illicit signatures are set from the node spec and should be learnt from labelled cases (Bustabit for betting, Elliptic for graph structure).
- Gaussian noise model on daily aggregates; very low-count merchants (a few payments a day) are clipped and need a count model.
- The economics are a calibrated model, not measured costs; every number in `config.py` is illustrative.
- Fleet linking uses payer overlap only; device, bank-account and location links would strengthen it.
