# Real data

The prototype runs out of the box on the simulator. To run it on real data, download
these public datasets (a Kaggle account is needed for the Kaggle ones) into `data/raw/`
(ignored by git) and pass the paths to the CLI.

| Dataset | What it is used for | File(s) the loader reads | Where |
|---|---|---|---|
| dunnhumby, The Complete Journey | Real grocery stores: the legitimate, kirana-like baseline | `transaction_data.csv` | Kaggle: `frtgnn/dunnhumby-the-complete-journey` |
| Bustabit gambling behaviour | Real betting flows, injected into real stores as semi-real fronts | `bustabit.csv` (columns `Username, Bet, PlayDate`) | OpenML dataset 43364, "Gambling-Behavior-Bustabit" (also mirrored on Kaggle) |
| IPL complete dataset | Real match calendar for match-night effects | `matches.csv` | Kaggle: `patrickb1912/ipl-complete-dataset-20082020` |
| Recruit Restaurant Visitor Forecasting | Restaurant holiday and weekend uplift | `air_visit_data.csv`, `date_info.csv` | Kaggle competition `recruit-restaurant-visitor-forecasting` |
| Corporacion Favorita Grocery Sales | Grocery holiday uplift | `train.csv`, `holidays_events.csv` | Kaggle competition `favorita-grocery-sales-forecasting` |
| Elliptic Bitcoin dataset | Labelled illicit graph: checks that illicit actors cluster (fleet premise) | `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` | Kaggle: `ellipticco/elliptic-data-set` |

With the Kaggle CLI:

```bash
kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey -p data/raw --unzip
kaggle datasets download -d patrickb1912/ipl-complete-dataset-20082020 -p data/raw --unzip
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
kaggle competitions download -c recruit-restaurant-visitor-forecasting -p data/raw
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw
```

Then:

```bash
# real stores + real bets injected into 10 of them, with the real IPL calendar
potf real --dunnhumby data/raw/transaction_data.csv --bustabit data/raw/bustabit.csv \
          --ipl data/raw/matches.csv --fronts 10

# calendar effects for node specs
potf calibrate --recruit-visits data/raw/air_visit_data.csv --recruit-dates data/raw/date_info.csv \
               --favorita-train data/raw/train.csv --favorita-holidays data/raw/holidays_events.csv --max-chunks 4

# fleet premise on a real labelled illicit graph
potf elliptic --classes data/raw/elliptic_txs_classes.csv --edges data/raw/elliptic_txs_edgelist.csv
```

Notes
- dunnhumby amounts are in USD; the loader converts with `fx=83` so the round-amount rule (multiples of Rs 100, at least Rs 500) means the same thing as for Indian merchants. Only stores active on at least 80% of days are kept.
- Bustabit bets are in bits; `--bet-to-rs` scales them. Stakes are rounded to Rs 100 because top-ups into betting books are paid in round amounts.
- The loaders are tested against tiny files with the same column layouts (`tests/test_loaders.py`); the real files have not been run inside this repo's CI.
