Hypothesis:
Market conditions change, therefore the parameters appropriate for the stock may also change, and an optimizer using
recent historical behavior may adapt to those changes.


Methodology:
We can test this by having two different optimizers. 

**A static optimizer will optimze for the best parameters over time. It finds robust parameters and leaves them fixed.**

```
2015                                  2026
|---------------|---------|-------------|
     TRAIN       VALIDATE       TEST

Finds one parameter set
```

**A dynamic walk-forward optimizer which will coninually test and optimze for the best parameters for the current market.**
*This tests our hypothesis.*

```
TIME ───────────────────────────────────────────►


     TRAIN 1             TEST 1
|-------------------|-----------|
      optimize         frozen
         ↓
      Params A ──────────►


             TRAIN 2             TEST 2
        |-------------------|-----------|
                 optimize         frozen
                    ↓
                 Params B ──────────►


                       TRAIN 3             TEST 3
                  |-------------------|-----------|
                           optimize         frozen
                              ↓
                           Params C ──────────►

IN-SAMPLE 1 → OOS 1
IN-SAMPLE 2 → OOS 2
IN-SAMPLE 3 → OOS 3
IN-SAMPLE 4 → OOS 4
```
If parameters change and the following out of sample (OOS) performs well, then this supports the dynamic theory. If the parameters constantly jump around and the OOS performance collapses then that is evidence of fitting for historical noise rather
than detecting changing conditions. 


There are 3 levels of research
                  LEVEL 3
           Optimizer configuration
        "How should adaptation work?"
                    │
                    ▼
                  LEVEL 1
             Strategy selection
          "What approach works?"
                    │
                    ▼
                  LEVEL 2
            Parameter selection
        "What settings work now?"
                    │
                    ▼
           NEXT UNSEEN PERIOD
                    │
                    ▼
              Actual result

Must have a final test/holdout sample to reduce overfitting as much as possible 


```
STOCK
  │
  ▼
OPTIMIZER POLICY
"How do I adapt?"
  │
  ├── historical lookback
  ├── reoptimization frequency
  ├── candidate requirements
  └── candidate scoring
  │
  ▼
STRATEGY SELECTION
"What currently works best?"
  │
  ├── RSI Double Dip
  ├── RSI Long Only
  ├── SMA Long Only
  └── ...
  │
  ▼
PARAMETER SELECTION
"What settings currently work best?"
  │
  ├── RSI length
  ├── thresholds
  └── Double Dip spacing
  │
  ▼
FROZEN CONFIGURATION
  │
  ▼
NEXT UNSEEN PERIOD
  │
  ▼
BUY / SELL / HOLD

```