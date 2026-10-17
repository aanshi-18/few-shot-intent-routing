# Core study protocol

Frozen before repeated-study execution: 45/15/7/10 category allocation,
three category split seeds, three adaptation seeds, 1/5/10 nested fitting samples.
Two fixed model families and acceptance scores as documented in README.
Additional labelled new-intent development data is used for calibration.
90% supported-development acceptance target, lower quantile method, >= comparison.
No model or threshold changes based on the pilot test results.

The pilot is one configuration, not a repeated study. For each method, the full
post-addition design contains 3 x 3 x 3 = 27 configurations. Pre-addition fits
can be reused once per category split because they do not depend on adaptation
seed or shot count. Preserve paired samples across model families.

Report category-split means and within-split sampling variation without treating
all runs as independent datasets. Report all-supported category F1, old/new
category F1, coverage, correct routing and unknown false acceptance separately.
Include per-model threshold/coverage curves only as diagnostics, not test tuning.
