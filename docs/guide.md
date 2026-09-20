# Cleaning

Need 3 outputs:

**Cleaned recordings (Parquet)**
	All trimmed/cleaned PLTs
	1 recording per visit per participant
	Filename: `UNIQUE_ID.parquet`

**Recordings index (CSV)**
    One file tracking all recordings.
	Participant, visit, drive scenario, corresponding filename (`UNIQUE_ID`)

**Segments index (CSV)**
    One file tracking each event in each recording
    Segment name, type, code, start, end, recording (`UNIQUE_ID`)

==Confirm start/end = timestamps or row positions? What units? Is end included? Must match the cleaned recording.==

# Representations

Cleaned time series → segment stats → feature table → preprocessing

**Initial plan:**

**Load + segment**
	Recordings index → load cleaned recordings
	Segments index → select events

**Calc features**
	Stats per variable/event: mean, SD, min, max as relevant
	Put into 1 row per recording for drive-level labels
	Keep IDs for tracking/splitting, not model inputs

**Compare reps**
	Same participant splits across approaches
	Fit learned preprocessing within training folds only

==Confirm exact variables/stats, how to combine repeated events, how to handle missing events. Confirm clinician labels and their level before finalizing.==

# I: Preprocess

Driving features → preprocessing → supervised model → predicted clinician assessment

Compare preprocessing + model combos for predicting error counts and safety labels.

| Approach                         | Inputs passed to predictor                                                | Consider                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Lit-guided**                   | Literature-guided variables; baseline                                     | Could miss useful/keep redundant feats; compare against PCA to see if it helps prediction |
| **Lit+PCA-guided**               | Literature-guided variables; clustering informs selection                 | Helps spot redundancy; high variance != predictive value                                  |
| **PCA components**               | PCA's weighted combos of selected, scaled variables                       | May lose clinically useful variation                                                      |
| **ICA components**               | Combos seeking statistically independent components                       | Different objective from PCA's max variance; check stability and predictive value         |
| **Variable clustering**          | \*                                                                        | \*                                                                                        |
| **Factor analysis**              | \*                                                                        | \*                                                                                        |
| **Supervised feature selection** | OG feats selected for predicting clinician error counts or safety labels. | Uses clinician targets; can overfit unless selection stays within training folds          |
\**TBD*
# II: Predict

| Task / clinician target               | Models to compare                                                                                  | Prediction                                                               | Evaluation                                                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Regression:** error count per drive | Poisson /<br>neg binom regression<br><br>Rand forest /<br>grad boosting regressor                  | Predicted error count                                                    | Mean absolute error                                         |
| **Classification:** safe/unsafe label | Logistic regression<br><br>Random forest classifier<br><br>Gradient boosting classifier<br><br>SVM | Safety label<br><br>(+ probability / decision score, depending on model) | ROC-AUC<br>(sensitivity & specificity @ selected threshold) |

==Check whether incoming “validation data” can be used for training.==

**Research question:** How do measured driving-performance *scores* change from baseline (T1) to the short-term post-FFP visit (T2) and the eight-week follow-up (T3)?

**Limitations:** This describes change, not FFP causality. There are practice/time effects and T3's scenario effect (drive not random) remain.

# Stats approach

The hypotheses below assume a higher-is-better score. For predicted error count, improvement means a decrease instead.

>**Descriptive hypothesis:** Mean *scores* increase from T1 to T2 and again from T2 to T3 in our sample.

$$
\bar{S}_{T2}-\bar{S}_{T1}>0
\quad\text{and}\quad
\bar{S}_{T3}-\bar{S}_{T2}>0
$$

>**Population hypothesis:** Mean *scores* increase from T1 to T2 and again from T2 to T3 in the population our sample represents.

$$
\Delta_{12}=\mu_{T2}-\mu_{T1}>0
\quad\text{and}\quad
\Delta_{23}=\mu_{T3}-\mu_{T2}>0
$$

**Analysis:**
Fix the trained scoring rule across visits
→ estimate T2−T1 & T3−T2 accounting for repeated scores per driver
→ use an appropriate repeated-measures model for each output; linear mixed-effects is a candidate for continuous scores.

# Post-meet

Goal is to use multiple ML methodologies to best represent variables as a composite score.

How to test accuracy?
Beyond a norm value = bad performance
Above is good

Clinician in passenger seat scores while participant driving. This is where we get clinical ground-truth based on below errors. This is in the validation data Prof will send soon
	
Supervised target / ground truth = expert / clinician's error count & safety classification
- How do they classify errors?
	- In certain zones, >speed limit like in school = ERROR
	- roundabout screw up = ERROR
	- hitting cyclist = ERROR

Looks like we can use this for classification now



