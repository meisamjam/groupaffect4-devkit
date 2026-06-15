# Literature Review Audit for `icmi2026_short.tex`

Date: 2026-06-08

Purpose: support the ICMI/BEACON manuscript story with a wider, source-grounded
literature review, identify claims that need citations, and suggest revisions
that keep the paper consistent with the current results.

## 1. Claim-support map

| Manuscript claim or pattern | Current relevance to the paper | Stronger literature support |
|---|---|---|
| Momentary affect is naturally represented as Valence, Arousal, and Dominance rather than only discrete categories. | Supports the use of VAD/SAM targets throughout the paper. | Russell circumplex model (`russell1980circumplex`), PAD model (`mehrabian1996pleasure`), SAM (`bradley1994sam`). |
| SAM ratings are ordinal, tied, and not naturally nominal 3-class labels. | Supports the move from fixed thresholding to balanced tertiles and the original-label ordinal track. | Label distribution learning (`geng2016label`), ordinal label distribution learning (`wen2023ordinal`), class imbalance surveys (`he2009imbalanced`, `johnson2019classimbalance`). |
| Fixed low/mid/high thresholds can create misleading class distributions. | Central to C1 and the argument that target construction is a first-order contribution. | DEAP/DREAMER use dimensional ratings commonly binarised or thresholded; imbalance literature explains why majority-class and missing-class artifacts distort metrics. |
| Physiological affect recognition is established, but most datasets are stimulus-controlled and individual-level. | Supports the gap that GroupAffect-4 addresses. | DEAP, DREAMER, MAHNOB-HCI, WESAD, ASCERTAIN, physiological-signal reviews (`shu2018review`, `ahmad2022survey`). |
| Eye tracking and pupil signals are relevant but not uniquely affect-specific. | Supports the paper's finding that pupil overlaps with EDA/PPG and gaze direction is near-redundant. | MAHNOB-HCI includes eye gaze with physiology; pupil reviews and studies (`partala2003pupil`, `laeng2012pupillometry`) show pupil indexes arousal and cognitive effort, not affect alone. |
| IMU/wrist movement can dominate arousal in co-located group tasks. | Supports the ablation finding that removing IMU causes the largest drop. | Body-movement affect survey (`karg2013body`) and wearable/HAR validation literature. |
| Dominance is harder than Valence/Arousal with physiology-only features. | Supports D6 and future-work framing around posture, relational, gaze, and interaction dynamics. | PAD literature, ELEA/emergent leadership literature, body movement and nonverbal dominance work. |
| Existing social datasets cover dyads, meetings, free mingling, or media viewing, but not structured four-person collaboration with per-person physiology, eye tracking, VAD, and personality. | Main dataset gap. | IEMOCAP, RECOLA, K-EmoCon, AMIGOS, AMI, SALSA, MatchNMingle, ELEA, UDIVA. |
| Personality can moderate affect but may not be learnable from 15-second physiology at N~100. | Supports the BFI conditioning result and reduces overclaiming. | Big Five taxonomy, Watson and Clark, ASCERTAIN/AMIGOS, Meng CTSEM. |
| Semi-supervised learning helps only when pseudo-labels are reliable. | Supports GP LOO filtering, lambda_A=0, and FixMatch caveats. | Mean Teacher (`tarvainen2017mean`), FixMatch (`sohn2020fixmatch`), GP uncertainty (`rasmussen2006gaussian`, `atcheson2017gp`). |
| Subject-dependent or record-wise splits can overestimate generalisation in wearable sensor modelling. | Supports known-team primary split plus LOSO/LOGO stress tests. | Physiological emotion-recognition survey (`ahmad2022survey`), HAR validation-methodology paper (`ventura2022validation`). |
| Classical baselines remain competitive at small sample sizes. | Supports SVM > MLP result without making the neural result look weak. | DEAP/DREAMER/WESAD traditions, physiological signal reviews, class imbalance literature. |

## 2. Comprehensive literature review draft

This is a paper-ready draft, but it is too long for the current manuscript page
budget. The shorter version has been integrated into `icmi2026_short.tex`; this
version can be used for a longer paper, rebuttal, appendix, or thesis chapter.

```latex
\section{Literature Review}
\label{sec:litreview}

\paragraph{Dimensional affect, self-report, and target construction.}
Affective computing has long used dimensional models to describe affective
state. Russell's circumplex model places affective concepts in a
Valence--Arousal space, capturing pleasantness and activation as coupled but
separable dimensions~\cite{russell1980circumplex}. The PAD tradition extends
this view with Dominance/control, a dimension particularly relevant for social
and interpersonal settings~\cite{mehrabian1996pleasure}. The Self-Assessment
Manikin (SAM) operationalises these dimensions as compact visual self-report
scales~\cite{bradley1994sam}. However, SAM responses are ordinal judgments:
a difference between 1 and 2 is not guaranteed to equal a difference between 7
and 8, and integer-valued scales create ties at common thresholds. Treating
these labels as nominal classes can therefore change the learning problem.
Class imbalance further complicates evaluation because majority classes can
dominate training and make accuracy or macro-F1 unstable when classes are
missing in a split~\cite{he2009imbalanced,johnson2019classimbalance}. This
motivates a dual treatment of GroupAffect-4 labels: balanced tertile
classification for comparability with affect-recognition benchmarks, and
ordinal original-label modelling that preserves ordered uncertainty
\cite{geng2016label,wen2023ordinal}.

\paragraph{Physiological and oculomotor affect recognition.}
Physiological affect recognition has matured around multimodal benchmarks.
DEAP combines EEG and peripheral physiology during music-video viewing
\cite{koelstra2012deap}; DREAMER records EEG and ECG from low-cost sensors
\cite{katsigiannis2018dreamer}; MAHNOB-HCI adds synchronized face video,
eye gaze, and peripheral/central physiology in response to affective
stimuli~\cite{soleymani2012mahnob}; and WESAD studies wearable stress and
affect sensing with wrist and chest devices~\cite{schmidt2018wesad}.
ASCERTAIN is especially relevant because it links Big Five personality,
physiological responses, and self-reported affect, although still in a
stimulus-viewing paradigm~\cite{subramanian2018ascertain}. Reviews of
physiological emotion recognition emphasise that EDA, cardiac/PPG features,
respiration, temperature, and motion can all contribute, but performance is
sensitive to preprocessing, annotation, inter-subject variability, and data
splitting~\cite{shu2018review,ahmad2022survey}. These patterns explain why
the present paper treats sensor preprocessing and evaluation protocol as
core contributions rather than boilerplate.

\paragraph{EDA, PPG, pupil, gaze, and motion as affect channels.}
The physiological signals used in GroupAffect-4 map onto partially overlapping
autonomic and behavioural mechanisms. EDA is a sympathetic arousal channel and
is routinely decomposed into tonic and phasic components; cvxEDA provides a
convex decomposition framework and NeuroKit2 implements validated signal
processing routines for EDA and PPG/heart features~\cite{greco2016cvxeda,
makowski2021neurokit2}. Broader autonomic reviews show that emotion-related
responses distribute across skin conductance, cardiac, respiratory, vascular,
and thermal systems rather than a single sensor~\cite{kreibig2010autonomic}.
Pupil diameter is likewise informative but not specific: it responds to
affective arousal and to attention, cognitive effort, and preconscious
processing~\cite{partala2003pupil,laeng2012pupillometry}. Body movement is
a recognised affective expression channel~\cite{karg2013body}. This literature
aligns with the paper's ablation: IMU features dominate arousal in structured
group tasks, while pupil shares variance with EDA/PPG and gaze direction adds
little unique affect information in the physiology-only feature set.

\paragraph{Social, dyadic, and group affect corpora.}
Existing corpora capture important parts of the target setting but rarely the
full combination of co-located four-person collaboration, per-person
physiology, eye tracking, self-reported VAD, and personality. IEMOCAP provides
dyadic acted interaction with rich multimodal recordings~\cite{busso2008iemocap},
and RECOLA records dyadic remote collaboration with continuous valence/arousal
annotations and physiological signals~\cite{ringeval2013introducing}.
K-EmoCon moves closer to natural social interaction through dyadic debate,
wearable sensors, and self/partner/observer annotations~\cite{park2020kemocon}.
AMIGOS explicitly studies affect, personality, mood, and social context, but
its group setting is shared media viewing rather than collaborative task
performance~\cite{mirandacorrea2021amigos}. Meeting and social-signal corpora
such as AMI~\cite{mccowan2005ami}, SALSA~\cite{alameda2015salsa},
MatchNMingle~\cite{cabrera2018matchnmingle}, ELEA~\cite{sanchez2012emergent},
and UDIVA~\cite{palmero2021context} provide rich social behaviour,
leadership, personality, or dyadic context, but generally lack the
physiology-eye-tracking-VAD-personality combination required for the present
question. GroupAffect-4 is therefore best framed not as a larger benchmark but
as a denser bridge between affective computing and social signal processing.

\paragraph{Personality and affect dynamics.}
The Big Five is the dominant taxonomy for broad personality traits
\cite{john1999bigfive}. Neuroticism and Extraversion have long been associated
with negative and positive affective experience~\cite{watson1992traits}, and
datasets such as ASCERTAIN and AMIGOS show that personality can shape
physiological affect responses~\cite{subramanian2018ascertain,
mirandacorrea2021amigos}. More recent dynamic modelling suggests that the Big
Five moderate VAD trajectories in dimension-specific ways and over different
timescales~\cite{meng2026moderating}. This supports per-dimension BFI
conditioning as a hypothesis, but it also predicts difficulty: stable traits
operate over longer temporal horizons than 15-second windows, and their
moderating effects may depend on the social task. The paper's negative
canonical BFI result should therefore be framed as evidence that trait
conditioning is context- and sample-size-sensitive, not as evidence that
personality is irrelevant.

\paragraph{Learning from sparse, imbalanced, and uncertain labels.}
GroupAffect-4 has far more unlabelled sensor windows than labelled VAD
windows, a common pattern in wearable affect datasets. Time-series
augmentation and contrastive representation learning can help when labels are
scarce~\cite{um2017data,iwana2021augmentation,eldele2021tscc}, but the paper's
results show that adding data is useful only when the target semantics remain
aligned. Label smoothing improves calibration in neural classifiers but can
also remove useful class-similarity information~\cite{muller2019label};
ordinal label distribution learning is better matched to SAM because adjacent
ratings are semantically closer than distant ratings~\cite{wen2023ordinal}.
Semi-supervised consistency methods such as Mean Teacher and FixMatch use
unlabelled data by enforcing prediction stability, but FixMatch explicitly
depends on high-confidence pseudo-labels~\cite{tarvainen2017mean,
sohn2020fixmatch}. Gaussian Processes provide an alternative label-space
strategy with explicit posterior uncertainty~\cite{rasmussen2006gaussian,
atcheson2017gp}. The present results fit this literature: GP augmentation
helps task-CV after threshold alignment, hurts canonical T3 when the augmented
target distribution is mismatched, and Arousal augmentation must be disabled
when GP LOO calibration is below chance.

\paragraph{Evaluation protocol and generalisation.}
Physiological affect recognition is strongly affected by inter-subject
variance, annotation choices, and data splitting~\cite{ahmad2022survey}.
Wearable activity-recognition studies show that record-wise or ordinary
k-fold validation can overestimate performance when windows from the same
individual appear in both train and test sets~\cite{ventura2022validation}.
For GroupAffect-4, no single split answers all deployment questions. The
temporally ordered known-team split is appropriate for monitoring an already
observed group over later task phases. LOSO estimates cross-person transfer on
the same target task, and LOGO estimates the harder unseen-group setting.
Presenting these protocols as distinct regimes is central to the paper's
credibility: the lower LOSO and LOGO results are not failures, but evidence
that the system has measured the deployment boundary rather than hiding it.
```

## 3. Re-evaluation of the paper story

### What the new literature strengthens

- The strongest story is not "personality-modulated neural model beats all
  baselines." The results do not support that.
- The strongest story is: in small naturalistic group affect data, careful
  target construction, uncertainty-aware augmentation, and deployment-matched
  validation determine whether model results are meaningful.
- The SVM result becomes a positive contribution rather than an embarrassment:
  it is consistent with physiological affect literature where handcrafted
  features and margin-based classifiers remain strong under small N.
- The ordinal track should stay prominent because it is the cleanest response
  to the literature on ordinal labels, label distributions, and SAM scale
  semantics.
- The LOGO result should remain in the paper. It is low, but it sharpens the
  boundary between known-team monitoring and unseen-group deployment.

### What should be toned down

- Avoid implying that BFI conditioning is generally effective. The literature
  supports personality as a moderator, but the paper's results show the current
  conditioning method is underpowered and task-context-sensitive.
- Avoid describing GP augmentation as a general solution. The literature and
  results both point to calibration-gated, dimension-specific augmentation.
- Avoid making dominance a model failure. Dominance is an interpersonal and
  relational dimension; physiology-only inputs are plausibly under-instrumented
  for it.
- Avoid presenting FixMatch as a uniform gain. Its usefulness depends on
  confidence and split context.

### Suggested paper improvements

1. Keep Related Work compact but cite the missing bridge datasets:
   MAHNOB-HCI, ASCERTAIN, AMIGOS, K-EmoCon, AMI, and SALSA.
2. Use the phrase "known-team monitoring" consistently for the primary split,
   and reserve "unseen-group deployment" for LOGO.
3. Add one sentence to Discussion D1 saying that ordinal modelling is not an
   add-on metric but the label-space version of the paper's main thesis.
4. In D4/D6, connect IMU and dominance more explicitly to body movement and
   nonverbal social-signal literature.
5. Consider a small table in the supplement comparing corpora by group size,
   physiology, eye tracking, VAD self-report, and personality.
6. In the abstract, avoid too many method names. The high-level claim should
   be target quality + uncertainty + evaluation protocol, then give the key
   results.
7. If page budget is tight, move the full literature review to a supplement
   and leave the manuscript with the five-paragraph related work now in the
   `.tex` file.

## 4. Key external sources consulted

- MAHNOB-HCI: https://archive-ouverte.unige.ch/unige:47406
- ASCERTAIN: https://iris.unitn.it/handle/11572/212724
- AMIGOS: https://colab.ws/articles/10.1109%2Ftaffc.2018.2884461
- K-EmoCon: https://www.nature.com/articles/s41597-020-00630-y
- SALSA: https://arxiv.org/abs/1506.06882
- Ordinal Label Distribution Learning: https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Ordinal_Label_Distribution_Learning_ICCV_2023_paper.html
- FixMatch: https://papers.nips.cc/paper_files/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html
- Physiological signal survey: https://www.mdpi.com/2306-5354/9/11/688
- Validation methodology in wearable sensing: https://www.mdpi.com/1424-8220/22/6/2360
