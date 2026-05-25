# CHAPTER 3 — METHODOLOGY AND SYSTEM DESIGN

> **How to use this file.** This document is ready to paste into the
> AASTMT FYP Word template under *Chapter 3*. Headings follow the
> template's outline numbering style. Figures are referenced by their
> filenames in `Current Thesis/figures/`; insert each at the bracketed
> location and apply the *Caption* style. Equations should be inserted
> via Word's *Insert → Equation* tool; the LaTeX-style source for each
> equation is included verbatim below the equation reference so the
> author can re-render in Word's equation editor.

---

## 3.1  HIGH-LEVEL SYSTEM ARCHITECTURE

The Eshara system is decomposed into three loosely-coupled layers, the
boundaries of which are shown in *Figure 3-1*. The *client layer* is
either a web browser or a mobile phone and is responsible for capturing
the camera frame and producing a numerical representation of the
signer's hands and body. The *server layer* — a FastAPI-based Python
service — routes each frame through a motion-based mode detector and
selects either a letter classifier (a multi-layer perceptron) or a
word classifier (a recurrent network) on a per-frame basis. The
*auxiliary layer* — a separate Node.js service — handles user accounts
and conversational features that are orthogonal to recognition but
required for a production product. This decomposition was deliberate:
it isolates compute-heavy machine-learning code from authentication
state and from user data, which can therefore be updated independently
and benefit from different scaling policies in the cloud.

[Insert *Figure 3-1: High-level system architecture*. File:
`figures/fig_3-1_system_architecture.png`. Caption: "Block diagram of
the Eshara system. Camera frames are converted to landmarks on the
client, then routed by a mode detector on the server to either the
letter or word classifier, before being decoded into characters or
words for display."]

A practical advantage of running the landmark extractor on the client
is privacy: the raw video never leaves the user's device. Only the
landmark vector (63 or 258 floats per frame) is transmitted, which is
roughly three orders of magnitude smaller than the corresponding video
frame and removes any biometric reconstruction risk. A practical
advantage of running the classifier on the server is consistency: every
user receives predictions from the same model weights, and model
updates are deployed once rather than to every installed mobile app.

## 3.2  DATASETS

### 3.2.1  Overview

Four publicly-available datasets were used in this work, one per
recognition task. *Table 3-1* summarises them. Two datasets address
letter recognition (one per language) and two address word recognition
(one per language). The letter datasets are *image-based* — each sample
is a still photograph of a single signed letter — while the word
datasets are *video-based* — each sample is a short clip of the signer
producing a single word.

[Insert *Table 3-1: Summary of datasets used in this work.* Columns:
*Dataset*, *Language*, *Type*, *Classes*, *Samples*, *License*.]

| Dataset      | Language | Type   | Classes | Samples         | License   |
|--------------|----------|--------|---------|-----------------|-----------|
| ASL Alphabet | ASL      | Images | 29      | 87,000          | CC0       |
| ArASL2018    | ArSL     | Images | 32      | 54,049          | Research  |
| WLASL-157    | ASL      | Videos | 157     | ≈ 1,700         | Research  |
| KArSL-502    | ArSL     | Videos | 502     | ≈ 24,000        | Research  |

The most challenging of the four is KArSL-502: although it contains
twenty-four thousand clips, they are distributed across five hundred
and two word classes, three signers, and eight repetitions, yielding
only eight samples per (class, signer) pair. This severe class
scarcity dictates several methodological choices that are revisited
below (smaller network width, stronger regularisation, label smoothing,
and balanced class weights).

### 3.2.2  ASL Alphabet (Akash, Kaggle)

The ASL Alphabet dataset comprises 29 classes (the 26 letters A–Z plus
the auxiliary tokens *space*, *delete*, and *nothing*), with
approximately three thousand 200 × 200-pixel RGB photographs per class.
The dataset was released under the Creative Commons CC0 licence on
Kaggle. The images vary in lighting and background, which the present
work exploits by treating each image as a single MediaPipe-Hands
landmark vector rather than as raw pixels, thereby discarding most of
the per-image variance in favour of pose-invariant skeletal features.

### 3.2.3  ArASL2018

ArASL2018 contains 32 Arabic letter classes and 54,049 greyscale
images. Because Arabic finger-spelling is right-hand dominant for the
majority of signers, an additional horizontal flip is applied during
pre-processing to augment a left-handed minority. The dataset is
research-licensed and is cited from the original publication.

### 3.2.4  WLASL — Subset of 157 ASL Words

The Word-Level American Sign Language (WLASL) dataset originally
contains two thousand isolated word classes drawn from YouTube. For
this work we used a curated 157-class subset, selected to maximise
overlap with the publicly used 1000- and 100-class subsets in the
literature while still being trainable on the available GPU budget.
The same MediaPipe Holistic pipeline (Section 3.3.2) is applied to
every clip.

### 3.2.5  KArSL-502

The KArSL-502 dataset (Sidig et al., 2021) is, to the author's
knowledge, the most comprehensive standardised Arabic sign-language
corpus available. It contains 502 word classes performed by three
signers, with eight repetitions per word, for a total of
approximately twenty-four thousand video clips. The clips average
two seconds in length at 25 frames per second. For this work, the
dataset is partitioned in a *signer-aware* manner: signers 01 and 02
contribute their full set of clips to the training and validation
splits, while signer 03 is held out as the test set. This split
provides a realistic estimate of signer-independent generalisation,
which is the deployment scenario the system must ultimately satisfy.

### 3.2.6  Pre-processing Pipeline

The pre-processing pipeline is identical for the two word datasets and
is depicted in *Figure 3-2*. Each input clip is decoded with OpenCV,
uniformly sub-sampled to a fixed number of frames *T* (30 for ASL
words, 48 for ArSL words v2), and passed through MediaPipe Holistic to
obtain pose, left-hand, and right-hand landmarks for every frame.
Missing landmarks (frames in which a hand is occluded or off-screen)
are replaced by zeros. The resulting tensor of shape *(T, F)* — where
*F* is the per-frame feature dimension — is wrist-centred and
scale-normalised, then concatenated into a single 258-dimensional
vector per frame. To avoid re-extracting features on every run, the
entire *(N, T, F)* training tensor is serialised to a compressed `.npz`
cache file. The cache check additionally verifies the recorded shape
and sample count, so that any reconfiguration of *T* or *F*
automatically invalidates a stale cache.

[Insert *Figure 3-2: Pre-processing pipeline*. File:
`figures/fig_3-2_preprocessing_pipeline.png`. Caption: "Pre-processing
pipeline from raw MP4 clips to cached *.npz* tensor."]

### 3.2.7  Data Augmentation

For the word datasets, on-the-fly augmentation is applied within the
TensorFlow data pipeline (see Section 3.8). For each training sample
*x ∈ ℝ ^ (T × F)* the following five transformations are applied with
their stated probabilities:

1. **Gaussian noise.** Independent additive noise *N (0, σ²)* with
   *σ = 0.005* is added to every entry of *x*. This simulates the
   per-frame jitter inherent to MediaPipe's landmark estimator.

   *x ← x + N (0, σ²)*    where σ = 0.005     (Eq. 3.1)

2. **Temporal shift.** A circular roll along the time axis by *k*
   frames, where *k* is uniformly sampled from *{−3, … , 3}*. This
   makes the model invariant to small differences in when the sign
   actually begins inside the *T*-frame window.

   *x ← roll(x, k)*      *k ~ U {−3, … , 3}*   (Eq. 3.2)

3. **Frame dropout.** A binary mask is generated independently per
   frame with *P(keep) = 0.9* and is broadcast across the feature
   dimension. The model is therefore trained to remain accurate even
   when up to ten percent of frames are missing.

4. **Random scale.** The entire tensor is multiplied by a scalar
   *s ~ U (0.9, 1.1)*. This emulates the variability in signer
   distance from the camera.

5. **Horizontal flip (left/right hand swap).** With probability 0.5,
   the left-hand block and right-hand block of the feature vector are
   exchanged. Because pose features remain in place, the network must
   learn hand-shape semantics that are invariant to chirality.

These five augmentations were selected after small ablation
experiments (Chapter 5) and together expand the effective training
set by roughly a factor of ten.

## 3.3  FEATURE EXTRACTION

### 3.3.1  Letter Feature Vector — 63 Dimensions

For the letter pipeline, MediaPipe Hands is run on the single input
frame and returns 21 hand landmarks *(x_i, y_i, z_i)* for i = 1, … ,
21, where landmark zero is the wrist. The 63 raw values are then
wrist-centred,

* x̃_i = x_i − x_0,
* ỹ_i = y_i − y_0,
* z̃_i = z_i − z_0,         (Eq. 3.3)

and scaled by the maximum landmark-to-wrist Euclidean distance,
yielding a unit-magnitude vector that is independent of the signer's
position in the frame and of the absolute camera distance.

### 3.3.2  Word Feature Vector — 258 Dimensions

For the word pipeline, MediaPipe Holistic is used because many word
signs are not localised at the hands alone but use body landmarks for
spatial reference (for example *house*, *I*, or *you*). The
representation concatenates 33 pose landmarks — each carrying *(x, y,
z, visibility)* — with the left- and right-hand landmark blocks:

*F_word = 33 × 4 (pose) + 21 × 3 (left hand) + 21 × 3 (right hand) = 258*    (Eq. 3.4)

The internal layout adopted in code is

```
indices  0 ... 131 | 132 ... 194 | 195 ... 257
         pose 132   |  L hand 63   |  R hand 63
```

This layout is preserved through the augmentation function so that the
horizontal-flip augmentation can swap the left- and right-hand blocks
by simple tensor slicing.

### 3.3.3  Temporal Sampling

For letter recognition, a single frame suffices because finger-spelling
is, by definition, a static hand shape. For word recognition, two
window lengths are evaluated: a 30-frame window for the ASL pipeline
(Model 3) and a 48-frame window for the ArSL v2 pipeline (Model 4).
The longer window in the ArSL pipeline was motivated by the typical
two-second duration of KArSL clips at 25 frames per second; a 30-frame
window often truncates the sign's articulation phase, whereas a
48-frame window comfortably contains the full sign arc. The ablation
study in Chapter 5 quantifies the effect of this design choice.

## 3.4  MODEL 1 — ASL LETTER MLP

The ASL letter classifier is a feed-forward Multi-Layer Perceptron
(MLP) that maps a 63-dimensional landmark vector to a 29-way softmax
distribution over the letter classes. The architecture is shown in
*Figure 3-3* and consists of two hidden Dense layers (256 and 128
units respectively, both with ReLU activation) interleaved with
Dropout (rate 0.3) followed by the output Dense layer.

[Insert *Figure 3-3: ASL letter MLP architecture*. File:
`figures/fig_3-5_asl_letter_mlp.png`. Caption: "ASL letter MLP — input
63-dimensional landmark vector → Dense(256) → Dropout → Dense(128) →
Dropout → Dense(29) softmax."]

The forward pass of each hidden layer ℓ is the standard MLP recurrence

*h ^ (ℓ) = σ ( W^(ℓ) h^(ℓ−1) + b^(ℓ) )*      with    *σ(x) = max(0, x)*    (Eq. 3.5)

and the output layer uses softmax to produce a probability
distribution over the 29 classes. Training is performed with the Adam
optimiser at a learning rate of 1 × 10⁻³ and a batch size of 64 for up
to 50 epochs, with EarlyStopping on validation accuracy with patience
8. The model has approximately forty-nine thousand parameters — about
two orders of magnitude smaller than a comparable MobileNetV2 image
classifier — and runs in well under one millisecond on a desktop CPU,
which makes it suitable for in-browser inference.

## 3.5  MODEL 2 — ArSL LETTER MLP

The ArSL letter classifier shares Model 1's architectural family with
two differences. First, the output layer has 32 units to match the
ArASL2018 class count. Second, the pre-processing pipeline applies an
additional mirror augmentation because Arabic finger-spelling is
right-hand dominant. The same Adam optimiser settings as Model 1 are
used, and the same EarlyStopping criterion applies.

It is worth noting that, although the architecture is identical to
Model 1, the two models are trained completely independently because
the language-specific class labels do not align. A shared backbone is
investigated as future work in Chapter 8.

## 3.6  MODEL 3 — ASL WORD BiLSTM WITH TEMPORAL ATTENTION

The ASL word classifier is the first of the two recurrent models and
is shown in *Figure 3-4*. The 30-frame, 63-dimensional input tensor
is first processed by two stacked Bidirectional LSTM layers (128 and
64 units respectively, both returning the full sequence), followed by
a custom *temporal attention* layer that collapses the time dimension
into a fixed-size context vector. A single Dense layer with 128 ReLU
units, followed by Dropout and the softmax output layer, completes the
classifier.

[Insert *Figure 3-4: ASL Word BiLSTM with TemporalAttention*. File:
`figures/fig_3-6_asl_word_bilstm.png`. Caption: "Bidirectional LSTM
stack with a custom additive temporal-attention head used for ASL
word recognition."]

### 3.6.1  The LSTM Cell

The Long Short-Term Memory (LSTM) cell (Hochreiter and Schmidhuber,
1997) maintains an internal cell state c_t and hidden state h_t
updated by an input gate i_t, forget gate f_t, and output gate o_t:

```
i_t = σ ( W_i x_t + U_i h_{t-1} + b_i )           input gate
f_t = σ ( W_f x_t + U_f h_{t-1} + b_f )           forget gate
o_t = σ ( W_o x_t + U_o h_{t-1} + b_o )           output gate
c̃_t = tanh ( W_c x_t + U_c h_{t-1} + b_c )        candidate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t                  cell state
h_t = o_t ⊙ tanh(c_t)                             hidden state
```                                                  (Eq. 3.6)

The bidirectional variant (Schuster and Paliwal, 1997) processes the
sequence in both directions and concatenates the forward and backward
hidden states,

*h_t = [ forward h_t ; backward h_t ]*           (Eq. 3.7)

so that each position in the output has access to both past and
future context.

### 3.6.2  Temporal Attention

The custom *TemporalAttention* layer implements an additive
(Bahdanau-style) attention over the time dimension. Given a sequence
of hidden states *{h_1, … , h_T}*, the layer learns a weight vector
*W ∈ ℝ ^ d × 1* and a bias *b ∈ ℝ ^ T × 1* and produces a context
vector *c ∈ ℝ ^ d* by

* e_t = tanh ( W h_t + b ),
* α_t = exp(e_t) / Σ_{t'=1}^{T} exp(e_{t'}),
* c   = Σ_{t=1}^{T} α_t h_t.                       (Eq. 3.8)

In practice the layer adds fewer than one percent of the parameters of
the underlying BiLSTM stack — *d + T* additional parameters versus
hundreds of thousands in the recurrent layers — yet provides a
measurable Top-5 accuracy gain in the ablation study of Chapter 5.

The Keras implementation of the layer is given in *Listing 3-1*.

```python
class TemporalAttention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight('att_W',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform')
        self.b = self.add_weight('att_b',
            shape=(input_shape[1], 1),
            initializer='zeros')

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)
```

[*Listing 3-1: The custom* TemporalAttention *Keras layer used by
Model 3.*]

## 3.7  MODEL 4 — ArSL WORD V2 (IMPROVED ARCHITECTURE)

The fourth and most architecturally elaborate model is the Improved
ArSL Word v2 classifier, depicted in *Figure 3-5*. Six design choices
distinguish it from Model 3: (i) the input is the richer 258-dimensional
pose-and-hands vector defined in Section 3.3.2 rather than the 63-D
hands-only vector; (ii) the sequence length is extended from 30 to 48
frames; (iii) two TimeDistributed Dense layers act as a *per-frame
spatial encoder* before the recurrent stack, projecting the 258-D
input down to 192 then to 128 dimensions per frame, with intervening
BatchNorm and Dropout; (iv) the recurrent stack consists of two
Bidirectional LSTMs (128 and 96 units) with SpatialDropout1D between
them, followed by a single unidirectional LSTM (64 units) that returns
the last hidden state only; (v) a deep classifier head of two Dense
layers (384 then 192 units, each with BatchNorm and Dropout) precedes
the softmax output; (vi) the training schedule uses Cosine Annealing
with warm restarts (Section 3.8) rather than ReduceLROnPlateau.

[Insert *Figure 3-5: ArSL Word v2 architecture*. File:
`figures/fig_3-7_arsl_word_v2.png`. Caption: "ArSL Word v2 — a
TimeDistributed spatial encoder feeds a stacked BiLSTM + LSTM
temporal encoder followed by a deep classifier head."]

The full hyperparameter set for Model 4 is collected in *Table 3-2*.
All values are taken from Cell 3 of
`Words/ArSL Word (Arabic)/ArSL_Word_Training_v2.ipynb`.

[Insert *Table 3-2: Hyperparameters for the ArSL Word v2 model.*]

| Hyperparameter         | Value                              |
|------------------------|------------------------------------|
| Sequence length        | 48 frames                          |
| Feature dim            | 258 (pose 132 + left 63 + right 63) |
| Batch size             | 32                                 |
| Optimiser              | Adam + Cosine-Annealing w/ restarts |
| Initial learning rate  | 5 × 10⁻⁴                           |
| Label smoothing ε      | 0.10                               |
| Gradient clip-norm     | 1.0                                |
| Dropout (general)      | 0.40                               |
| Dropout (TD-encoder)   | 0.20                               |
| L2 weight decay        | 1 × 10⁻⁴                           |
| Spatial-encoder dims   | 192 → 128                          |
| LSTM widths            | 128 (Bi) → 96 (Bi) → 64 (uni)      |
| Dense-head widths      | 384 → 192                          |
| Class-weight clip      | [ 0.5, 10 ]                        |

The rationale for each value is given in the comments of the source
notebook: the smaller LSTM widths (128 → 96 → 64) compensate for the
KArSL-502 class scarcity of eight samples per class, the SpatialDropout
rate of 0.4 is deliberately high to combat the same scarcity, and the
class-weight clip prevents very rare classes from dominating gradient
updates.

## 3.8  TRAINING METHODOLOGY

### 3.8.1  Loss Function

All four models are trained against a label-smoothed categorical
cross-entropy loss. Given one-hot targets *y*, the smoothed targets

*ỹ_k = (1 − ε) y_k + ε / K*                       (Eq. 3.9)

(where ε = 0.10 for the word models and 0 for the letter models, and
K is the class count) are substituted into the standard cross-entropy

*L = − Σ_{k=1}^{K} ỹ_k log ŷ_k*                  (Eq. 3.10)

producing the per-sample loss. Label smoothing discourages the model
from becoming over-confident on small classes, which is particularly
desirable for KArSL-502.

### 3.8.2  Optimiser

The Adam optimiser (Kingma and Ba, 2015) is used throughout, with the
standard recurrences

```
m_t = β1 m_{t-1} + (1 − β1) g_t
v_t = β2 v_{t-1} + (1 − β2) g_t ⊙ g_t
m̂_t = m_t / (1 − β1 ^ t)
v̂_t = v_t / (1 − β2 ^ t)
θ_t = θ_{t-1} − η · m̂_t / ( √v̂_t + ε )
```                                                 (Eq. 3.11)

For the word models, a gradient clip norm of 1.0 is applied before
the parameter update to prevent the occasional exploding-gradient
events characteristic of recurrent networks.

### 3.8.3  Learning-Rate Schedule

Model 4 uses Cosine Annealing with warm restarts (Loshchilov and
Hutter, 2017). Within each restart of length T_i, the learning rate
follows

*η_t = η_min + (η_max − η_min) · (1 + cos(π · t / T_i)) / 2*    (Eq. 3.12)

with the cycle length doubling at each restart (T_mul = 2.0) and the
amplitude shrinking by ten percent (M_mul = 0.9). This schedule has
been shown empirically to escape sharp local minima on small datasets
and is particularly suited to the small-per-class KArSL setting.

### 3.8.4  Class-Weight Balancing

Because the KArSL-502 dataset's per-class counts vary slightly, a
balanced class-weight scheme is used during training. The weight for
class k is

*w_k = clip ( N / (K · n_k),  0.5,  10 )*           (Eq. 3.13)

where N is the total sample count, K is the number of classes, and
n_k is the number of samples of class k. The clip prevents extremely
rare classes from dominating the loss while still up-weighting
under-represented classes.

### 3.8.5  Regularisation Stack

A multi-pronged regularisation stack is applied to the word models:
Dropout (rate 0.4) inside every Dense and recurrent layer;
SpatialDropout1D (rate 0.4) on the BiLSTM outputs to drop entire
feature channels rather than individual time-step activations;
BatchNorm after every Dense and recurrent layer; L2 weight decay of
1 × 10⁻⁴ on every kernel; gradient clipping (Pascanu et al., 2013) at
unit norm; and label smoothing as described above. The cumulative
effect of these techniques on validation accuracy is reported in the
ablation study of Chapter 5.

### 3.8.6  Train / Validation / Test Splits

For the letter datasets, a stratified 80 / 20 split into train and
validation is used; no separate test set is necessary because the
letter datasets are themselves very large relative to the model
capacity. For the word datasets, a 60 / 20 / 20 stratified split is
used. For KArSL-502 specifically, the split is *signer-aware*:
signers 01 and 02 contribute their full clip sets to training and
validation, while signer 03 is held out as the test set. This
construction gives a conservative estimate of signer-independent
generalisation, which is the deployment scenario the system must
ultimately satisfy.

## 3.9  EVALUATION METRICS

Four families of metrics are reported in Chapter 5: classification
accuracy, per-class and macro F1 scores, top-K accuracy for the word
models, and run-time metrics (latency and frames per second). Their
definitions are given below for completeness.

### 3.9.1  Precision, Recall, and F1

For each class k, the precision P_k, recall R_k, and F1_k are
defined in the usual way:

* P_k = TP_k / ( TP_k + FP_k ),
* R_k = TP_k / ( TP_k + FN_k ),
* F1_k = 2 P_k R_k / ( P_k + R_k ).                (Eq. 3.14)

The *macro-F1* is the unweighted mean across classes,

*F1_macro = (1 / K) Σ_{k=1}^{K} F1_k*               (Eq. 3.15)

and the *weighted F1* uses the per-class support n_k as weights,

*F1_weighted = Σ_{k=1}^{K} (n_k / N) F1_k.*         (Eq. 3.16)

The macro F1 is reported preferentially because it does not let large
classes dominate.

### 3.9.2  Top-K Accuracy

For the word models, top-1 and top-5 accuracy are reported. The
top-K accuracy is

*Top-K = (1 / N) Σ_{i=1}^{N} 𝟙 [ y_i ∈ TopK( p̂_i ) ]*    (Eq. 3.17)

i.e., the fraction of test examples whose true label appears among
the model's top-K predicted classes. Top-5 is particularly informative
for KArSL-502 because the 502-way classification is hard, and a
top-5 list is short enough to be useful at the user-interface level
(for example as a disambiguation menu).

### 3.9.3  Confusion Matrices

For each model a confusion matrix is computed. For the letter models
the matrix is shown in full; for the word models the full 502 × 502
matrix is too dense to print, so a top-20 sub-matrix is presented in
Chapter 5 and the full matrix is included in Appendix G.

### 3.9.4  Latency and Throughput

For deployment-readiness the end-to-end latency is decomposed as

*T_e2e = T_capture + T_MP + T_net + T_infer + T_render*    (Eq. 3.18)

where T_capture is the camera-to-RGB time, T_MP is the MediaPipe hand
detection cost, T_net is the network round-trip (zero for the on-device
mobile pipeline), T_infer is the model forward pass, and T_render is
the browser/app paint cost. The frame-rate is the inverse of T_e2e
and is reported per-device in Chapter 5.

---

*End of Chapter 3 draft.* Estimated length in Word with the AASTMT
formatting: approximately 16–18 pages (including 5 inserted figures
and 2 inserted tables). The next chapter (Implementation) draws on
this methodology to describe the FastAPI backend, React web client,
Node.js authentication service, React-Native mobile client, and
Docker-based cloud deployment.
