# Pre-registration: 5G-NIDD federated over eight Dirichlet-split clients, with a 1D-CNN baseline (2026-11-27)

Written and committed before any partition was drawn or any model trained (Round 86, batch 8), on the owner's choice of
2026-11-27 to take both suggestions of the model review of 2026-11-26 (`references/model-choice-vs-2026-practice.md`),
each pre-registered. Nothing here is changed after a figure is seen; a change is an addendum with its date and reason.

## Why

The page's federation (`references/telecom-nidd-federated-preregistration.md`, record
`docs/measurements/telecom-nidd-federated-2026-11-26.json`) trains a linear model across the corpus's two real base
stations. The review found the 5G-NIDD federated literature evaluates otherwise: Jeong, Park, Hong and Seo, "A
Comprehensive Evaluation of Distributed Learning Frameworks in AI-Driven Network Intrusion Detection", CMC 87(1)
(published 2026-02-10), section 5.3.1, use eight local nodes, "the number of attack classes", whose class proportions are
drawn from a Dirichlet distribution at alpha = 0.3, and a network of a linear projection, two 1D convolutions and fully
connected layers, which reaches accuracy 0.9711 under federated learning in their non-IID scenario. This runs that set-up
beside the two stations. It changes nothing the page ships.

## Data and split

The federation's, exactly: `D.load_5g_nidd()`, its 36 features, nine classes, the student's split (75/25 stratified at
seed 42, then a stratified 10 percent of the training part at seed 0 as validation), and the standardisation the
federation computes (each client's row count, feature means and population deviations, exchanged in the clear and
combined by the parallel formula; clipped to $`[-20, 20]`$). The eight clients partition the fit and validation rows of
both stations together; the test rows are each station's, as the federation's.

## The partition

For each class, its fit rows are shuffled and divided among the eight clients in proportions drawn from
$`\mathrm{Dir}(0.3 \cdot \mathbf{1}_8)`$, and its validation rows in the same proportions. This is the heterogeneous
partition of Yurochkin et al., "Bayesian Nonparametric Federated Learning of Neural Networks", ICML 2019
(arXiv:1905.12022), section 4: "simulating $`p_k \sim \mathrm{Dir}_J(0.5)`$ and allocating a $`p_{k,j}`$ proportion of the
instances of class $`k`$ to batch $`j`$", which leaves both the batch sizes and the class proportions unbalanced. It is this
project's reading of Jeong et al., who draw "per-node class proportions from a Dirichlet distribution" so that "both the
data volume and class composition differ across local nodes" and do not give the procedure; the per-client draw of Hsu,
Qi and Brown (arXiv:1909.06335) would keep the clients' sizes equal. Partition seeds 1 to 5, numpy's
default generator. A partition leaving a client fewer than 100 fit rows is drawn again from the seed
$`100 s + k`$ for $`k = 1, 2, \ldots`$, and every redraw is recorded. Jeong et al. also drop a quarter of the nodes from
each round at random; that is not done here, since the page's masks do not survive a client that drops (the double
masking of Bonawitz et al. is not implemented).

## The arms, each for each of the five seeds

1. **Linear, FedAvg under pairwise masks, eight clients.** The federation's model and procedure, unchanged but for the
   number of clients: multinomial logistic regression on the 36 inputs from zeros; the penalty $`\lambda`$ from
   $`10^{-6}`$ to $`10^{-1}`$ by decades; 40 rounds, in each of which every client runs 20 iterations of L-BFGS-B on its own
   fit rows from the global model and sends its parameters weighted by its share of the fit rows, quantised to
   $`2^{-20}`$ and masked by `qsh.ai.federated.SecureAggregator` with eight clients; the penalty chosen by the pooled
   validation log loss of each run's final model and the temperature fitted by the same bounded search on
   $`\log T \in [-4, 4]`$, each evaluation a masked sum of the clients' validation losses weighted by their shares.
2. **1D-CNN, FedAvg in the clear, eight clients.** This project's reading of Jeong et al.'s network: a linear layer from
   the 36 inputs to 128 values, read as a sequence of length 128 with one channel; a 1D convolution of 128 filters and a
   1D convolution of 64 filters, both of kernel 3, each followed by ReLU; max pooling of width 2; flattening; a fully
   connected layer of 128 with ReLU; a fully connected layer to the nine classes. Each round every client trains one
   local epoch from the global model (Adam, learning rate $`10^{-3}`$, batch 256, unweighted cross-entropy, initialisation
   from the seed), and the next global model is the average of the clients' models weighted by their shares of the fit
   rows; 30 rounds. The round reported is the one with the best pooled validation macro-F1.
3. **1D-CNN pooled.** The same network and optimiser on every fit row for 30 epochs, the epoch reported chosen the same
   way: the centralised reference for arm 2.
4. **References already recorded**, not rerun: the two-station federated model, each station alone, and pooling, from
   the federation's record.

## Metrics

For every arm and seed, as the federation reports them: macro-F1 over the nine classes on each station's test rows, and
their mean (FLamby's metric); macro-F1, weighted F1 and accuracy on the union of the two stations' test rows (the
student's test split). Weighted F1 is added because the 5G-NIDD federated papers report it. For each arm, the mean and
standard deviation over the five seeds.

## Stated before any figure

- **C1.** Whether the eight-client linear model's mean FLamby metric is within 0.01 of pooling's (the federation's
  criterion 2, on the harder split); the gap is stated either way.
- **C2.** The eight-client 1D-CNN's mean FLamby metric minus the eight-client linear model's, with a 95 percent t
  interval over the five paired seeds, stated as measured.

Neither decides what ships. The page's encrypted federated model stays the two-station linear one whatever these show,
since no network here is evaluated under CKKS. Every figure is shown on the federated card beside the two stations, and
kept in `docs/measurements/telecom-nidd-dirichlet-2026-11-27.json`.

## What the masks hide here

With eight clients, each client learns from the sum only the sum of the other seven, so the masks hide one client's
update from each other client unless the six others collude, as well as from the aggregator; with the page's two, each
station could compute the other's. The seeds still come from X25519 alone, which is not post-quantum; the masked updates
are not sealed; one process plays every client and the aggregator; no differential privacy is applied; and the 1D-CNN
arms are not masked at all, being baselines.

## Compute

The linear arm on the CPU. The 1D-CNN arms on the GPU if the owner approves the job, else on the CPU; the device is
recorded and changes no rule above.
