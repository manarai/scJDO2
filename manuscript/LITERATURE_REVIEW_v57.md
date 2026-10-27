# Literature review for the scJDO2 / manuscript v57 novelty statement

Search date: **2026-10-26**. Every citation in sections 1–5 was checked against Crossref (`api.crossref.org/works/<DOI>`), Europe PMC, the bioRxiv API or the arXiv API. Title, authors, venue and year were confirmed on that record. Where a paper's content is described beyond its title, the description comes from its abstract and, where marked "FT", from a keyword scan of the open-access full text (Europe PMC). Summaries are in our own words. Anything I could not verify is in section 6 and must not be cited.

**Note on the two DOIs found in the authors' files (area G):**
- `10.1101/2025.09.12.674509` is **not** scJDO. It is Balubaid A, Bernal Tamayo JP, Kumar R, Agboola H, Gomez Cabrero D, Kiani N, Tegner J. *Fundamental Limits of Inferring Dynamical Gene Regulatory Models from Single-Cell Data.* bioRxiv, posted 2025-09-12. It is relevant prior work (see B and section 4).
- `10.1101/2025.02.05.636568` **does not resolve**: doi.org returns 404, Crossref returns 404, and the bioRxiv API has no record. An old v56 notebook (`concordance_04_compare.ipynb`) cites it as "Niemiec et al. 2025 bioRxiv". I could not find that paper. It is listed under unverified leads.
- **The scJDO v1 preprint itself** is: Redd DM, Green SG, Terooatea TW. *Time-resolved operator archetypes characterize dynamical sensitivity during cell-state transitions.* bioRxiv, posted **2026-08-23** (v1, Bioinformatics), DOI **10.64898/2026.08.21.745996** (bioRxiv's new DOI prefix; it resolves via doi.org to biorxiv.org). Its abstract establishes the authors' priority for (a) "operator archetypes" (time-indexed Jacobians projected into a shared subspace and decomposed into recurrent archetypes), (b) a drift learned by diffusion score matching from cell-state geometry, and (c) an explicit statement of what snapshot-fitted drifts can and cannot recover (a density-dominant separation, not the transverse instability, at a fate saddle).

---

## 1. Summary verdict

- **Already done by others (do not claim):** scalar landscapes and potentials from scRNA-seq (Wang; scEpath; PBA; Topographer; PRESCIENT; PI-SDE; ddHodge; Cislo 2025). Per-cell Jacobians of velocity fields, with divergence, curl and eigenvalues (dynamo; ddHodge; spliceJAC; scHopfield docs). The leading Jacobian eigenvalue approaching 0 or turning positive at transition cells (DNB theory; spliceJAC; Barcenas 2024; Freedman 2023; CellTran). Velocities from time-course OT (OTVelo; TrajectoryNet, MIOFlow, TIGON, DeepRUOT, DiffusionOT). Jacobians of OT-learned drifts, used for GRNs (TIGON; DiffusionOT). Gradient-plus-flux decomposition and entropy production from scRNA-seq (Zhu et al. 2024). A score and a drift learned together from time courses (DiffusionOT; DeepRUOT; SF2M; PFI). Eigenvalue-level identifiability of Jacobians from scRNA-seq (Balubaid 2025). Snapshot non-identifiability (Weinreb 2018; Coomer 2022).
- **Partially done:**
  - A per-cell **Hessian of a potential** exists in ddHodge, but its potential is the gradient part of an RNA-velocity field, not the log-density.
  - **Operator embeddings** exist in Cell-MNN, which shows a UMAP of per-cell linear operators with no clustering, eigen-analysis or archetypes.
  - **Score + flow in one model** exists in DiffusionOT, DeepRUOT and PFI. Each uses it at the level of a global field or decomposition, never as per-cell operators.
  - **Operator archetypes** are the authors' own prior work (scJDO v1, 2026-08-23).
- **Not found as of 2026-10-26:**
  1. A per-cell Hessian of a score-matched log-density in a cell-state embedding, read as stiff/soft directions, with only its inertia treated as invariant.
  2. Per-cell, similarity-invariant eigen-descriptors (leading eigenvalue for branching, rotational index for cycles) of a drift fitted to an entropic-OT barycentric velocity, evaluated against lineage barcodes.
  3. Both operators in one basis-invariant descriptor space, with archetypes and their per-cell disagreement used as a non-equilibrium read-out.
  4. A pre-registered, multi-axis identifiability analysis (representation, clock, rotation, stationarity, diffusion, composition) of such operators.
- **Defensible novelty claim (narrow wording):** "To our knowledge, scJDO2 is the first framework that computes, per cell, both a symmetric score operator (the Hessian of a score-matched log-density, used only through its inertia) and a flow operator (the Jacobian of a drift fitted to a measured velocity, including a label-free entropic-OT velocity), places them in one basis-invariant operator space with archetypes, and tests against independent truth, with pre-registered gates, what such operators can and cannot identify. Each ingredient on its own (score matching, OT velocities, velocity-field Jacobians and their eigenvalues, landscape–flux decomposition) has precedent, which we cite."

---

## 2. Tables per area

Columns: **(i)** scalar potential/energy · **(ii)** per-cell operator (Jacobian/Hessian) · **(iii)** basis-invariant descriptors (eigenvalues, inertia, trace, etc.) used as per-cell features · **(iv)** archetypes/patterns of operators · **(v)** pairing of density-derived and flow-derived quantities · **(vi)** independent validation reported (per abstract or FT). Y = yes, P = partial, N = not found, ? = not checked.

### A. Landscapes, energies and potentials from single-cell data

| Paper (verified) | What it computes | Data | i | ii | iii | iv | v | vi |
|---|---|---|---|---|---|---|---|---|
| Wang J, Xu L, Wang E. Potential landscape and flux framework of nonequilibrium networks… *PNAS* 2008. doi:10.1073/pnas.0800579105 | U = −ln P_ss plus a curl flux. Drift = gradient part + flux/P_ss. Dissipation. | Model ODE/SDE (not scRNA-seq) | Y | N | N | N | Y (theory) | model only |
| Wang J, Zhang K, Xu L, Wang E. Quantifying the Waddington landscape and biological paths… *PNAS* 2011. doi:10.1073/pnas.1017017108 | Landscape and paths for a differentiation GRN model | Model | Y | N | N | N | Y (theory) | model only |
| Li C, Wang J. Landscape and flux reveal a new global view… mammalian cell cycle. *PNAS* 2014. doi:10.1073/pnas.1408628111 | Landscape and flux for a cell-cycle model | Model | Y | N | N | N | Y | model only |
| Zhu L, Yang S, Zhang K, Wang H, Fang X, Wang J. Uncovering underlying physical principles and driving forces of cell differentiation and reprogramming from single-cell transcriptomics. *PNAS* 2024;121:e2401540121. doi:10.1073/pnas.2401540121 | Fits a dynamo vector field to RNA velocity, then landscape U and flux, least-action paths, MFPT, entropy production rate, loop flux (FT: no Hessian or eigenvalue analysis found) | scRNA-seq + RNA velocity | Y | N (Jacobian only for GRN) | N | N | **Y (global landscape vs flux)** | ? |
| Jin S, MacLean AL, Peng T, Nie Q. scEpath… *Bioinformatics* 2018. doi:10.1093/bioinformatics/bty058 | Per-cell "scEnergy" scalar; transition probabilities | Snapshot | Y | N | N | N | N | ? |
| Zhang J, Nie Q, Zhou T. Revealing dynamic mechanisms of cell fate decisions from single-cell transcriptomic data (Topographer). *Front Genet* 2019. doi:10.3389/fgene.2019.01280 | Per-cell potential + pseudotime; fate and transition probabilities | Snapshot | Y | N | N | N | N | ? |
| Weinreb C, Wolock S, Tusi BK, Socolovsky M, Klein AM. Fundamental limits on dynamic inference from single-cell snapshots (PBA). *PNAS* 2018. doi:10.1073/pnas.1714723115 | Population balance analysis (potential-like, under stated assumptions); states what snapshots cannot identify | Snapshot | Y | N | N | N | N | Y (consistent with published HPC fate assays) |
| Shi J, Teschendorff AE, Chen W, Chen L, Li T. Quantifying Waddington's epigenetic landscape: a comparison of single-cell potency measures. *Brief Bioinform* 2018. doi:10.1093/bib/bby093 | Compares potency scalars as landscape height | Snapshot | Y | N | N | N | N | benchmark |
| Shi J, Li T, Chen L, Aihara K. Quantifying pluripotency landscape… continuous birth-death process. *PLoS Comput Biol* 2019. doi:10.1371/journal.pcbi.1007488 | Landscape from a birth–death model | Snapshot | Y | N | N | N | N | ? |
| Shi J, Aihara K, Li T, Chen L. Energy landscape decomposition for cell differentiation with proliferation effect. *Natl Sci Rev* 2022. doi:10.1093/nsr/nwac116 | Landscape decomposed with growth | Snapshot | Y | N | N | N | P | ? |
| Teschendorff AE, Enver T. Single-cell entropy… differentiation potency (SCENT). *Nat Commun* 2017. doi:10.1038/ncomms15599 | Per-cell signalling entropy (potency) | Snapshot + PPI network | Y (potency) | N | N | N | N | Y (potency labels) |
| Gulati GS et al. Single-cell transcriptional diversity is a hallmark of developmental potential (CytoTRACE). *Science* 2020. doi:10.1126/science.aax0249; Kang M et al. CytoTRACE 2. *Nat Methods* 2025. doi:10.1038/s41592-025-02857-2 | Per-cell potency scalar | Snapshot | Y (potency) | N | N | N | N | Y |
| Li H et al. Characterizing cellular differentiation potency and Waddington landscape via energy indicator. *Research* 2023. doi:10.34133/research.0118 | Hopfield-type energy per cell | Snapshot | Y | N | N | N | N | ? |
| Fard AT, Srihari S, Mar JC, Ragan MA. Not just a colourful metaphor… Hopfield networks. *npj Syst Biol Appl* 2016. doi:10.1038/npjsba.2016.1; Guo J, Zheng J. HopLand. *Bioinformatics* 2017. doi:10.1093/bioinformatics/btx232 | Hopfield energy landscapes | Snapshot | Y | N | N | N | N | ? |
| Zhou P, Wang S, Li T, Nie Q. MuTrans. *Nat Commun* 2021. doi:10.1038/s41467-021-25548-w | Multiscale stochastic dynamics; separates stable from transition cells | Snapshot | P | N | N | N | N | ? |
| Yeo GHT, Saksena SD, Gifford DK. PRESCIENT. *Nat Commun* 2021. doi:10.1038/s41467-021-23518-w | Neural potential Φ with drift = −∇Φ, fitted to a **time course** | Time course (+ lineage for evaluation) | Y | N (its Jacobian would be a Hessian, but this is not used) | N | N | N | Y (lineage fate bias) |
| Jiang Q, Wan L. PI-SDE. *Bioinformatics* 2024;40:ii120. doi:10.1093/bioinformatics/btae400 | Neural SDE with a Hamilton–Jacobi-regularised potential | Time course | Y | N | N | N | N | held-out time points |
| Huizing GJ et al. STORIES. *Nat Methods* 2026;23:522. doi:10.1038/s41592-025-02855-4 | Spatially informed potential (Wasserstein gradient flow) | Spatial time course | Y | N | N | N | N | ? |
| Maehara K, Ohkawa Y. Modeling latent flows on single-cell data using the Hodge decomposition. bioRxiv 2019. doi:10.1101/592089; **ddHodge**: Geometry-preserving vector field reconstruction… *Nat Commun* 2025;16:11342. doi:10.1038/s41467-025-67782-6 | From RNA velocity: potential (gradient part), curl, divergence, **per-point Hessian of U**, Jacobian = Hessian + curl; divergence as per-cell stability; Schur vectors at chosen points (FT) | scRNA-seq + RNA velocity | Y | **Y (Hessian of velocity-potential and Jacobian)** | P (divergence, det J) | N | P (gradient vs curl of the *same* velocity field, not density vs flow) | sim + biological interpretation |
| Su Z, Tong Y, Wei GW. Hodge decomposition of single-cell RNA velocity. *J Chem Inf Model* 2024;64:3558. doi:10.1021/acs.jcim.4c00132 | Curl-free / divergence-free / harmonic split of velocity in 2-D embeddings | RNA velocity | P | N | N | N | P | N |
| Sáez M et al. Statistically derived geometrical landscapes… *Cell Syst* 2022;13:12. doi:10.1016/j.cels.2021.08.013 | Catastrophe-theory (Morse–Smale) landscapes fitted by ABC | Flow cytometry, signalling time courses | Y | P (normal forms) | N | N | N | Y (predicted fate outcomes) |
| Rand DA, Raju A, Sáez M, Corson F, Siggia ED. Geometry of gene regulatory dynamics. *PNAS* 2021. doi:10.1073/pnas.2109729118 | Theory of gradient-like landscapes | Theory | Y | P | N | N | N | — |
| Cislo DJ, Delás MJ, Briscoe J, Siggia ED. Reconstructing Waddington's landscape from data. *PNAS* 2025;122:e2521762122. doi:10.1073/pnas.2521762122 (bioRxiv 10.1101/2025.08.11.669575) | Fits dynamical landscapes to high-dimensional time-evolving distributions; fixed points, unstable manifolds, basins; Hessian checks at sinks and saddles | Flow cytometry and RNA-seq time courses | Y | P (Hessian at fixed points, not per cell) | P (saddle index) | N | N | Y (morphogen-dependent fates) |
| Mochulska V, François P. Generative epigenetic landscapes map the topology and topography of cell fates. *PNAS* 2025. doi:10.1073/pnas.2514508122 | Evolves gradient + rotational landscape models | Developmental data | Y | N | N | N | P (gradient + rotation) | ? |
| Fontaine M et al. Dynamic landscape analysis of cell fate decisions… bioRxiv 2025 (v2 2025-12-16). doi:10.1101/2025.05.28.656648 | Dynamical-systems landscape (DLA) from scRNA-seq; predicts unseen signalling regimes | scRNA-seq + signalling | Y | P | N | N | N | Y (unseen regimes; organoids) |
| Coomer MA, Ham L, Stumpf MPH. Noise distorts the epigenetic landscape… *Cell Syst* 2022;13:83. doi:10.1016/j.cels.2021.09.002 | Shows −log P_ss ≠ deterministic potential under noise; identifiability limits | Models | — | — | — | — | (caution) | — |
| Zhou JX, Aliyu MDS, Aurell E, Huang S. Quasi-potential landscape in complex multi-stable systems. *J R Soc Interface* 2012. doi:10.1098/rsif.2012.0434 | Quasi-potential for non-gradient systems | Models | Y | N | N | N | P | — |
| Brackston RD, Lakatos E, Stumpf MPH. Transition state characteristics during cell differentiation. *PLoS Comput Biol* 2018. doi:10.1371/journal.pcbi.1006405 | Transition states in non-gradient landscapes | Models | Y | N | N | N | P | — |
| Han Y, Chen B, Zhang J, Shang X. Beyond metaphor: quantitative reconstruction of Waddington landscape… *Brief Bioinform* 2025. doi:10.1093/bib/bbaf661 | Review | — | — | — | — | — | — | — |

**Score-matching / diffusion-model landscapes (A, specific).** No single-cell paper was found that fits a neural energy by score matching to a cell-state embedding and uses the **per-cell Hessian** of the log-density as a stability operator. Closest:
- DiffusionOT (below, C) parameterises the score ∇log p and reconstructs an energy landscape from it.
- Tong et al. (SF2M, arXiv:2307.03672) combine score and flow matching and illustrate a Waddington-type potential.
- scJDO v1 (2026) used diffusion score matching to learn a drift.
- Theory: Alain & Bengio (arXiv:1211.4246) relate an autoencoder's Jacobian to the log-density Hessian. Yakovlev, Markovich & Puchkin (arXiv:2512.24378, 2025) give rates for log-density Hessian estimation via score matching. Hyvärinen, *Estimation of non-normalized statistical models by score matching*, JMLR 6 (2005), and Vincent, *A connection between score matching and denoising autoencoders*, *Neural Comput* 2011, doi:10.1162/NECO_a_00142, are the method citations.

### B. Jacobians and local stability of single-cell vector fields

| Paper (verified) | What it computes | Data | i | ii | iii | iv | v | vi |
|---|---|---|---|---|---|---|---|---|
| Qiu X et al. Mapping transcriptomic vector fields of single cells (**dynamo**). *Cell* 2022;185:690. doi:10.1016/j.cell.2021.12.045 | Analytic vector field from velocity; per-cell Jacobian, divergence, curl, acceleration, curvature; fixed points classified by eigenvalue signs (FT); least-action paths; in-silico perturbation; potential | Splicing or metabolic-labelling velocity | Y | **Y (per-cell Jacobian)** | P (divergence/curl per cell; eigen-signs only at fixed points) | N | N | Y (perturbation predictions vs experiments) |
| Bocci F, Zhou P, Nie Q. **spliceJAC**. *Mol Syst Biol* 2022;18:e11176. doi:10.15252/msb.202211176 | Cluster/state-level Jacobian from a splicing model; eigen-spectrum; positive eigenvalue marks unstable states; gene "instability score" (FT) | Spliced/unspliced counts + state labels | N | P (per state, not per cell) | **Y (eigen-spectrum)** | N | N | sim + literature |
| Barcenas M, Bocci F, Nie Q. Tipping points in epithelial-mesenchymal lineages from single-cell transcriptomics data. *Biophys J* 2024;123:2849. doi:10.1016/j.bpj.2024.03.021 | Splicing-model Jacobian along a lineage ordered by pseudotime; spectral (eigenvalue) analysis locates an unstable intermediate (saddle-node) | Spliced/unspliced + ordering | N | P (windows along lineage) | **Y (leading eigenvalue)** | N | N | ? |
| Balubaid A et al. Fundamental limits of inferring dynamical gene regulatory models from single-cell data. bioRxiv 2025. doi:10.1101/2025.09.12.674509 | Benchmarks spliceJAC, scMomentum and dynamo by Jacobian **eigenvalues** (stability, oscillation) on ground truth. Behaviour is captured, structure (entries) is not. | Simulations | — | evaluated | evaluated | N | N | sim ground truth |
| Soto L et al. scMomentum: inference of cell-type-specific regulatory networks and energy landscapes. bioRxiv 2020. doi:10.1101/2020.12.30.424887 | Cell-type GRN + energy from velocity | RNA velocity | Y | P | N | N | N | ? |
| Wang W et al. RegVelo: gene-regulatory-informed dynamics of single cells. *Cell* 2026. doi:10.1016/j.cell.2026.04.022 (bioRxiv 10.1101/2024.12.11.627935) | GRN-coupled velocity model; perturbation | Splicing + GRN prior | N | P | ? | N | N | ? (not checked) |
| Chen Y et al. GraphVelo… *Nat Commun* 2025. doi:10.1038/s41467-025-62784-w | Manifold-consistent multimodal velocities; downstream dynamo-style analysis | Multimodal velocity | N | P | ? | N | N | ? |
| Karbalayghareh A et al. DynaVelo. *Nat Biotechnol* 2026. doi:10.1038/s41587-026-03348-1 | Neural-ODE vector field on multiome velocities; in-silico perturbation GRNs | Multiome | N | P | N | N | N | **Y (in vivo rescue experiments)** |
| Burdziak C et al. scKINETICS. *Bioinformatics* 2023. doi:10.1093/bioinformatics/btad267 | Regulatory velocity with linear dynamics | Snapshot + priors | N | P | N | N | N | ? |
| Hossain I et al. PHOENIX (biologically informed NeuralODEs). *Genome Biol* 2024. doi:10.1186/s13059-024-03264-0 | NeuralODE GRN dynamics | Time series | N | P | N | N | N | ? |
| Dou J, Lyu W, Chen F, Nie Q, Li C. DynNet. *Nat Commun* 2026;17:6901. doi:10.1038/s41467-026-73257-z | Biophysical neural ODE + noise; stable states, landscape, transition probabilities | scRNA-seq | Y | P (stability of states) | P | N | N | sim |
| Freedman SL, Xu B, Goyal S, Mani M. A dynamical systems treatment of transcriptomic trajectories in hematopoiesis. *Development* 2023;150:dev201280. doi:10.1242/dev.201280 | Statistical signatures of bifurcations along trajectories; uses linear-instability geometry to find the bifurcation plane | Snapshot (pseudotime) | N | P | P | N | N | ? |
| Wang Y et al. CellTran: a statistical approach for systematic identification of transition cells from scRNA-seq data. *Cell Rep Methods* 2024. doi:10.1016/j.crmeth.2024.100913 | Per-cell transition score from gene-pair correlations (DNB-like) | Snapshot | N | N | N | N | N | sim (SERGIO) + biology |
| Sritharan D, Wang S, Hormoz S. Computing the Riemannian curvature… scRNA-seq data manifolds. *PNAS* 2021. doi:10.1073/pnas.2100473118 | Per-point **manifold** (geometric) curvature, not density curvature | Snapshot | N | P (geometric) | Y (scalar curvature) | N | N | ? |
| von Bassewitz J-P et al. Learning explicit single-cell dynamics using ODE representations (Cell-MNN). arXiv:2510.02903 (2025) | Per-state **locally linear operator** A(z,t) in latent space; gene interactions checked against TRRUST; UMAP of operators shown, no formal clustering, no eigenvalue or stability analysis (FT via arXiv HTML) | Time course (no OT) | N | **Y (per-cell linear operator)** | N | P (visual only) | N | TRRUST |
| Chen Z et al. DeepVelo. *Sci Adv* 2022. doi:10.1126/sciadv.abq3745 | Neural-ODE velocity field; instability by simulation | RNA velocity | N | P | N | N | N | ? |
| Lange M et al. CellRank. *Nat Methods* 2022. doi:10.1038/s41592-021-01346-6; Weiler P et al. CellRank 2. *Nat Methods* 2024. doi:10.1038/s41592-024-02303-9 | Markov-chain macrostates and fate probabilities (no Jacobians); CellRank 2 accepts OT/real-time kernels | Velocity / OT / pseudotime | N | N | N | P (macrostates of cells, not of operators) | N | Y |
| La Manno-type velocity: Bergen V et al. scVelo. *Nat Biotechnol* 2020. doi:10.1038/s41587-020-0591-3 | RNA velocity (input to the above) | Splicing | — | — | — | — | — | — |

Also seen (software documentation only, no paper verified): **scHopfield** (schopfield.readthedocs.io). It fits a Hopfield GRN to RNA velocity and documents per-cell Jacobian eigenvalues, rotational (antisymmetric) parts and Hopfield energy. It is close to "per-cell eigenvalues + energy" but not citable as a paper (see section 6).

### C. Optimal-transport flows and learned dynamics from time courses

| Paper (verified) | What it computes | Data | Differentiates drift → per-cell Jacobian/stability? | i | ii | iii | iv | v | vi |
|---|---|---|---|---|---|---|---|---|---|
| Schiebinger G et al. Waddington-OT. *Cell* 2019;176:928. doi:10.1016/j.cell.2019.02.026 | Entropic unbalanced OT couplings between days; ancestors/descendants, fates | Time course | N | N | N | N | N | N | Y (reprogramming biology) |
| Klein D et al. moscot. *Nature* 2025;638:1065. doi:10.1038/s41586-024-08453-2 | Scalable multimodal OT across time and space | Time course / spatial | N | N | N | N | N | N | Y |
| Forrow A, Schiebinger G. LineageOT. *Nat Commun* 2021. doi:10.1038/s41467-021-25133-1 | OT + lineage-tree information | Time course + lineage | N | N | N | N | N | N | Y |
| Lavenant H, Zhang S, Kim Y-H, Schiebinger G. Toward a mathematical theory of trajectory inference. *Ann Appl Probab* 2024. doi:10.1214/23-AAP1969 (arXiv:2102.09204) | Identifiability of potential-driven SDE trajectories from marginals; gWOT | Time course | N | — | — | — | — | — | theory |
| Zhao W, Larschan E, Sandstede B, Singh R. **OTVelo**. *PLoS Comput Biol* 2025. doi:10.1371/journal.pcbi.1012476 | **Label-free velocity from OT-predicted past/future states (finite difference)**; GRN by lagged correlation / Granger | Time course | N (no Jacobian) | N | N | N | N | N | ? |
| Tong A et al. TrajectoryNet. arXiv:2002.04461 (2020) | Dynamic-OT continuous normalising flow | Time course | N | N | N | N | N | N | ? |
| Huguet G et al. MIOFlow. arXiv:2206.14928 (2022) | Manifold-interpolating OT flows (neural ODE in a latent space) | Time course | N | N | N | N | N | N | ? |
| Bunne C, Meng-Papaxanthos L, Krause A, Cuturi M. JKOnet: proximal OT modeling of population dynamics. arXiv:2106.06345 (2021) | Learns an energy whose JKO flow fits the snapshots | Time course | N | Y | N | N | N | N | ? |
| Tong A et al. Simulation-free Schrödinger bridges via score and flow matching (SF2M). arXiv:2307.03672 (2023) | Learns flow + score from entropic OT; Waddington-style potential illustration | Time course | N | P | N | N | N | **P (score + flow learned jointly)** | ? |
| Sha Y, Qiu Y, Zhou P, Nie Q. **TIGON**. *Nat Mach Intell* 2024. doi:10.1038/s42256-023-00763-w | Dynamic unbalanced OT (WFR); velocity + growth; **GRN from the Jacobian of the learned velocity** | Time course | **Y (Jacobian, used for GRN; no eigen/stability)** | N | Y | N | N | N | ? |
| Zhang Z, Li T, Zhou P. Learning stochastic dynamics from snapshots through regularized unbalanced OT (**DeepRUOT**). arXiv:2410.00844 (2024) | Velocity + growth + score (Schrödinger-bridge-type RUOT) | Time course | ? (not checked) | P | ? | N | N | P | ? |
| Liu J, Zhou P, Nie Q, Li C. **DiffusionOT**: Learning stochastic dynamics and cell-fate landscapes from single-cell snapshots via OT. *Sci Adv* 2026;12:eaeb4205. doi:10.1126/sciadv.aeb4205 | Velocity, growth and **score ∇log p** as networks; energy landscape from log-density; **Jacobian of velocity, averaged over cells for a GRN** (FT: no eigenvalue analysis) | Time course (incl. Stereo-seq) | **P (Jacobian yes; per-cell eigen/stability no)** | Y | Y (averaged) | N | N | **P (density and velocity in one model; not compared as operators)** | sim + 4 real datasets |
| Chardès V, Maddu S, Shelley MJ. Stochastic force inference via density estimation. arXiv:2310.02366 (2023); Maddu S, Chardès V, Shelley MJ. Inferring biological processes with intrinsic noise from cross-sectional data (**PFI**). arXiv:2410.07501 (2024); Zhang SY, Maddu S, Qiu X, Chardès V. Inferring stochastic dynamics with growth from cross-sectional data (UPFI). arXiv:2505.13197 (2025); Maddu, Chardès, Shelley. Probability flow matching. arXiv:2604.25062 (2026) | Score matching separates the force from intrinsic noise via the probability flow (force = probability-flow velocity + D∇log p); recovers non-conservative forces from non-stationary data | Time course | ? (GRN interactions; per-cell eigen not checked) | N | P | N | N | **Y (score + force decomposition, global)** | sim + haematopoiesis |
| Vinyard ME et al. scDiffEq: Learning cell dynamics with neural differential equations. *Nat Mach Intell* 2025. doi:10.1038/s42256-025-01150-3 | Neural SDE drift + diffusion; fate prediction on LARRY; perturbation | Time course + lineage | N | N | N | N | N | N | Y (LARRY fates, CRISPR perturbation) |
| Taylor-King JP, Riseth AN, Macnair W, Claassen M. Dynamic distribution decomposition… *PLoS Comput Biol* 2020. doi:10.1371/journal.pcbi.1007491 | Linear (Koopman/Perron–Frobenius-type) operator on distributions | Time course | N (operator on densities, not per cell) | N | N | N | N | N | ? |
| Zhang Z, Wang Z, Sun Y, Shen J et al. Deciphering cell-fate trajectories using spatiotemporal single-cell transcriptomic data. *npj Syst Biol Appl* 2025. doi:10.1038/s41540-025-00624-9; Zhang Z et al. Integrating dynamical systems modeling with spatiotemporal scRNA-seq data analysis. arXiv:2503.11347 | Reviews of OT / score / flow-matching dynamics | — | — | — | — | — | — | — | — |

**Answer to "does any of them differentiate the learned drift to get per-cell Jacobians or stability?"** TIGON and DiffusionOT differentiate the OT-learned velocity, but use it for GRN edges (DiffusionOT averages across cells). Neither reports per-cell eigenvalue-based stability, a branching marker or a cycle index. DynNet analyses the stability of attractor states. I did not find a time-course OT method that reports per-cell Jacobian eigenvalues as a decision-cell marker validated against lineage.

### D. Operator archetypes / clustering cells by local operators

| Paper | What | ii | iii | iv | Note |
|---|---|---|---|---|---|
| **Redd DM, Green SG, Terooatea TW.** Time-resolved operator archetypes… *bioRxiv* 2026-08-23. doi:10.64898/2026.08.21.745996 | Time-indexed Jacobians of a DSM drift → shared subspace → recurrent **operator archetypes** | Y | P | **Y** | Authors' own priority for the term and the idea |
| Cell-MNN (arXiv:2510.02903) | Per-cell latent linear operators; UMAP of operators coloured by markers | Y | N | P (visual only) | Nearest outside work on an "operator space" |
| spliceJAC; Barcenas 2024 | Jacobians per state / along lineage, compared across states | P | Y | N | Comparison across states, not archetypes |
| CellRank / CellRank 2; MuTrans | Macrostates / transition cells (Markov chains) | N | N | P | Patterns of cells, not of operators |
| Sáez 2022 (*Cell Syst*) | "Archetypal designs" of decisions (two catastrophe types) | P | — | P (conceptual) | Different meaning of "archetype" |
| Ding C, Li T, Jordan MI. Convex and semi-nonnegative matrix factorizations. *IEEE TPAMI* 2010;32:45. doi:10.1109/TPAMI.2008.277; Cutler A, Breiman L. Archetypal analysis. *Technometrics* 1994. doi:10.1080/00401706.1994.10485840 | Method citations for semi-NMF / archetypes | — | — | — | cite as methods |

No other single-cell work using "operator archetypes" or "Jacobian archetypes" was found (exact-phrase search, 2026-10-26).

### E. Critical transitions / early warning (context only)

| Paper (verified) | What | Relation |
|---|---|---|
| Chen L, Liu R, Liu Z-P, Li M, Aihara K. Dynamical network biomarkers. *Sci Rep* 2012;2:342. doi:10.1038/srep00342 | Variance/correlation surge as the dominant eigenvalue → 0 | Theory behind "leading eigenvalue marks transition" |
| Mojtahedi M et al. Cell fate decision as high-dimensional critical state transition. *PLoS Biol* 2016. doi:10.1371/journal.pbio.2000640 | Critical-transition index in single-cell data | Snapshot covariance indicator |
| Yang XH et al. BioTIP: Detecting critical transition signals from single-cell transcriptomes to infer lineage-determining transcription factors. *Nucleic Acids Res* 2022;50:e91. doi:10.1093/nar/gkac452 | Tipping-point / CTS genes | Runnable (Bioconductor) |
| CellTran (Wang Y et al. 2024, above); Barcenas 2024 (above); Freedman 2023 (above) | Per-cell or along-lineage transition / instability | Closest per-cell analogues of λ₁ |
| Barter E, Brechtel A, Drossel B, Gross T. A closed form for Jacobian reconstruction from timeseries and its application as an early warning signal in network dynamics. arXiv:1910.09698 | Jacobian from fluctuations as an early-warning signal | General systems, not single-cell |
| Wang Y et al. Sloppiness and action constraint in cell state transitions: are single cells sloppy? bioRxiv 2026 (v2 2026-02-22). doi:10.64898/2025.12.31.697145 | Fisher-information stiff/sloppy directions during transitions | Closest use of "stiff vs soft" language in single-cell (parameters, not density Hessian) |

### F. Non-equilibrium measures

| Paper (verified) | What | Single-cell? | Pairing density vs flow? |
|---|---|---|---|
| Wang, Xu, Wang 2008 (above); Fang X, Kruse K, Lu T, Wang J. Nonequilibrium physics in biology. *Rev Mod Phys* 2019;91:045004. doi:10.1103/RevModPhys.91.045004 | Landscape–flux theory, dissipation | Review | Y (theory: drift = D∇log P_ss-type gradient + flux/P_ss) |
| Zhu et al. 2024 *PNAS* (above) | Landscape, curl flux, entropy production rate from scRNA-seq velocity | **Y** | Y (global) |
| Yu C, Liu C, Wang E, Wang J. Quantifying the single-cell morphological landscape… force field reconstruction. *Adv Sci* 2025. doi:10.1002/advs.202512325 | Landscape + curl flux + EPR for morphology dynamics | Single-cell morphology (not transcriptome) | Y |
| Boffi NM, Vanden-Eijnden E. Deep learning probability flows and entropy production rates in active matter. *PNAS* 2024;121:e2318106121. doi:10.1073/pnas.2318106121 | Learned score + known dynamics → probability current and **local EPR** | No (active matter) | Y (score-based) |
| Chardès/Maddu/Shelley PFI series (above) | Score-based separation of force and noise | Y (time courses) | Y |
| Battle C et al. Broken detailed balance at mesoscopic scales in active biological systems. *Science* 2016. doi:10.1126/science.aac8167; Gnesotto FS et al. Broken detailed balance and non-equilibrium dynamics in living systems: a review. *Rep Prog Phys* 2018. doi:10.1088/1361-6633/aab3ed | Detailed-balance violation measures | No | Context |
| Godrèche C, Luck JM. Characterising the nonequilibrium stationary states of Ornstein–Uhlenbeck processes. *J Phys A* 2018. doi:10.1088/1751-8121/aaf190 | For linear SDEs: non-equilibrium ⇔ drift not aligned with D × inverse covariance; rotation/flux | No (theory) | **Y (exact linear analogue of score-vs-flow disagreement)** |
| Holehouse J. Quantifying broken detailed balance in transcription. *npj Complexity* 2026. doi:10.1038/s44260-025-00064-w (arXiv:2405.12897) | Entropy production at gene level; flux on observed mRNA can look equilibrium | Gene-level | Caution for marginal observers |
| Dieball C, Godec A. Perspective: time irreversibility in systems observed at coarse resolution. arXiv:2412.02675 | Coarse-graining hides irreversibility | No | Caution |

---

## 3. Closest prior art (top 5) and the precise difference

1. **ddHodge — Maehara & Ohkawa, *Nat Commun* 2025 (doi:10.1038/s41467-025-67782-6).** From RNA velocity it reconstructs a potential, a per-point **Hessian of that potential**, curl and divergence, and the Jacobian as Hessian + curl. It uses divergence as per-cell stability and Schur/eigen directions at chosen points. *Difference:* ddHodge's symmetric operator is the Hessian of the **gradient part of a velocity field**, so it needs velocity. scJDO2's score operator is the Hessian of the **log-density** of a snapshot, needs no velocity, and is used only through its inertia. ddHodge splits one field into gradient and curl. It does not compare a density-derived operator with an independent flow-derived operator, build a basis-invariant descriptor space, find archetypes, or use time-course OT. The manuscript must not claim to be the first per-cell Hessian/curvature of a single-cell landscape.
2. **DiffusionOT — Liu, Zhou, Nie & Li, *Sci Adv* 2026 (doi:10.1126/sciadv.aeb4205)**, with **TIGON** (Sha et al., *Nat Mach Intell* 2024) and **DeepRUOT** (arXiv:2410.00844). These learn a velocity from multi-time-point snapshots by (unbalanced) OT, plus (DiffusionOT, DeepRUOT) a **score network**. They reconstruct an energy landscape from the log-density and take the **Jacobian of the learned velocity** for GRN inference (DiffusionOT averages it across cells). *Difference:* they do not compute per-cell eigenvalue descriptors (λ₁, rotational index), branching markers or archetypes. They do not form the Hessian of the score. Score and velocity serve as parts of one generative model, not as two operators compared per cell. They report no lineage-barcode validation of a per-cell Jacobian read-out. scJDO2's flow pipeline (entropic OT → barycentric velocity → fitted drift → Jacobian) is therefore an **incremental, evaluation-focused** variant of an existing recipe, not a new recipe.
3. **Landscape–flux from scRNA-seq — Zhu et al., *PNAS* 2024 (doi:10.1073/pnas.2401540121)**, building on Wang et al. 2008/2011. It quantifies the Waddington landscape (U ≈ −ln P), the curl flux, least-action paths and the entropy production rate from an RNA-velocity vector field. *Difference:* the quantities are global or path-level fields from one velocity-based model. They are not per-cell operators, no symmetric-vs-asymmetric operator descriptors are built, and there is no OT/time-course or lineage validation. scJDO2's "score vs flow disagreement as a non-equilibrium measure" and "f ≈ D∇log p + flux" are applications of this framework. They must be presented as such, not as new theory.
4. **dynamo (Qiu et al., *Cell* 2022), spliceJAC (Bocci et al., *Mol Syst Biol* 2022) and Barcenas, Bocci & Nie (*Biophys J* 2024).**
   - dynamo gives per-cell Jacobians of a velocity field, with divergence, curl and eigenvalue-sign classification of fixed points.
   - spliceJAC computes state-level Jacobian eigen-spectra, where positive eigenvalues mark unstable/transition states.
   - Barcenas et al. place the tipping point where the leading eigenvalue changes along a lineage.

   *Difference:* their velocities come from splicing or labelling kinetics, and their eigen-analysis works per state, per lineage window or at fixed points. They have no density-derived operator, no basis-invariant descriptor space across cells and no archetypes. When scJDO2's flow operator takes RNA-velocity or labelling input, it **is** a dynamo-style Jacobian. Using λ₁ to mark decision cells is the same idea as spliceJAC/Barcenas (and DNB theory), applied per cell to an OT-derived drift.
5. **Score-based force/flow decomposition — Chardès, Maddu & Shelley (arXiv:2310.02366; PFI arXiv:2410.07501; UPFI arXiv:2505.13197; PFM arXiv:2604.25062), and Boffi & Vanden-Eijnden (*PNAS* 2024).** These use score matching on time-resolved cross-sections to separate the deterministic force from intrinsic noise: force = probability-flow velocity + D∇log p. They recover non-conservative forces, and Boffi & Vanden-Eijnden compute local entropy production from a learned score. *Difference:* these are global field decompositions, not per-cell operator comparisons. They do not build symmetric (Hessian) and non-symmetric (Jacobian) operator descriptors or archetypes. They are the most direct prior art for scJDO2's proposed alignment f ≈ D∇log p (+ flux), and also for its finding that diffusion is hard to identify from marginals.

*Authors' own prior art, cited separately:* scJDO v1 (doi:10.64898/2026.08.21.745996) already introduced operator archetypes and DSM-learned drift Jacobians. v57 must describe archetypes as carried over from v1, and the score/flow split and operator space as new in v2.

*Next tier:* Cell-MNN (per-cell operator embedding), OTVelo (label-free OT velocity), Cislo et al. 2025 (Hessian at landscape fixed points), Balubaid et al. 2025 (eigenvalue-level identifiability of Jacobian methods).

---

## 4. Claims to avoid and claims that are defensible

### Claims to avoid
| Claim | Why | Bounding papers |
|---|---|---|
| "First to reconstruct a Waddington/energy landscape from single-cell data" or "first potential from score matching" | Many prior landscapes; DiffusionOT builds a landscape from a learned score; SF2M | Wang 2011; scEpath; Topographer; PBA; PRESCIENT; PI-SDE; ddHodge; Cislo 2025; DiffusionOT; SF2M |
| "First per-cell Hessian / curvature of the cell-state landscape" | ddHodge estimates per-point Hessians of a velocity-potential; Sritharan computes per-point manifold curvature | ddHodge 2025; Sritharan 2021 |
| "First per-cell Jacobian / local stability from single-cell data" | dynamo, spliceJAC, scHopfield (docs), ddHodge, DynNet | dynamo 2022; spliceJAC 2022; ddHodge 2025 |
| "First label-free velocity/drift from a time course" or "first Jacobian of an OT-learned drift" | OTVelo gives OT velocities; TIGON and DiffusionOT differentiate OT-learned velocity fields | OTVelo 2025; TIGON 2024; DiffusionOT 2026; TrajectoryNet; MIOFlow; PRESCIENT |
| "λ₁ discovers/defines decision cells" as a new principle | The leading eigenvalue approaching 0 or becoming positive at transitions is classical and used in single-cell work | Chen 2012 (DNB); Mojtahedi 2016; spliceJAC 2022; Barcenas 2024; Freedman 2023; CellTran 2024 |
| "New non-equilibrium measure / new landscape–flux theory" | The gradient + flux decomposition, EPR and curl flux exist, including from scRNA-seq and from learned scores | Wang 2008; Fang 2019; Zhu 2024; Boffi 2024; PFI; Godrèche & Luck 2018 |
| "Jacobian eigenvalues are identifiable but entries are not" presented as our discovery | Already reported for spliceJAC/scMomentum/dynamo | Balubaid 2025 |
| "Snapshots cannot identify rotation/flux; −log p is not the potential" as a new finding | Known | Weinreb 2018; Coomer 2022; Lavenant 2024 |
| "Operator archetypes" as new in v57 | Introduced in the authors' own v1 | Redd et al. 2026 (v1) |
| "First to embed cells by their local operators" | Cell-MNN shows operator UMAPs | Cell-MNN 2025 |

### Defensible claims (with wording)
| Claim | Bounded by |
|---|---|
| "A **score operator**: the per-cell Hessian of a score-matched log-density in the embedding, interpreted as stiff/soft directions and reported only through coordinate-invariant inertia (Sylvester), which needs no velocity. We found no single-cell method that does this." | Differs from ddHodge (velocity-potential Hessian), DiffusionOT (score used for landscape height, not Hessian), Cislo (Hessian at fixed points only) |
| "**Per-cell similarity-invariant eigen-descriptors** (λ₁, rotational index) of a drift fitted to an **entropic-OT barycentric velocity**, evaluated against **lineage barcodes** (LARRY) and held-out days. Prior OT-flow methods differentiate the learned field only for GRN edges." | TIGON; DiffusionOT; OTVelo; scDiffEq (LARRY, but no Jacobian read-out) |
| "**Two operators in one basis-invariant operator space with archetypes**, together with their per-cell disagreement as a read-out of departure from the gradient (equilibrium-like) case. This applies the landscape–flux decomposition at the operator level, per cell." | Wang/Zhu (global), PFI/Boffi (global), Cell-MNN (operator embedding, one operator type), scJDO v1 (archetypes, one operator) |
| "A **pre-registered identifiability analysis** across representation, clock, rotation, stationarity, diffusion and composition, with reported negative results (e.g., flow-J inversion on Factor Analysis; diffusion not identifiable at typical sampling intervals)." It is novel as a systematic, pre-registered evaluation. Each individual limit has precedent and must be cited. | Weinreb 2018; Coomer 2022; Lavenant 2024; Balubaid 2025; PFI (noise vs force) |
| "The method works on datasets without splicing kinetics" | Shared with OT/time-course methods; true relative to dynamo/spliceJAC |

---

## 5. Methods to cite or compare against (prioritised)

R = runnable baseline with public code (repository seen during this search, or a widely used package; versions not checked here). C = cite only.

1. **dynamo** (Qiu 2022) — R. Per-cell Jacobian, divergence, curl; the natural flow-operator comparator when velocity/labelling is available. *Already planned (S16).*
2. **spliceJAC** (Bocci 2022) — R. State-level eigen-spectra / instability. *Already planned (S16).*
3. **ddHodge** (Maehara & Ohkawa 2025) — R (Julia; github.com/kazumits/ddHodge.jl). Closest comparator for the score operator (per-cell Hessian) and for the gradient/curl split. **High priority, not yet in plan.**
4. **DiffusionOT / TIGON / DeepRUOT** — R for TIGON (code reported in the paper; not checked here). DiffusionOT/DeepRUOT code availability not checked. Closest comparators for the OT-derived flow Jacobian. At minimum cite all three. Ideally compare λ₁ from TIGON's velocity Jacobian with scJDO2's flow J on the same time course.
5. **Zhu et al. 2024 landscape–flux** (via dynamo vector field) — C (cite for the pairing / non-equilibrium framing; possibly reproduce EPR on one dataset).
6. **PFI / SF2M (score + flow)** — C (SF2M code exists in TorchCFM; not checked). Cite for f ≈ D∇log p + flux and for force–noise non-identifiability.
7. **OTVelo** — R (cite for label-free OT velocity; compare the velocity itself with barycentric velocity).
8. **PRESCIENT, PI-SDE, scDiffEq** — R (PI-SDE on GitHub per paper; PRESCIENT and scDiffEq are packages). Potential/SDE baselines on time courses; scDiffEq already uses LARRY.
9. **Waddington-OT, moscot, CellRank 2 (RealTimeKernel)** — R. Cite as the OT infrastructure; CellRank macrostates as a non-operator baseline for "decision cells".
10. **CellTran, BioTIP, Barcenas 2024 (spliceJAC-based)** — R for BioTIP (Bioconductor); CellTran and Barcenas code not checked. Per-cell / along-lineage transition-cell baselines against λ₁.
11. **Cell-MNN** — R (github.com/czi-ai/cell-mnn). Per-cell operator embedding; cite in D.
12. **Balubaid 2025; Weinreb 2018; Coomer 2022; Lavenant 2024** — C (identifiability section).
13. **scJDO v1** (Redd et al. 2026) — R (own; comparator already in S16).
14. Method citations: Hyvärinen 2005 (JMLR 6); Vincent 2011; Ding, Li & Jordan 2010 (semi-NMF); Cutler & Breiman 1994; Schiebinger 2019; Weinreb 2020 LARRY (*Science*, doi:10.1126/science.aaw3381); Godrèche & Luck 2018 (linear non-equilibrium analogue).

---

## 6. Unverified leads (do not cite until verified)

- **"Niemiec et al. 2025 bioRxiv", DOI 10.1101/2025.02.05.636568.** It is cited in the v56 notebook `concordance_04_compare.ipynb` as showing that splicing-derived Jacobians are noisy or inconsistent. The DOI returns 404 at doi.org and Crossref, the bioRxiv API has no record, and two web searches found no such paper. Ask the authors for the actual title; Balubaid et al. 2025 may be the intended paper.
- **scHopfield** (schopfield.readthedocs.io; KAUST PhD defence abstract 2025-11-26). Its documentation describes per-cell Jacobian eigenvalues, rotational parts and Hopfield energy fitted to RNA velocity. No peer-reviewed paper or preprint DOI was found. It is highly relevant to B and D if published; recheck before submission.
- **Publication venues stated only by search snippets**, so cite the arXiv IDs:
  - DeepRUOT (reportedly ICLR 2025)
  - Cell-MNN (ICLR 2026 poster page seen in results)
  - UPFI (NeurIPS 2025 poster page seen in results)
  - TrajectoryNet (ICML 2020)
  - MIOFlow (NeurIPS 2022)
  - JKOnet (ICML 2022)
  - SF2M (AISTATS 2024)
- **QBIO 2025 conference abstract** (Peking University) proposing entropy production / curl flux / time irreversibility as early-warning signals for cancer. Abstract only; methods not checked.
- **Kinger S et al. Inferring active neural circuits using diffusion scores.** arXiv:2605.02852 (2026). Verified on arXiv, but it is neuroscience: products of score blocks recover the Jacobian of a transition map. It is a conceptual neighbour of "operators from scores" and is optional to cite.
- **Yakovlev, Markovich & Puchkin**, arXiv:2512.24378 (2025). Verified; theory of log-density Hessian estimation by score matching. Optional method citation.
- Content details of **RegVelo** (*Cell* 2026), **GraphVelo** (*Nat Commun* 2025) and **DeepRUOT**: the records are verified, but whether they compute per-cell eigen-descriptors was not checked.

---

## 7. Search log (2026-10-26)

Verification tools: Crossref REST (`/works/{DOI}`, `query.bibliographic`), Europe PMC REST (abstracts; full-text XML keyword scans for DiffusionOT, Zhu 2024, ddHodge, spliceJAC, dynamo), bioRxiv API (`/details/biorxiv/{DOI}`), arXiv API (`id_list`, `ti:` search), JMLR landing page (Hyvärinen 2005), arXiv HTML (Cell-MNN). The local author files were grepped for the two DOIs.

Web searches (WebSearch):
1. Niemiec 2025 bioRxiv RNA velocity Jacobian splicing gene regulatory inconsistent
2. scJDO single-cell Jacobian dynamical operator Terooatea bioRxiv
3. score matching single-cell energy landscape potential Waddington diffusion model 2024 2025 (extended)
4. Hessian of potential single-cell landscape per-cell stability eigenvalues cell state curvature (extended)
5. optimal transport velocity field single-cell time course Jacobian gene regulatory TIGON DeepRUOT (extended)
6. entropy production detailed balance violation single-cell RNA-seq irreversibility non-equilibrium flux (extended)
7. "score matching" single-cell RNA-seq potential landscape learned energy cell fate (extended)
8. physics-informed neural SDE single-cell potential energy landscape PI-SDE
9. scHopfield single-cell Hopfield energy landscape Jacobian RNA velocity
10. clustering cells by local Jacobian dynamical regime single-cell vector field classification (extended)
11. denoising score matching cell state landscape bioRxiv 2025 "score" "Waddington" Hessian stability (extended)
12. "operator archetypes" OR "Jacobian archetypes" single-cell dynamics
13. Jacobian eigenvalues identify transition cells instability scRNA-seq vector field bifurcation per cell (extended)
14. Hodge decomposition RNA velocity potential curl gradient flux single-cell ddhodge 2025 Nature Communications
15. Maddu Chardès Shelley probability flow inference PFI single-cell published 2025
16. statistical approach systematic identification of transition cells scRNA-seq Cell Reports Methods 2024
17. BioTIP R package biological tipping point Nucleic Acids Research single-cell
18. Topographer stochastic dynamics cell fate decisions single-cell RNA-seq landscape
19. BioTIP "Detecting critical transition signals from single-cell transcriptomes" Nucleic Acids Research 2022
20. neural ODE learned drift Jacobian eigenvalues bifurcation cells time-course single-cell optimal transport stability analysis (extended)
21. diffusion model score function single-cell transcriptomics cell state stability landscape Hessian "log density" 2025 2026 bioRxiv (extended)
22. scDiffEq neural SDE single-cell lineage tracing LARRY drift diffusion Vinyard
23. Koopman operator single-cell RNA-seq time series dynamic distribution decomposition reprogramming
24. time irreversibility entropy production estimate from single-cell transcriptomics time course lineage tracing cell fate 2025 (extended)
25. Cell-MNN locally linearized ODE single-cell gene interactions state-dependent operator
26. "Jacobian of the score" OR "Hessian of the log-density" single-cell cell states stiffness plasticity (extended)
27. landscape curvature cell plasticity single-cell energy landscape local curvature per cell soft modes 2024 2025 (extended)
28. bioRxiv 2025 2026 "score-based" generative model Waddington landscape single-cell attractors energy function learned neural network (extended)
29. Niemiec bioRxiv February 2025 RNA velocity benchmark gene regulatory network Jacobian (extended)

Limitations:
- Searches were in English, through a US web index plus bibliographic APIs.
- Paywalled full texts (most *PNAS*/*Science* pages returned 403) were assessed from abstracts and, where available, Europe PMC open-access full text.
- A paper posted in the weeks before 2026-10-26 may not yet be indexed. Rerun searches 3, 4, 5, 13, 20 and 26 just before posting.
