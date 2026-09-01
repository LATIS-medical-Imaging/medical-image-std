# Research Directions for Top-Tier Venue (CVPR/MICCAI/ICCV)

## Current State

- Best Dice: ~0.38, Lesion Recall: ~0.85 on INbreast with standard architectures (UNet, UNet++, Attention UNet, DeepLabV3+)
- 152 baseline experiments completed on INbreast, 29 on CBIS-DDSM
- Synthetic benchmark with 32 cluster-level + 33 object-level failure cases exposing Dice/IoU limitations
- Particle Topology Score (PTS) metric in development (v1/v2/v3), with known theoretical issues

---

## Direction 1: Topology-Aware Loss for Tiny Object Segmentation (Strongest Angle)

### Motivation

The synthetic benchmark exposes that Dice/IoU are blind to topological failures: object merging, splitting, displacement, and count mismatch. No existing differentiable loss addresses this for particle-like tiny objects.

### How It Differs from Existing Work

- **clDice** (2021, MICCAI): focuses on tubular structures (vessels), not particle-like objects
- **Persistent homology losses** (Hu et al. 2019, TopSeg): expensive, focus on H0/H1 Betti numbers — not well-suited to tiny scattered objects
- **Panoptic Quality**: not differentiable

### Concrete Approach

1. Use the synthetic benchmark to formally characterize where Dice fails (32+ cases already available)
2. Design a loss with two components:
   - **Soft object counting** via differentiable connected components (approximate with Gaussian blob detection on sigmoid output)
   - **Spatial assignment** via differentiable optimal transport between predicted and GT centroids
3. Combined loss: `L = L_dice + λ₁ · L_count + λ₂ · L_transport`
4. Show on INbreast + CBIS-DDSM that this improves lesion recall and FROC without hurting Dice

### Related Work to Position Against

clDice, TopoLoss, Betti matching loss, Boundary loss, HD loss

---

## Direction 2: Scale-Aware Architecture for Sub-5-Pixel Objects

### Motivation

Microcalcifications are 2–5 pixels. Standard encoders (ResNet34) downsample 32x, making these objects disappear in deep features.

### Option A: Multi-Scale Feature Preservation Network

- Keep a high-resolution branch that never downsamples below 2x (HRNet-style but for tiny objects)
- Cross-attention between high-res and low-res branches
- Key insight: for objects smaller than the receptive field, high-res features must survive

### Option B: Point-based Detection + Segmentation

- Stage 1: Detect MC candidates as points (heatmap regression, CenterNet-style)
- Stage 2: For each detected point, crop a local patch and segment at full resolution
- Decouples "where" from "what shape" — matches clinical workflow

### Option C: Frequency-Domain Augmented Features

- Inject frequency-domain features (from existing FEBDS pipeline) as additional encoder channels
- High-frequency components preserve tiny calcification edges that spatial convolutions smooth out
- Novel element: learnable frequency band selection per encoder stage

---

## Direction 3: Topology-Aware Metric + Benchmark (Metrics Paper)

### Motivation

The synthetic benchmark (32 cluster + 33 object failure cases) and 3 PTS versions provide a foundation. A well-designed metric paper with a public benchmark is highly citable.

### What Is Needed for Publication

1. Fix PTS theoretical issues (zero-propagation, triangle inequality violation)
2. Proposed fix — **Wasserstein-based panoptic matching**:
   - Match GT ↔ Pred objects via Hungarian algorithm on centroid distance
   - Per-matched-pair: compute overlap quality (IoU) + shape similarity
   - Unmatched GT = false negatives, unmatched Pred = false positives
   - Final score = mean matched quality × F1(matching)
3. **Public benchmark release** with the synthetic cases — this is the differentiator
4. Validate: show that the new metric correlates with radiologist rankings better than Dice/IoU/HD

### Venue Fit

MICCAI (metrics track), Medical Image Analysis journal, or CVPR if framed as "evaluation of tiny object segmentation"

---

## Direction 4: Foundation Model Adaptation for Microcalcification

### Motivation

SAM2, MedSAM, BiomedParse are prominent but struggle with tiny objects.

### Approach

- Fine-tune a medical foundation model with:
  - **Point prompts** at MC locations (natural fit — radiologists click on MCs)
  - **Custom decoder head** operating at full resolution (SAM's decoder loses tiny objects)
  - **Topology-aware fine-tuning loss** (from Direction 1)
- Novelty: demonstrate that foundation models fail catastrophically on sub-5px objects, then fix it with minimal architectural change

---

## Direction 5: Multi-Task Learning — Detection + Segmentation + Counting

### Approach

Joint model that simultaneously:
1. Segments MC masks (pixel-level)
2. Detects individual MC centers (point-level)
3. Predicts cluster-level MC count (image-level)

### Why It Works

- The counting head provides a global constraint that prevents merging (if count=15, the segmentation can't produce 3 blobs)
- The detection head provides localization that prevents displacement
- Architecture: shared encoder → 3 decoder heads with mutual consistency loss

---

## Recommended Strategy: Combine Directions 1 + 3

### Proposed Paper

> **"Beyond Dice: Topology-Preserving Loss and Evaluation for Tiny Object Segmentation"**

### Four Contributions

1. **Benchmark**: Release synthetic MC benchmark showing Dice/IoU failures — 65 controlled cases, public code
2. **Metric**: Propose a topology-aware metric based on optimal-transport matching
3. **Loss**: Differentiable version of the metric as a training loss
4. **Experiments**: Show on INbreast + CBIS-DDSM that topology-aware training improves lesion recall and FROC while maintaining Dice, using existing 152-config baseline as comparison

### CVPR Checklist

- **Novel** — no existing topology loss for particle-like tiny objects
- **Well-motivated** — synthetic benchmark proves the problem exists
- **Practical** — plug-in loss, no architecture change needed
- **Strong baselines** — 152 experiments already completed
- **Reproducible** — public benchmark + code

---

## Direction 6: Hybrid CNN-ViT Modeling (Deep Dive)

### Key Empirical Insight

Patch size 32×32 is the sweet spot from 152 baseline experiments (best Dice + best FROC). This maps directly to a natural ViT patch embedding size. MCs (2–5 pixels) live *within* a single 32×32 token — so the ViT's job is inter-patch reasoning (cluster context, spatial layout), while the CNN's job is intra-patch feature extraction (detecting the MC within the patch).

### Why ViT Patch = 32×32 (Not Smaller)

Empirical finding: training a ViT with small patches (4×4, 8×8, 16×16) on MC segmentation causes the model to lose the ability to recognize microcalcifications. The reason: a small ViT patch fragments the MC and its local context across multiple tokens. Each token sees only a few bright pixels with no surrounding tissue context — indistinguishable from noise. At 32×32, the entire MC (2–5px) plus its local tissue context lives within one token. The token has enough information to decide "this patch contains an MC" and the transformer's job becomes reasoning about *relationships between MC-containing patches* — which is what defines a cluster.

This means: **the CNN must handle intra-patch feature extraction, and the ViT must handle inter-patch reasoning.** Any architecture that asks the ViT to reason at pixel-level will fail.

### The Core Problem with Pure ViT for Tiny Objects

Standard ViT patch embedding (linear projection of flattened patch) destroys sub-patch spatial structure. A 3-pixel MC in a 32×32 patch becomes a single token — the ViT knows "this patch has an MC" but not *where within the patch*. This is fine for classification but fatal for segmentation — the decoder cannot recover the MC position.

Standard CNN encoders (ResNet34 with stride 32) have the opposite problem: they preserve local features through skip connections but lack global context to reason about cluster-level patterns.

### Approach A: CNN Local Encoder + ViT Global Reasoning (Two-Stream Fusion)

**Architecture:**

```
Input patch (32×32 or tiled to 512×512)
    │
    ├──► CNN Branch (ResNet18, stride=4)
    │      → Feature map: H/4 × W/4 × C_cnn
    │      → Preserves sub-pixel MC detail
    │      → Skip connections at 1/1, 1/2, 1/4
    │
    ├──► ViT Branch (patch_embed=32×32)
    │      → Sequence of N tokens (each = one 32×32 region)
    │      → Self-attention across patches → cluster-level context
    │      → Each token "knows" about all other MC locations
    │
    └──► Fusion Module
           → ViT tokens reshaped to spatial grid
           → Cross-attention: CNN features query ViT tokens
           → CNN provides "where exactly" within the patch
           → ViT provides "what's around" across patches
           → Fused features → lightweight decoder → mask
```

**Why this is novel for MC segmentation:**
- CNN branch operates at stride 4 (not 32), so a 3px MC still occupies 1-2 feature pixels — it survives
- ViT branch at patch=32 sees each MC-containing region as one token — efficient global reasoning
- Cross-attention fusion lets the CNN ask "is this bright spot an MC or noise?" by consulting the ViT's cluster-level context
- Existing hybrid approaches (TransUNet, HiFormer, MaxViT) don't explicitly design around preserving sub-patch structure

**Key design choices:**
- CNN branch: shallow (ResNet18 or first 2 blocks of ResNet34), stride ≤ 4, to keep MCs alive
- ViT branch: small (6-8 layers, dim=256, 4 heads), patch=32, learnable position embedding
- Fusion: bidirectional cross-attention (CNN→ViT for detail, ViT→CNN for context), not just concatenation
- Decoder: simple bilinear upsampling + 1×1 conv (novelty is in the encoder, not decoder)

### Approach B: ViT with CNN Patch Embedding (Drop the Linear Projection)

**Key modification:** Replace ViT's standard linear patch embedding with a small CNN that preserves spatial structure within each patch.

```
Standard ViT:   32×32 patch → flatten → linear → 1 token (768-d)
                 ❌ Spatial info within patch is destroyed

Proposed:        32×32 patch → 3-layer CNN (stride 2,2,2) → 4×4×C feature map
                 → flatten → linear → 1 token (768-d)
                 ✅ CNN learns to encode MC position within patch

                 PLUS: keep the 4×4 intermediate features for the decoder
                 → "sub-token" skip connections
```

**Architecture:**

```
Input (512×512, grayscale)
    │
    ├──► CNN Patch Embedder (per 32×32 patch)
    │      Conv 3×3 stride 2 → 16×16×32
    │      Conv 3×3 stride 2 → 8×8×64
    │      Conv 3×3 stride 2 → 4×4×128
    │      Flatten + Linear → token (256-d)
    │      Side output: save 4×4×128 features (sub-token map)
    │
    ├──► Transformer Encoder (6 layers, 256-d, 8 heads)
    │      → Global self-attention over 16×16 = 256 tokens
    │      → Each token is contextually enriched
    │
    └──► Sub-Token Decoder
           → Reshape tokens to 16×16 grid
           → Upsample 2× → 32×32, concatenate sub-token features (4×4 per token → 64×64)
           → Upsample 2× → 128×128
           → Upsample 2× → 256×256
           → Upsample 2× → 512×512
           → 1×1 conv → binary mask
```

**Why this matters:**
- Standard ViTs (even with overlapping patches) lose intra-patch spatial info
- The CNN embedder learns a spatial encoding of where the MC sits *within* the 32×32 patch
- "Sub-token skip connections" are analogous to UNet skips but at sub-patch resolution
- This is architecturally simple — a clean contribution

**Positioning vs. existing work:**
- **TransUNet** (2021): Uses CNN features + ViT but the CNN is the full ResNet (stride 32) — MCs are already gone before the ViT sees them
- **CoTr** (2021): Deformable attention on CNN features — still starts from CNN stride-32 features
- **HiFormer** (2023): Double-branch Swin + CNN — Swin window size is fixed, not designed for sub-5px objects
- **MaxViT** (2022): Multi-axis attention, but the patch embedding is still a linear projection
- **Yours**: The CNN patch embedder is specifically designed to preserve MC-scale features before tokenization

### Approach C: Hierarchical ViT with Particle-Aware Tokens

**Idea:** Instead of fixed-grid patches, generate tokens centered on detected candidate locations.

```
Stage 1: Lightweight CNN → MC candidate heatmap (like CenterNet)
          → Top-K candidate locations (e.g., K=64)

Stage 2: For each candidate, extract a 32×32 patch → CNN embed → token
          PLUS: add background tokens from a regular grid (e.g., 8×8 = 64 tokens)
          Total: 128 tokens (64 candidate + 64 background)

Stage 3: Transformer Encoder
          → Self-attention between all 128 tokens
          → Candidate tokens attend to each other → cluster reasoning
          → Candidate tokens attend to background → context

Stage 4: Decode each candidate token → per-candidate 32×32 segmentation mask
          → Stitch masks back into full image
```

**Why this is novel:**
- Token generation is content-adaptive, not grid-based
- The transformer explicitly reasons about MC-to-MC relationships (which is what defines a cluster)
- Computational cost scales with number of candidates, not image size
- Naturally handles variable MC counts

**Trade-off:** Requires a good Stage 1 detector. Could use your existing FEBDS or TopHat as a classical pre-filter, or train a simple heatmap CNN.

### Approach D: Dual-Resolution ViT (Full-Res + Context)

**Idea:** Two ViT branches at different resolutions processing the same image.

```
Branch 1 (Detail): patch_size=8, processes a 256×256 crop around the ROI
    → 32×32 = 1024 tokens
    → Each token covers 8×8 pixels → a 3px MC spans ~1 token
    → Fine-grained spatial reasoning
    → Use windowed attention (Swin-style) to handle 1024 tokens efficiently

Branch 2 (Context): patch_size=32, processes the full 512×512 image
    → 16×16 = 256 tokens
    → Each token covers one 32×32 region → cluster-level view
    → Full self-attention (only 256 tokens)

Cross-attention fusion:
    → Detail tokens query Context tokens for cluster awareness
    → Context tokens query Detail tokens for precise localization
    → Fused Detail tokens → upsample → mask
```

**Why it works:**
- Branch 1 preserves MC detail at near-pixel resolution
- Branch 2 provides the "is this part of a cluster?" context
- Cross-resolution attention is the key novelty — existing multi-scale ViTs (PVT, Twins) downsample progressively, losing detail
- Matches your finding: patch 32 for context, but finer resolution needed for segmentation quality

### Approach E: UNet Encoder with Bottleneck Transformer (Simplest, Most Practical)

**Idea:** Keep the UNet++ encoder (your best model) but replace the bottleneck with transformer layers.

```
Encoder (ResNet34, stride 16 instead of 32):
    Stage 1: 256×256 × 64   (skip → decoder)
    Stage 2: 128×128 × 128  (skip → decoder)
    Stage 3: 64×64 × 256    (skip → decoder)
    Stage 4: 32×32 × 512    (skip → decoder)

Bottleneck Transformer (replaces Stage 5):
    → Reshape 32×32×512 → 1024 tokens of dim 512
    → 4 transformer layers with self-attention
    → Each token = one 16×16 region of original image
    → Global reasoning about MC cluster layout
    → Reshape back to 32×32×512

Decoder (UNet++ style):
    → Progressive upsampling with dense skip connections
    → Final: 512×512 × 1 (binary mask)
```

**Key modification:** Stop the CNN at stride 16 instead of stride 32. At stride 16, a 3px MC still occupies a fraction of a feature pixel — marginal but present. At stride 32, it's completely gone.

**Why this is practical:**
- Minimal change from your existing best model (UNet++)
- `timm` and `smp` already support this via encoder truncation
- The transformer bottleneck adds cluster-level context that pure CNN lacks
- Easy ablation: UNet++ vs UNet++ w/ transformer bottleneck → clean comparison
- Similar spirit to TransUNet but with the critical stride-16 modification

### Implementation Notes

**Available in your environment:**
- `timm==1.0.27`: ViT, DeiT, Swin, MaxViT, EfficientViT backbones
- `segmentation_models_pytorch==0.5.0`: UNet/UNet++ decoder infrastructure
- `transformers==5.3.0`: HuggingFace ViT, BEiT, SegFormer
- `torch==2.10.0`: FlashAttention, nested tensors, compile support

**Recommended starting point:**
1. Start with **Approach E** (bottleneck transformer) — lowest risk, fastest to implement, cleanest ablation against your 152 baselines
2. If results are promising, upgrade to **Approach B** (CNN patch embedding) — stronger novelty, clean story
3. **Approach A** (two-stream) is the full paper architecture — implement after validating the core hypothesis

**Encoder choices from `timm`:**
- `maxvit_tiny_tf_512` — built-in multi-axis attention, handles 512×512 natively
- `efficientvit_b1` — fast, designed for dense prediction
- `swin_v2_tiny_window8_256` — strong baseline, window attention
- `convnext_tiny` — pure CNN baseline to compare against hybrid

---

## Direction 7: Collaborative, Aggregation, Contrastive & Local Feature Approaches (ViT patch=32)

All approaches below share the same premise: ViT with patch=32×32, where each token encapsulates one MC + its local context. The question is how tokens interact and how we recover pixel-level segmentation.

### 7.1 Collaborative Token Refinement

**Core idea:** Tokens don't just attend to each other — they *collaboratively refine* each other's local feature maps through iterative message passing.

```
Step 1: CNN Patch Embedder
    Each 32×32 patch → CNN → local feature map F_i (8×8×C) + token t_i (d-dim)

Step 2: Collaborative Refinement (N rounds)
    For each round:
        a) Global attention: tokens attend to all other tokens
           → t_i' = MultiHeadAttn(t_i, {t_j}_{j≠i})
           → Each token absorbs cluster-level context

        b) Token-to-local feedback: enriched token refines its own local feature map
           → F_i' = F_i + reshape(MLP(t_i'), 8×8×C)
           → The global context tells the local map "yes, this bright spot
             is part of a cluster, enhance it" or "no, this is noise, suppress it"

        c) Local-to-token update: refined local features update the token
           → t_i'' = t_i' + pool(F_i')
           → If the local map changed (e.g., suppressed a false positive),
             the token reflects this for the next round

Step 3: Decode each refined F_i' → per-patch 32×32 mask → stitch
```

**Why this is novel:**
- Standard ViT: tokens interact but never feed back into local feature maps
- Standard CNN+ViT hybrids: one-shot fusion (concatenate or cross-attend once)
- This creates an iterative loop: global context refines local detail, refined local detail updates global context
- Analogous to Expectation-Maximization: local features are the E-step, global tokens are the M-step

**Number of rounds:** 2–3 is likely sufficient. Each round is lightweight (one attention layer + one MLP).

**Positioning:** Related to message-passing neural networks (MPNNs) and graph attention networks, but applied to ViT tokens with the novel token↔local-feature feedback loop.

### 7.2 Multi-Scale Feature Aggregation with Gated Fusion

**Core idea:** Extract features at multiple CNN depths *within* each 32×32 patch, aggregate them with learned gates, then let the ViT reason over aggregated tokens.

```
Per-patch CNN Feature Pyramid:
    32×32 patch → Conv block 1 → F1 (32×32×16)   [texture, edges]
                → Conv block 2 → F2 (16×16×32)   [local structure]
                → Conv block 3 → F3 (8×8×64)     [MC-level features]
                → Conv block 4 → F4 (4×4×128)    [patch-level summary]

Gated Aggregation:
    → Gate_k = sigmoid(W_k · [GAP(F_k); global_context])
    → token_i = Σ_k Gate_k · flatten(AdaptivePool(F_k, 1×1))

    The gates learn WHICH scale matters for each patch:
    - MC-containing patch: Gate_1 (texture) high, Gate_3 (MC features) high
    - Background patch: Gate_4 (summary) high, others suppressed
    - Tissue-boundary patch: Gate_2 (structure) high

ViT Transformer:
    → Self-attention over gated tokens
    → Each token carries scale-adaptive information

Decoder:
    → Reshape tokens to grid
    → For each patch position, use ALL skip features (F1-F4) gated by
      the refined token → progressive upsampling → mask
```

**Why this matters for MC segmentation:**
- MCs have discriminative features at very specific scales (2–5px = one specific CNN layer)
- Standard feature pyramids (FPN) aggregate with fixed weights — wrong for heterogeneous patches
- The learned gates adapt per-patch: a patch with a 3px MC upweights fine-scale features; a patch with dense tissue upweights coarse features
- The ViT's attention can leverage scale information: "this token is upweighting fine features → likely contains an MC → attend to it more from neighboring tokens"

### 7.3 Contrastive Token Learning

**Core idea:** Use contrastive learning to shape the ViT's token embedding space so that MC-containing tokens are close together and far from background tokens. This directly encourages the ViT to learn cluster structure.

```
Architecture (training time):
    CNN Patch Embedder → tokens {t_1, ..., t_N}
    ViT Encoder → enriched tokens {t_1', ..., t_N'}

    Segmentation head: tokens → decoder → mask → L_seg (Dice + BCE)

    Contrastive head (auxiliary):
        Classify each token: MC-positive or MC-negative
        (based on whether its 32×32 patch contains any GT MC pixels)

        L_contrast = SupCon({t_i'}, {y_i})
            where y_i ∈ {positive, negative}

        Pull positive tokens together → ViT learns "MC tokens form clusters"
        Push negatives away → ViT learns to distinguish MC from noise

    Total loss: L = L_seg + λ · L_contrast

Architecture (inference time):
    Drop the contrastive head. The ViT has already learned
    a token space where MC tokens cluster together.
```

**Why contrastive learning helps here specifically:**
- MCs are rare (< 0.1% of pixels). Without contrastive guidance, the ViT's attention may ignore MC tokens entirely since background tokens dominate
- Contrastive loss forces the ViT to create a representation where MC tokens are *distinguishable* — this is a prerequisite for good segmentation
- At inference, the learned attention patterns naturally focus on MC-containing patches

**Extensions:**
- **Hard negative mining:** tissue calcifications, dense breast regions, imaging artifacts — things that look like MCs but aren't
- **Pixel-level contrastive loss:** within each 32×32 patch, pull MC pixels together and push background pixels away in the CNN feature space
- **Cross-image contrastive:** MCs from different patients should still be close in embedding space → improves generalization

**Positioning vs. existing contrastive segmentation:**
- **ContrastiveSeg** (2021): pixel-level contrastive for semantic seg — not designed for tiny objects, no token-level component
- **Dense contrastive learning** (2021): pretraining method — doesn't address the "MC tokens are overwhelmed by background tokens" problem
- **Yours**: token-level contrastive specifically targeting the tiny-object-in-large-image problem

### 7.4 Local Feature Preservation with Spatial Attention Tokens

**Core idea:** Each ViT token carries not just a feature vector but a *spatial attention map* indicating where within its 32×32 patch the important content is. The ViT learns to refine these attention maps through inter-token interaction.

```
CNN Patch Embedder:
    32×32 patch → CNN → feature map F_i (8×8×C)
    → Global: token t_i = GAP(F_i)           [what is in this patch]
    → Local:  attn_i = Conv1x1(F_i) → 8×8    [where in this patch]
    → Token becomes: (t_i, attn_i) pair

ViT with Spatial Attention Tokens:
    Standard self-attention on t_i (content tokens)
    PLUS: attention-weighted pooling uses attn_i

    Key insight: when token_j attends to token_i, it should attend
    to the *important region* of token_i's local features, not the
    average. So the attention value is:

        V_i = Σ_{(x,y)} attn_i(x,y) · F_i(x,y,:)

    instead of the standard:

        V_i = GAP(F_i)

    This means the ViT's self-attention is guided by local spatial attention.

Spatial Attention Refinement:
    After each transformer layer, update attn_i using the attention
    weights from neighboring tokens:

        attn_i' = attn_i + Σ_j α_{ij} · transfer(attn_j)

    "If my neighbor's MC is at the top-right of its patch, and I'm the
     patch below it, my MC is likely at the top of my patch" → spatial
     coherence across patch boundaries.

Decoder:
    For each patch, the refined attn_i' tells the decoder exactly
    where the MC is within the 32×32 region:
    → F_i ⊙ attn_i' → weighted features → upsample → mask
```

**Why this is the strongest local feature approach:**
- Solves the fundamental problem: ViT patch=32 token knows *that* an MC exists but not *where*
- The spatial attention map preserves "where" through the entire transformer
- Attention refinement propagates spatial coherence across patches (MCs near patch boundaries)
- At decode time, the attention map directly guides upsampling — no guessing

### 7.5 Graph-Based Token Collaboration with Adaptive Topology

**Core idea:** Instead of full self-attention (every token attends to every other), build a dynamic graph where edges connect tokens that should collaborate — primarily MC-containing neighboring tokens.

```
Step 1: CNN Patch Embedder → tokens {t_1, ..., t_N} on a 16×16 grid

Step 2: Dynamic Graph Construction
    → Initial edges: 8-connected spatial neighbors (like a grid graph)
    → Adaptive edges: add long-range edges between tokens with high
      cosine similarity (top-K per token, K=4)
    → This creates a graph where distant MC clusters can communicate
      without routing through background tokens

Step 3: Graph Attention Network (4 layers)
    → GAT-style attention: each token attends to its graph neighbors
    → Edge features encode spatial relationship (relative position)
    → Much cheaper than full self-attention: O(N·K) vs O(N²)

Step 4: Token Aggregation for Segmentation
    → Each token's final representation includes:
      - Its own features (local MC info)
      - Neighbor-aggregated features (cluster context)
      - Graph-level readout (global image context via a [CLS]-like token)
    → Decode per-token → per-patch mask → stitch
```

**Why graph-based collaboration:**
- Full self-attention wastes compute: a background token in the corner doesn't need to attend to an MC token on the opposite side
- The adaptive graph naturally discovers cluster structure: MC tokens become densely connected, background tokens stay locally connected
- Scales to larger images (1024×1024, 2048×2048) where full attention is prohibitive
- The graph topology itself is interpretable: visualize which tokens are connected → shows learned cluster boundaries

**Positioning:**
- **ViG** (Vision GNN, 2022): graph-based vision backbone, but fixed K-NN graph — not adaptive to content
- **DynamicViT** (2021): token pruning, not graph construction
- **Yours**: content-adaptive graph specifically for the MC clustering problem, where graph topology = cluster topology

### 7.6 Contrastive Cluster-Aware Aggregation (Combined Approach)

**This is the full proposed architecture combining the best of 7.1–7.5:**

```
┌─────────────────────────────────────────────────────────┐
│                  Input: 512×512 grayscale                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  CNN Patch Embedder (shared, per 32×32 patch)     │    │
│  │    → local feature map F_i (8×8×C)                │    │
│  │    → spatial attention map attn_i (8×8)            │    │
│  │    → token embedding t_i (d-dim)                   │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Collaborative ViT Encoder (6 layers)             │    │
│  │                                                    │    │
│  │  Each layer:                                       │    │
│  │    1. Self-attention with spatial-attention-        │    │
│  │       weighted values (from 7.4)                   │    │
│  │    2. Token-to-local feedback: refine F_i          │    │
│  │       using enriched token (from 7.1)              │    │
│  │    3. Local-to-token update: refined F_i           │    │
│  │       updates token (from 7.1)                     │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Training Losses                                   │    │
│  │    L_seg:      Dice + TopK BCE on decoded masks    │    │
│  │    L_contrast: SupCon on token embeddings (7.3)    │    │
│  │    L_spatial:  MSE on attn_i vs GT MC locations     │    │
│  │                                                    │    │
│  │    L = L_seg + λ₁·L_contrast + λ₂·L_spatial       │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Sub-Token Decoder                                 │    │
│  │    → Refined F_i ⊙ refined attn_i                 │    │
│  │    → Per-patch 32×32 mask prediction               │    │
│  │    → Stitch to full 512×512 output                 │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Three key novelties in one architecture:**
1. **Collaborative refinement loop** (token ↔ local feature feedback) — Direction 7.1
2. **Spatial attention tokens** (ViT preserves intra-patch localization) — Direction 7.4
3. **Contrastive token shaping** (MC tokens cluster in embedding space) — Direction 7.3

**Ablation study design:**
- Baseline: CNN-only (UNet++ with ResNet34) — your best at Dice 0.38
- +ViT bottleneck only (Approach E from Direction 6) — validates transformer helps
- +Collaborative refinement — validates iterative feedback
- +Spatial attention tokens — validates local feature preservation
- +Contrastive loss — validates token space shaping
- Full model — all combined

---

## Comparison: All Approaches

| Approach | Novelty | Effort | Risk | Best For |
|---|---|---|---|---|
| **Direction 6** | | | | |
| A: Two-Stream CNN+ViT | High | High | Medium | Full paper contribution |
| B: CNN Patch Embedding | High | Medium | Low | Clean architectural contribution |
| C: Particle-Aware Tokens | Very High | High | High | If detection is reliable |
| D: Dual-Resolution ViT | Medium | High | Medium | If compute is not a concern |
| E: Bottleneck Transformer | Low-Medium | Low | Very Low | Quick validation, ablation |
| **Direction 7** | | | | |
| 7.1: Collaborative Refinement | High | Medium | Medium | Novel token↔feature loop |
| 7.2: Gated Multi-Scale Aggregation | Medium | Medium | Low | If scale variance is key |
| 7.3: Contrastive Token Learning | Medium-High | Low | Low | Easy add-on to any ViT |
| 7.4: Spatial Attention Tokens | High | Medium | Medium | Preserves intra-patch location |
| 7.5: Graph-Based Collaboration | High | High | Medium | Scalability to large images |
| 7.6: Combined Architecture | Very High | High | Medium | Full paper contribution |

### Recommended Paper Structure (If Going Hybrid CNN-ViT)

> **"Particle-Preserving Vision Transformer for Microcalcification Segmentation"**

1. **Problem statement:** Tiny objects (2–5px) vanish in standard encoder downsampling; ViT patch embeddings destroy intra-patch spatial structure
2. **Contribution 1:** CNN patch embedding that preserves sub-patch MC features before tokenization
3. **Contribution 2:** Sub-token skip connections that carry intra-patch spatial detail to the decoder
4. **Contribution 3:** Ablation on INbreast + CBIS-DDSM against 152 CNN baselines and standard ViT architectures (TransUNet, Swin-UNet, SegFormer)
5. **Analysis:** Attention maps showing the ViT learns inter-MC relationships (cluster structure)

---

## Architecture Suggestions Summary

| Architecture | Why | Reference |
|---|---|---|
| HRNet + Object Query | Preserves high-res features for tiny objects | HRNet (CVPR'19) + DETR-style queries |
| UNet + Mamba | Long-range context with linear complexity | U-Mamba (2024), VM-UNet |
| Swin-UNet v2 | Shifted windows capture local detail well | Swin-UNet (2022) |
| EfficientSAM + custom decoder | Foundation model with efficient adaptation | EfficientSAM (2024) |
| nnU-Net | Self-configuring baseline, hard to beat | nnU-Net (2021) — export scripts already available |
| MaxViT | Multi-axis attention (block + grid) at all scales | MaxViT (ECCV'22) |
| EfficientViT | Hardware-efficient ViT for dense prediction | EfficientViT (CVPR'23) |
| ConvNeXt V2 + ViT bottleneck | Modern CNN + transformer hybrid | ConvNeXt V2 (2023) |

---

## Direction 8: PTS v3 Metric Paper (In Discussion with Supervisors)

### Current State of PTS

Three versions exist with known issues at each stage:
- **v1**: Harmonic mean zero-propagation (one bad component kills score), bottleneck distance too coarse (only 4 distinct S_T values across 32 cases)
- **v2**: Random baseline too high (mean=0.59), zero-overlap 100px-displaced prediction scores 0.93, effective score range compressed to [0.6, 1.0]
- **v3**: Three parameter-free fixes (object-scale normalization, per-pair overlap modulation φ, dominance ratio) — validation shows 100px displacement drops from 0.93→0.00, random baseline from 0.59→0.04

### What Makes PTS v3 Publishable

The core argument: **existing segmentation metrics fail for tiny scattered objects, and here's a principled fix with proof.**

You have unique assets no other group has:
1. **65+ controlled synthetic failure cases** (32 cluster + 33 object) that formally demonstrate where Dice/IoU/HD break
2. **Quantitative failure catalog**: 8 documented failure modes with root cause analysis
3. **Three-version evolution** showing iterative theoretical refinement (reviewers appreciate transparency)
4. **Ablation of each v3 change** independently on critical cases
5. **Real data** (INbreast + CBIS-DDSM) with 152 baseline experiments to validate on

### Paper Structure: PTS v3

> **"Particle Topology Score: A Structure-Aware Metric for Tiny Object Segmentation Evaluation"**

**Section 1 — Problem Statement:**
- Dice, IoU, Hausdorff are pixel-level metrics — they conflate localization, boundary quality, object count, and topology
- For microcalcifications (2–5px, 10–30 per image): a single merged blob can score Dice > 0.7 while destroying all clinical information (cluster morphology determines malignancy grading)
- Show 4–5 failure cases from your synthetic benchmark as motivation figures

**Section 2 — PTS Formulation:**
- Particle measure lifting: binary mask → weighted point process {(centroid_i, area_i)}
- Two-level evaluation:
  - Pixel-level: matched Dice with overlap modulation φ + Wasserstein transport with object-scale normalization
  - Object-level: persistence-based topology comparison + count accuracy
- Aggregation: harmonic mean (justified by: single catastrophic failure should dominate)
- Emphasize: **parameter-free** — all normalization constants derived from the GT geometry

**Section 3 — Synthetic Benchmark:**
- 65 controlled cases across 2 scales (cluster and single-object)
- Each case isolates one failure mode: missing, merging, splitting, displacement, hallucination, topology destruction, boundary error, count error
- Benchmark is public + reproducible (synthetic generation code released)
- Show the full matrix: Dice vs IoU vs HD vs PTS for all 65 cases

**Section 4 — Validation on Real Data:**
- Run PTS v3 on your 152 INbreast experiments — show that PTS ranking differs from Dice ranking and correlates better with lesion recall / FROC
- Key finding to highlight: configurations with high Dice but low PTS correspond to models that merge or miss individual MCs
- **Critical addition (see Direction 9):** validate on re-annotated CBIS-DDSM to show PTS is annotation-quality sensitive

**Section 5 — Comparison with Existing Metrics:**
- Dice, IoU, HD, HD95, ASSD, Boundary F1, clDice, Panoptic Quality
- Show that PTS is the only metric that simultaneously captures: displacement tolerance, topology preservation, count accuracy, and boundary quality
- Sensitivity analysis: PTS vs Dice vs PQ on graduated failure series (displacement sweep, fragmentation sweep, hallucination sweep)

**Section 6 — Limitations & Open Questions:**
- Harmonic mean still has cliff effects (discuss geometric mean alternative)
- Computational cost of Hungarian matching (O(N³) — acceptable for MC counts ~10-30, may not scale to hundreds)
- Not a true metric (no triangle inequality) — acknowledge this honestly, argue practical utility outweighs

### Remaining Theoretical Issues to Resolve (Before Submission)

1. **S_T still only produces few distinct values** — persistence diagram comparison via Wasserstein-1 is better than bottleneck but may still lack discrimination. Consider: is S_T even necessary, or does S_W + S_C already capture topology?
2. **Aggregation function** — harmonic mean is principled but harsh. Run the geometric mean experiment (you have scripts for this) and show both options to supervisors
3. **Radiologist correlation study** — ideally, get 3 radiologists to rank 20 prediction pairs by quality, then compute Kendall's τ between radiologist ranking and {Dice, PTS, PQ} ranking. This would be the strongest validation.

### Venue Fit

- **MICCAI 2027** (metrics/evaluation track) — natural fit, medical imaging audience
- **Medical Image Analysis** (journal) — allows longer format for theoretical depth
- **CVPR/ECCV** — if framed as "evaluation methodology for tiny object segmentation" (not just medical), broadens impact
- **IEEE TMI** — if combined with the re-annotated dataset (Direction 9)

---

## Direction 9: Re-Annotated CBIS-DDSM Dataset Paper

### Why This Is Valuable

CBIS-DDSM is the most widely used public mammography dataset, but its annotations have known problems:
- Coarse ROI-level annotations (bounding regions, not pixel-precise MC masks)
- Inconsistent annotation quality across cases
- Some ROI masks include surrounding tissue, not just the calcification
- No per-calcification instance segmentation (only cluster-level binary masks)

A **re-annotated version with pixel-precise, per-calcification instance masks** would be:
1. Immediately useful to every MC segmentation researcher
2. A prerequisite for properly evaluating topology-aware metrics like PTS
3. A contribution that strengthens every other paper you write (you benchmark on your own high-quality dataset)

### What to Annotate

**Annotation levels (from essential to ideal):**

1. **Per-calcification instance segmentation** (essential)
   - Each individual MC gets its own mask ID
   - Enables object-level evaluation (lesion recall, FROC, PTS)
   - This is what no public MC dataset currently provides at scale

2. **Calcification morphology labels** (high value)
   - Per-MC: amorphous, coarse heterogeneous, fine pleomorphic, fine linear/branching, round/punctate, vascular, eggshell/rim
   - Per-cluster: clustered, linear, segmental, regional, diffuse/scattered
   - These are the BI-RADS descriptors that determine clinical management
   - Enables morphology-aware evaluation: "did the model preserve the linear arrangement?"

3. **Annotation confidence** (nice to have)
   - Per-MC: confident / uncertain / very uncertain
   - Lets researchers filter by annotation quality for sensitivity analysis

### Paper Structure: Dataset Paper

> **"CBIS-DDSM-MC: Instance-Level Microcalcification Annotations for the CBIS-DDSM Dataset"**

**Section 1 — Motivation:**
- MC segmentation research bottlenecked by annotation quality
- Show examples of CBIS-DDSM annotation problems (coarse ROI, missed MCs, merged clusters)
- Quantify: in X% of cases, the original ROI mask includes >50% non-MC tissue

**Section 2 — Annotation Protocol:**
- Annotator qualifications (radiologists? radiology residents? trained annotators with radiologist review?)
- Annotation tool used
- Per-MC instance labeling protocol
- Inter-annotator agreement measurement (Cohen's kappa, Dice between annotators)
- Quality control: double-annotation of subset, adjudication process

**Section 3 — Dataset Statistics:**
- Total MCs annotated, MCs per image distribution, MC size distribution (pixels)
- Cluster statistics: MCs per cluster, cluster extent, inter-MC distances
- Comparison with original CBIS-DDSM annotations: how much does the mask differ?
- Morphology distribution (if annotated)

**Section 4 — Benchmarking:**
- Re-run your 152-config baseline experiments with the new annotations
- Key question: **do model rankings change with better annotations?**
  - If yes: the field has been benchmarking on noisy labels — your dataset corrects this
  - If no: the models are robust to annotation noise — also interesting
- Report Dice, IoU, lesion recall, FROC, and PTS with old vs new annotations

**Section 5 — PTS Validation on Real Data:**
- THIS IS WHERE PTS v3 AND THE DATASET PAPER REINFORCE EACH OTHER
- Show that PTS on the re-annotated dataset correlates better with clinical outcomes than Dice
- Instance-level annotations are *required* for PTS to work properly — your dataset enables proper PTS evaluation

**Section 6 — Public Release:**
- Release annotations as COCO-format JSON (compatible with your existing loader)
- Provide conversion scripts, visualization tools, evaluation code
- Hosted on a persistent repository (Zenodo, HuggingFace Datasets)

### Venue Fit

- **MICCAI 2027** (dataset track) — datasets are now first-class contributions
- **Scientific Data** (Nature) — dedicated dataset journal, high visibility
- **Medical Image Analysis** — if combined with comprehensive benchmarking
- **NeurIPS Datasets & Benchmarks** — broad ML audience, high impact for dataset papers

### How the Dataset Enables Other Papers

The re-annotated CBIS-DDSM is a **force multiplier** for all your other contributions:

| Paper | How Dataset Helps |
|---|---|
| PTS v3 metric | Instance annotations required for proper topology evaluation — validates PTS on real data |
| Hybrid CNN-ViT model | Train and evaluate on high-quality annotations — results are more trustworthy |
| Topology-aware loss | Instance labels enable counting loss and transport loss supervision |
| Contrastive learning | Per-MC labels provide positive/negative token classification ground truth |

---

## Direction 10: Multi-Paper Strategy (Recommended Roadmap)

### The Three-Paper Plan

Given your assets (PTS v3, re-annotated CBIS-DDSM, 152 baselines, synthetic benchmark, hybrid model ideas), the optimal publication strategy is three papers that build on each other:

### Paper 1: Dataset Paper (Submit First)

> **"CBIS-DDSM-MC: Instance-Level Microcalcification Annotations and Benchmarks"**

**Venue:** NeurIPS Datasets & Benchmarks 2027 or Scientific Data

**Timeline:** Can submit as soon as annotations are complete

**Contributions:**
1. Re-annotated CBIS-DDSM with per-MC instance masks
2. Baseline benchmarks (your 152 configs) with old vs new annotations
3. Public release with evaluation toolkit

**Why submit first:** Establishes the dataset. All subsequent papers cite it. Reviewers of Paper 2 and 3 can verify results on a public dataset.

### Paper 2: Metric Paper (Submit Second)

> **"PTS v3: Structure-Aware Evaluation for Tiny Object Segmentation"**

**Venue:** MICCAI 2027 or Medical Image Analysis

**Timeline:** After PTS v3 is finalized with supervisors + dataset paper is submitted

**Contributions:**
1. PTS v3 formulation (parameter-free, theoretically grounded)
2. Synthetic benchmark (65 controlled failure cases)
3. Validation on CBIS-DDSM-MC (Paper 1's dataset) — show PTS correlates with clinical quality better than Dice
4. Public benchmark + evaluation code

**Why submit second:** Uses the dataset from Paper 1 for real-data validation. The synthetic benchmark alone is strong but real-data validation makes it much more convincing.

### Paper 3: Architecture Paper (Submit Third)

> **"[Hybrid CNN-ViT Architecture] for Microcalcification Segmentation"**

**Venue:** CVPR 2028 or MICCAI 2028

**Timeline:** After model development and training

**Contributions:**
1. Novel hybrid CNN-ViT architecture (Direction 6 or 7)
2. Evaluated on CBIS-DDSM-MC (Paper 1) using PTS v3 (Paper 2) + standard metrics
3. Ablation study against 152 CNN baselines
4. Analysis: attention maps, topology preservation, cluster-level reasoning

**Why submit third:** Uses both the dataset and the metric from Papers 1 and 2. The paper can claim "we evaluate not just pixel-level (Dice) but structural fidelity (PTS)" — reviewers will appreciate the evaluation depth.

### Alternative: Two-Paper Strategy (Faster)

If time or supervisor preference dictates fewer papers:

**Paper A: Combined Metric + Dataset**

> **"PTS v3: A Topology-Aware Metric for Microcalcification Segmentation, Validated on Re-Annotated CBIS-DDSM"**

**Venue:** Medical Image Analysis (journal — allows length for both contributions)

**Contributions:**
1. PTS v3 metric
2. Re-annotated CBIS-DDSM-MC dataset
3. Synthetic benchmark
4. Baseline benchmarks showing PTS vs Dice on real data

This is a strong single paper because the metric and dataset *validate each other*: the dataset enables PTS evaluation, and PTS reveals annotation quality differences that Dice misses.

**Paper B: Architecture Paper**

Same as Paper 3 above, referencing Paper A for dataset and metric.

### Timeline Sketch

```
Now ─────────────────────────────────────────────────────► Future

│ Re-annotating CBIS-DDSM          │ PTS v3 finalization     │ Model development
│ (ongoing)                        │ (with supervisors)      │ (Direction 6/7)
│                                  │                         │
├── Dataset ready ─────────────────├── PTS v3 ready ─────────├── Model ready
│                                  │                         │
├── Submit Paper 1 (Dataset)       ├── Submit Paper 2 (PTS)  ├── Submit Paper 3 (Model)
│   NeurIPS D&B / Sci Data        │   MICCAI / MedIA        │   CVPR / MICCAI
│                                  │                         │
│   OR: Submit Paper A (combined metric + dataset) ──────────│
│        Medical Image Analysis                              │
```

### What Makes This Strategy Strong

1. **Each paper is self-contained** but references strengthen each other
2. **The dataset is the foundation** — it makes every other contribution more credible
3. **PTS v3 is already 80% done** — v3 implementation exists, validation scripts exist, presentation exists. The main gap is real-data validation (which the dataset fills)
4. **The model paper comes last** because it benefits most from having the dataset and metric established
5. **You avoid the "we evaluated on our own private dataset with our own metric" criticism** — by publishing dataset and metric first, Paper 3's evaluation is on public, peer-reviewed foundations