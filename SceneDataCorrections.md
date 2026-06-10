> **MERGED:** All items in this addendum were merged into SceneDB2.0.md Rev 2.2
> (2026-06). The stride limit stated below as 256 bytes was superseded by the
> 128-byte limit in Rev 2.1 §7.1. This file is retained for history only.

# Architectural Correction Notes: SceneDB 2.0 & Helio
**Document Classification:** Internal Engineering Addendum / Implementation Guardrails
**Target Systems:** Layer 1 (Storage), Layer 2 (Orchestration), Layer 3 (Execution)

---

## 1. LAYER BOUNDARY & LEASE DEADLOCK RESOLUTIONS (Layer 1 $\leftrightarrow$ Layer 2)

### 1.1 The Synchronous Lease Deadlock
* **Vulnerability:** The structural separation of `SceneDBCell` compaction paths (deferred to frame-boundary isolation blocks) creates a hard stall vulnerability if an asynchronous engine client—specifically the **Editor Subsystem** or background scripts—holds a persistent selection or read-lease open across frame boundaries.
* **Enforced Corrective Action:** Implement an atomic **Double-Buffered State Mask** for the `Liveness Bitmask` and index column registries. 
    * When a client requests a read-lease, it locks a snapshot identifier for the current frame topology.
    * If a lease boundary exceeds a maximum timeout window (hard-capped at $2.0	ext{ms}$ into the frame isolation phase), Layer 2 forces an automatic lease revocation. The long-running client handle is pushed to a secondary stale validation lane, allowing the primary memory layout to execute its vectorized swap-and-pop compaction routines immediately.

### 1.2 Macro-Driven Stride Masking
* **Vulnerability:** The $256	ext{-byte}$ per-element compile-time stride guardrail introduced in Section 7.1 checks individual type registrations in isolation. Developers can bypass this by declaring separate, highly atomic component types for the same structural entity IDs, leading to scattered runtime cache line allocation offsets and severe L1/L2 cache thrashing during composite queries.
* **Enforced Corrective Action:** Enhance the compile-time procedural macro ingestion tool. The macro system must process type configurations holistically across each **Cell Composition Domain**. It must dynamically aggregate the cumulative byte size of *all columns* registered against an shared index mapping within a specific cell type allocation. If the combined cross-component allocation exceeds $256	ext{ bytes}$ per unique element slot, the compiler must emit a hard allocation error.

---

## 2. SPATIAL GRID & CONCENTRIC DOMAIN MITIGATIONS (Layer 1 $\leftrightarrow$ Layer 3)

### 2.1 Green-to-Blue Perimeter Thrashing (Boundary Oscillations)
* **Vulnerability:** When a camera view matrix or high-frequency game thread observer hovers precisely along the mathematical boundary line dividing adjacent spatial grid cells, sub-pixel camera jitter or floating-point micro-movements trigger rapid, non-stop domain updates (alternating between the Green Simulation Margin and Blue Execution Core). This results in severe execution pipeline stuttering due to continuous host-to-device synchronization allocations.
* **Enforced Corrective Action:** Integrate a coordinate-based **Spatial Hysteresis Buffer** into the Layer 2 tracking matrix.
    $$\text{PromotionBoundary} = \text{CellBounds} + \Delta_{\text{pad}}$$
    $$\text{DemotionBoundary} = \text{CellBounds} + \Delta_{\text{pad}} + \delta_{\text{hysteresis}}$$
    A cell promoted to the Inner Execution Core [Blue Domain] must be locked into that operational state until the tracking coordinate center moves entirely out of its bounding framework plus an explicit physical padding offset, which is calculated as a fixed $10\%$ margin of the total cell spatial volume width.

### 2.2 Lockstep Array Underflows (The Null-Slot Cascade)
* **Vulnerability:** In highly sparse scene blocks (e.g., cell domains populated with thousands of non-visual script logic hooks or collision fields but containing only a handful of actual meshes), the multi-type compaction requirement populates the unified index stream with thousands of empty `0xFFFFFFFF` tokens. Layer 3 (Helio) compute culling shaders waste massive memory bandwidth processing and discarding empty descriptors.
* **Enforced Corrective Action:** Layer 2 must calculate a runtime **Density Efficiency Index (DEI)** before initiating an upload to VRAM.
    $$\text{DEI} = \frac{\text{Count}(\text{ValidComponentOffsets})}{\text{TotalAllocatedQueryBlockSize}}$$
    * If the DEI falls below a minimum hardware threshold of $25\%$, Layer 2 bypasses raw lockstep streaming.
    * The host CPU executes an immediate, vectorized data reduction pass via SIMD bitmanipulation hardware instructions (`_mm256_mask_compressstore_epi32` or ARM Neon equivalents), stripping out null tokens and generating a dense, packed index payload for the GPU execution layer.

---

## 3. MATHEMATICAL & CULLING CORRECTIONS (Layer 3)

### 3.1 Ceiled-Floor Hi-Z Sampling Gaps
* **Vulnerability:** Changing the continuous mipmap selection math to utilize the floor operator $\lfloor \log_2(\text{MaxDim}) \rfloor$ pulls from a higher-resolution mip layer. However, dropping down a mip level reduces the total real screen area covered by a fixed conservative $2 \times 2$ texel footprint. For highly elongated, non-symmetrical, or diagonally rotated bounding box geometries, this creates localized sub-pixel sampling leaks, underestimating depth coverage and triggering false occlusion culling.
* **Enforced Corrective Action:** The compute culling shader must dynamically evaluate the screen-space bounding projections against the UV dimensions of the selected floor mipmap level.
    $$\text{TexelSpan}_x = \text{UV}_{\text{max}.x} - \text{UV}_{\text{min}.x}$$
    $$\text{TexelSpan}_y = \text{UV}_{\text{max}.y} - \text{UV}_{\text{min}.y}$$
    If the projected extent spans across more than two discrete texels along either screen coordinate axis, the shader must dynamically expand its sampling kernel from a rigid $2 \times 2$ texture lookup to a wide $3 \times 3$ or $4 \times 4$ conservative texel gather loop.

---

## 4. ADVERSARIAL INTEGRATION TESTS (Verification Matrix)

To ensure these specific correction rules remain active and unbroken across all subsequent engine compilation targets, the following three compliance protocols are appended to the main verification test library:

```
===================== ADVERSARIAL COMPLIANCE MATRIX =====================

10. EDITOR_LEASE_STALL_COMPLIANCE
    - Action: Open an explicit, persistent entity selection handle in Layer 2.
    - Test: Force immediate frame isolation compaction in Layer 1.
    - Pass Criteria: Thread execution must continue cleanly with zero engine lockups.

11. GRID_BOUNDARY_OSCILLATION_COMPLIANCE
    - Action: Jitter camera parameters directly along cell grid coordinates at 60Hz.
    - Test: Monitor Host-to-Device buffer allocation triggers.
    - Pass Criteria: Zero redundant state allocation or buffer recreation requests.

12. SPARSE_CELL_COMPACTION_COMPLIANCE
    - Action: Populate test cell with 10,000 logic triggers and 5 meshes.
    - Test: Measure VRAM index payload layout density.
    - Pass Criteria: DEI reduction forces dense SIMD compression; no raw null-token cascades.
```