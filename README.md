# Vision Transformer Token Reduction — Research Summary (2025)

---

## 1. 현재 진행 중인 연구

### 1.1 목표

* **문제의식:** ViT는 토큰 수가 많아 FLOPs/지연시간 부담이 큼. 특히 **초기 레이어(L0–L2)**에서 무분별한 감축은 표현 붕괴로 정확도 하락을 유발.
* **최종 목표:** Training-free 조건에서 **Top-1 정확도를 유지하면서 FLOPs 약 40% 절감 (DeiT-Small 기준 2.7G 근처)**.

  1. 초기 레이어 민감성 완화
  2. 전층 균등/비균등 스케줄에서 안정적 감축 구현

---

### 1.2 제안 방법론 (요약)

#### (A) Head-Diversity 기반 Early 감축

* **선택자 (φ):**  ( \phi_j = [in_j^{(1)}, \dots, in_j^{(H)}] ), head-wise inflow로 head-signature 구성.
* **정책:** 허브-쿼터(inflow×size 상위 q%) + farthest-first (FF)로 K개 유지.
* **병합:** KV 공간 정합 (merge = kv), push-lite(α, β-cap, sparse top-r).
* **스케줄:** Early 과감 (keep 0.66~0.68), Mid 보수, Late no-merge.

#### (B) Size-Aware + Normalization (ours_size_norm)

* **size 추적:** 각 토큰의 L2-norm 기반 평균화.
* **정규화:** re-center + L2-clip(τ) + temperature scaling ( scale = (s/mean(s))^n )
* **보정:** push-lite(α≤0.15), β-cap을 size에 맞춰 적용 ( β_k = β_0 (s_k/mean_s)^δ )

#### (C) ToMe Reproduce (baseline)

* 원본 ToMe (Sinkhorn, CLS 보호, size-weighted merge) 정학히 재현.
* 동일 GFLOPs (≈2.7G 기준) 조건에서 ours vs ToMe 비교.

---

### 1.3 이론적 근거 (스케치)

* **선택 정당성:** inflow-TopK → 중복 선택 억제, FF/submod → 표현폭 보장 → 손실 상계로 설명 가능.
* **안정성:** convex combination으로 각 스텝 Lipschitz 상수 ≈ 1 유지 → 누적 drift 감소.
* **정규화 효과:** L2-clip/temperature scaling으로 분산 평창 억제, CKA/각도 보존, KL 증가 억제.

---

### 1.4 실험 설계

* **데이터/모델/환경:** ImageNet-1k/val, timm pretrained(ckpt 미사용), DeiT-Small/ViT-Small, 224×224, AMP off.
* **비교축:**

  1. Early 전용(L0/L1/L2/0-2): ours_size_norm vs ToMe
  2. 전층 r=13 (2.7G) ours 균등 keep≈0.60
  3. Ablation: size(on/off), norm(on/off), push-lite(α, β, top-r), merge(KV vs V)
* **지표:** Top-1/drop(pp), KL(B||M), RelCost/FLOPs, Early layer별 Δdrop, CKA/각도, tail Δ, latency/throughput.

---

## 2. 최근 구현/검증 결과 (2025-11 업데이트)

| 모델                      | 범위        | Top-1 (%)    | KL       | Δmargin(mean) |
| ----------------------- | --------- | ------------ | -------- | ------------- |
| ToMe                    | L0–L2     | 78.75        | ≈ 0.0254 | −0.1255       |
| Ours (hquota_ff, L0–L2) | **79.20** | **≈ 0.0137** | −0.0424  |               |

➡️ **ToMe 대비 Top-1 +0.45%p, KL/Δmargin 모두 안정화.**
➡️ **핵심 요인:** CLS 보호 + head-diversity 선택 + size-weighted V-merge.
→ Early 표현 붕괴 억제, KV-merge로 선택–병합 일관성 향상.

---

## 3. 향후 실험 제안 (업데이트)

### 3.1 Reproduce 우선 마일스톤

* ToMe r=13 재현 유지 (DeiT-S 기준 2.7G, drop ≈ −0.8pp)
* 동일 GFLOPs에서 ours vs ToMe **패러티 비교 (Top-1 vs FLOPs)**

### 3.2 Early 집중 최적화 (확장)

* **A. r-sweep:** r ∈ {4, 9, 13, 20, 28} @ L0–L2
* **B. match-feature:** k vs xnorm
* **C. merge 공간:** kv vs v
* **D. head-quota q:** {0.20, 0.30, 0.40}
* **E. 안정화 계수 η:** {0.00, 0.05, 0.10, 0.20}

### 3.3 전층 40% 절감 타깃 (keep≈0.60)

* ours 전층 keep≈0.60 vs ToMe r=13(2.7G) 동일 GFLOPs 비교.
* Late 보수화(last-K no-merge/profile) + Early 보강으로 정확도/안정성 균형.

### 3.4 강건성/OOD

* Blur/Noise/JPEG 변형 실험에서 size/norm 유무, KV vs V/q 영향 비교.
* ImageNet-V2/R/C 소규모 검증 진행 예정.

### 3.5 시스템 측정

* **GFLOPs ↔ latency/throughput 상관 (AMP on/off)**.
* **토큰 수 → HW 타일(64/128)** 정렬 시 실제 지연 패턴 측정.

---

## 부록. 현재 구현체 목록 (업데이트)

| 파일 경로                      | 주요 기능                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `tome_adapter/plugin.py`   | 원본 ToMe 패치 호출 (`tome.patch.timm(model)`), r/layer 주입                                 |
| `ours/attn.py`             | merge→attend→unmerge 파이프라인 (φ 생성, head-diversity 선택자, size-weighted V-merge, CLS 보호) |
| `ours/selectors/hquota.py` | 허브-쿼터 + farthest-first (φ 기반 선택자), CLS 보호 및 OOB fix 반영                               |
| `ours/merges/kv_merge.py`  | size-weighted V-merge, KV-merge 옵션                                                   |
| `core/sched/early_bias.py` | keep→r 변환 (50% cap, CLS 보호) 스케줄                                                      |
| `core/metrics.py`          | Top-1/KL/Δmargin/diversity 프로브                                                       |

---

**정리:**

* 현재 모델은 training-free 조건에서 약 40% FLOPs 절감 목표를 향해 수렴 중이며,
* 초기층(L0–L2) 감축 안정화 + size-normalization의 조합이 기존 ToMe 대비 일관된 개선을 보임.
* 다음 단계는 full-layer 0.60× 스케줄에서 정확도–FLOPs trade-off를 정량적으로 검증하는 것.
