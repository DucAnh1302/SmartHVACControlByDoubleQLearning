# Implementation Changelog & Architectural Ledger

## 1. Summary of Changes
This changelog details all modifications, newly introduced modules, scientific refactorings, test suites, and documentation files implemented in **SmartHVACControlByDoubleQLearning**.

The original Phase 1–6 implementation was strictly preserved under Git version control (`feature/generalization-and-physical-validity` branch), maintaining legacy models as historical baselines while adding an approved, physically valid target architecture.

---

## 2. Detailed File-by-File Changes

### 2.1 Multi-Domain Data & Provenance Layer
- **`src/data/provenance.py`** [NEW]
  - Implemented `ProvenanceType` enum (`OBSERVED`, `DERIVED`, `EXTERNAL`, `SIMULATED_ASSUMED`, `FITTED_WITH_CAVEAT`).
  - Implemented `SYSTEM_PROVENANCE_REGISTRY` mapping every variable, parameter, and assumption to source, confidence, and engineering caveats.
  - Added `export_provenance_table_md()` for automated Markdown documentation generation.
- **`src/data/domain_config.py`** [NEW]
  - Implemented `DomainSpec` dataclass for formal domain definitions.
  - Configured 4 verified domains: `2018_F2_Z1` (Primary), `2019_F2_Z1` (Cross-Year), `2018_F2_Z2` (Cross-Zone), `2018_F3_Z1` (Cross-Floor).
- **`src/data/domain_loader.py`** [NEW]
  - Implemented `load_domain_raw_dataframe()` and `preprocess_domain_timeseries()` with dynamic sensor outage window detection.
  - Implemented `split_temporal_dataset()` guaranteeing strictly sequential chronological cuts ($70\%$ train, $30\%$ test) with zero shuffling and zero future leakage.
  - Implemented caching mechanism in `data/processed/`.
- **`src/data/calibration.py`** [NEW]
  - Implemented `EquipmentCalibrator` performing K-Means clustering on active compressor power to extract operational stages (OFF, LOW, MED, HIGH).
  - Implemented `EquipmentStageCalibration` persisting calibration metadata to JSON.
  - Implemented `EquipmentDriftDetector` detecting equipment alterations across operational years.

### 2.2 Transition Models & Physical Thermodynamic Environment
- **`src/environment/transition_base.py`** [NEW]
  - Defined abstract base class `BaseTransitionModel` with single-step `predict_step()` and vectorized `predict_batch()` methods.
- **`src/environment/persistence_model.py`** [NEW]
  - Implemented `PersistenceTransitionModel` ($T_{t+1}=T_t$, $RH_{t+1}=RH_t$) as physical baseline.
- **`src/environment/rf_transition_model.py`** [NEW]
  - Encapsulated legacy Random Forest transition logic with vectorized batch inference and feature importance tracking.
- **`src/environment/greybox_model.py`** [NEW]
  - Implemented `GreyBoxThermalModel` combining 1st-order RC sensible heat transfer and latent moisture mass balance.
  - Implemented identifiable parameter fitting with Ridge regularization and standard error estimation ($k_{env}, k_{int}, k_{ac}, h_{env}, h_T, h_{ac}$).
  - Enforced thermodynamic consistency: cooling monotonicity, internal heat gain, ambient convergence, and psychrometric RH coupling.
- **`src/environment/model_validator.py` & `src/environment/run_model_validation.py`** [NEW]
  - Implemented comparative transition benchmark comparing Persistence vs Random Forest vs Grey-Box RC.
  - Evaluated single-step, 6h/24h multi-step rollouts, physical consistency tests, and cross-domain generalization.
- **`src/environment/target_env.py`** [NEW]
  - Implemented `TargetHVACEnv` featuring 600 discrete states ($10 \times 4 \times 3 \times 5$), 5 discrete actions, setpoint-to-stage dynamics, and closed-window initial invariant.
- **`src/reward/advanced_reward.py`** [NEW]
  - Implemented `AdvancedHVACRewardFunction` featuring ASHRAE 55 preferred zone ($[24^\circ\text{C}, 26^\circ\text{C}]$), quadratic severe violation penalties ($<21^\circ\text{C}, >29^\circ\text{C}$), humidity penalty, TOU peak tariff costs, compressor hunting penalties, and adverse window penalties.

### 2.3 Agent Architecture & Baseline Policies
- **`src/agent/target_double_q_agent.py`** [NEW]
  - Implemented `TargetDoubleQLearningAgent` with two independent tables ($Q_A, Q_B$) of shape $(600, 5)$, full visit count matrix $N(s, a)$, exploration decay, and deterministic tie-breaking.
- **`src/agent/baseline_policies.py`** [NEW]
  - Implemented 5 comparative control policies: `RandomPolicy`, `FixedSetpointPolicy` (25°C), `ComfortFirstPolicy`, `EnergySavingPolicy`, and `LegacyDQLAdapter` (mapping 45 legacy states into 600 target states).

### 2.4 Safety Gateway & BMS Hardware Interfaces
- **`src/safety/safety_gateway.py`** [NEW]
  - Implemented `SafetyGateway` with hard deterministic interlocks: Overheat emergency override ($29.5^\circ\text{C}$), Freeze emergency override ($18.0^\circ\text{C}$), Compressor short-cycle protection ($\ge 2$ hours), and Window night/weather safety vetos.
  - Implemented `ShadowModeRunner` for passive operational shadowing.
- **`src/safety/hil_interface.py`** [NEW]
  - Implemented `MockBMSControllerHIL` modeling BACnet MS/TP and Modbus TCP telemetry and actuator command packet dispatch with mandatory safety interlock verification.

### 2.5 Evaluation Benchmark, Robustness & Visualization
- **`src/evaluation/benchmark_suite.py`** [NEW]
  - Implemented `DomainShiftBenchmark` evaluating 4 domain shift axes across 3 transfer modes (`ZERO-SHOT`, `CALIBRATION-ONLY`, `FINE-TUNING`).
  - Implemented `evaluate_all_baselines()` for the 6-policy benchmark.
- **`src/evaluation/robustness_suite.py`** [NEW]
  - Implemented `RobustnessExperimentSuite` evaluating the policy across 7 plant perturbation scenarios and 3 random seeds.
- **`src/util/plot_benchmarks.py`** [NEW]
  - Generates high-resolution visualization figures: `baselines_comparison.png`, `domain_shift_comparison.png`, `robustness_comparison.png`, and `target_training_convergence.png`.
- **`train_target_dql.py`** [NEW]
  - Master pipeline script executing end-to-end data preparation, agent training (800 episodes), baseline evaluation, domain transfer benchmarking, and robustness testing.
- **`app/app.py`** [REFACTORED / UPGRADED]
  - Upgraded Streamlit web application into a 5-page interactive research Scenario Lab.

### 2.6 Automated Unit Tests (`tests/`)
- **`tests/test_provenance.py`** [NEW] — Tests data provenance types, mandatory variable registration, confidence levels, and caveats.
- **`tests/test_data_leakage.py`** [NEW] — Tests strict chronological dataset splitting and verifies zero future information leakage.
- **`tests/test_physical_consistency.py`** [NEW] — Tests cooling monotonicity, internal heat gain, envelope heat transfer, psychrometric coupling, and safe bounding.
- **`tests/test_target_dql.py`** [NEW] — Tests 600-state space, dual Q-table shapes, Double Q update decoupling, deterministic tie-breaking, and visit tracking.
- **`tests/test_safety_gateway.py`** [NEW] — Tests overheat/freeze overrides, short-cycle compressor locking, window weather/night vetos, and Mock BMS HIL controller.
- **`tests/test_domain_transfer.py`** [NEW] — Tests domain configurations, calibration persistence, and evaluation pipelines.

### 2.7 Documentation Deliverables
- **`PROJECT_ARCHITECTURE.md`** [NEW] — 9-layer system design with Mermaid diagrams.
- **`DATASET_AND_PROVENANCE.md`** [NEW] — CU-BEMS dataset documentation, outage analysis, and complete provenance registry.
- **`PHYSICAL_MODEL_VALIDATION.md`** [NEW] — Transition model benchmark (Persistence vs RF vs Grey-Box) and identifiability report.
- **`GENERALIZATION_BENCHMARK.md`** [NEW] — Cross-domain benchmark results (4 axes, 3 transfer modes) and in-depth analysis.
- **`TRAINING_AND_EVALUATION.md`** [NEW] — Training protocol, 600-state/5-action design, reward formulation, and 6-baseline comparison.
- **`REAL_WORLD_READINESS.md`** [NEW] — Tiered deployment framework, Safety Gateway rules, and BMS HIL/Shadow Mode specifications.
- **`LIMITATIONS_AND_FUTURE_WORK.md`** [NEW] — Dataset limitations, robustness results, and research roadmap.
- **`TECHNICAL_LIMITATIONS.md`** [NEW] — Exhaustive scientific disclosure of system and model limitations.
- **`BAO_CAO_TONG_KET_DO_AN.md`** [NEW] — Comprehensive Vietnamese academic final report.
- **`README.md`** [UPDATED] — Updated system architecture, reproducibility guide, benchmark highlights, and directory layout.
- **`IMPLEMENTATION_STATUS.md`** [UPDATED] — 100% completion recorded across all implementation tasks.

---

## Scientific Validation and Rigor Audit Phase (2026-11-09)

### 1. Data Leakage Elimination & 3-Way Target Domain Partition
- **`src/data/domain_loader.py`**:
  - Replaced unconstrained `bfill().ffill()` with bounded interpolation (`max_interpolate_hours=3`).
  - Added binary quality tracking masks: `temp_is_interpolated` and `rh_is_interpolated`.
  - Implemented `split_target_domain_dataset(df, calib_ratio=0.20, adapt_ratio=0.40, test_ratio=0.40)` with zero temporal overlap.
  - Implemented dynamic archive path resolution (no hardcoded machine usernames).
- **`src/data/loader.py`**:
  - Removed hardcoded machine paths; added dynamic environment variable and relative fallback resolution.

### 2. Physical Model Integrity & Fitting Pipeline
- **`src/environment/greybox_model.py`**:
  - Enforced explicit fitting requirements: raises `ValueError` if valid observations $N < 50$.
  - Added `save()` and `load()` JSON parameter serialization.
  - Added `perform_sensitivity_analysis(...)` calculating local parameter sensitivity ($\pm 25\%$).
  - Updated default placeholder parameters to `SIMULATED_ASSUMED` (preventing silent masquerading as fitted).
- **`src/environment/target_env.py`**:
  - Added `get_model_config()` method exposing active physical model parameters, calibration stages, and provenance.

### 3. Scientific Provenance Corrections
- **`src/data/provenance.py`**:
  - Added `FITTED` and `SIMULATED` to `ProvenanceType`.
  - Corrected `ambient_temp` and `ambient_rh` from `EXTERNAL` to `SIMULATED_ASSUMED` reflecting synthetic diurnal equation.

### 4. Multi-Seed Baseline Benchmark & Single Q-Learning Direct Comparison
- **`src/agent/baseline_policies.py`**:
  - Added `RuleBasedThermostatPolicy` (commercial deadband thermostat) and `ACOffPolicy` (lower-bound energy reference).
- **`src/evaluation/benchmark_suite.py`**:
  - Implemented `evaluate_all_baselines_multiseed` across 5 random seeds (`42, 101, 202, 303, 404`).
  - Computed 95% Confidence Intervals, Degree-Hours, Switching events, and Energy savings vs Fixed-25°C.
  - Rewrote `DomainShiftBenchmark` to strictly use the 3-way non-leaking partition with multi-budget fine-tuning (10 and 25 episodes).
- **`train_target_dql.py`**:
  - Trained `QLearningAgent` (600 states) on identical environment to evaluate overestimation bias against Double Q-Learning.
  - Saved `dql_vs_ql_overestimation.png` and `baselines_multiseed_raw.csv`.

### 5. Automated Testing Expansion (32/32 Tests Passing)
- **`tests/test_data_leakage.py`**: Added 3 tests (3-way partition integrity, deliberate leakage detection, bounded interpolation with quality mask).
- **`tests/test_physical_consistency.py`**: Added test for un-fitted model flag and sample size rejection ($N < 50$).
- **`tests/test_target_dql.py`**: Added test for environment physical configuration reflection via `get_model_config()`.

### 6. User Interface Contrast & Full-Width Compliance
- **`.streamlit/config.toml`**: Configured dark theme (`base="dark"`, `backgroundColor="#0b1120"`, `textColor="#f1f5f9"`).
- **`app/app.py`**:
  - Applied fluid full-width layout (`[DIR-202611-002]`).
  - Enforced high-contrast typography (`#38bdf8` electric cyan headings, `#1e293b` cards with `#f1f5f9` text).
  - Integrated all 9 selectable controllers, sensitivity table, and multi-seed benchmark tabs.

### 7. Documentation & Final Academic Report
- **`BAO_CAO_TONG_KET_DO_AN.md`**: Expanded into 20 comprehensive Vietnamese chapters with exact executed numbers.
- **`GENERALIZATION_BENCHMARK.md`**, **`TRAINING_AND_EVALUATION.md`**, **`PHYSICAL_MODEL_VALIDATION.md`**, **`DATASET_AND_PROVENANCE.md`**, **`IMPLEMENTATION_STATUS.md`**, **`README.md`**: Fully updated.
