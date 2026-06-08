# Ecological model notes

This document records the real-ecosystem principles the simulator targets and
how the headless probe (`bench/probe.mjs`) measures them. The goal is for the
sandbox to reproduce *qualitative* properties of real ecosystems, not to be a
calibrated model of any specific biome.

## Principles targeted

| Principle | Real ecology | Where in `sim.js` |
|-----------|--------------|-------------------|
| Logistic primary production | Producers grow toward a resource-set carrying capacity | `updateEnvironment` growth term with `competition = 1 - veg/capacity` |
| Emergent carrying capacity | Consumer numbers limited by food / predation / disease, not hard ceilings | per-type `maxCount` raised to non-binding safety rails; every guild verified to settle *below* its cap (cap-utilization metric) |
| Lindeman trophic efficiency (~10%) | Each trophic level holds ~10% of the energy below it; losses are respiration | `PRED_ASSIMILATION` assimilation + metabolic respiration; no free predation bonus. Measured predator/herbivore standing energy lands ~10-20% |
| Depletable resources | A consumer's food must be a finite stock it can draw down, or it cannot be food-limited | nectar (`c.flower`) is a standing stock that regenerates toward a vegetation-set target (not recomputed each tick), so pollinators are nectar-limited, not cap-limited |
| Holling type II functional response | Predator intake saturates with prey density (handling time) | `hunt`: `1 - exp(-attack * preyCount)` + `huntLock` handling time |
| Kleiber's law | Basal metabolic rate ∝ mass^0.75 | `massMetabolic = body^0.75` in `updateAgent` |
| Q10 temperature dependence | Ectotherm metabolic rate ~2.3x per +10 °C | `q10Factor` in `updateAgent` |
| Seasonal recruitment | Temperate animals breed in a spring/summer pulse | seasonal multiplier in `maybeReproduce` |
| Predator–prey coupling | Predator peaks lag prey peaks (Lotka–Volterra) | emerges from Holling predation + energy flow |
| Nutrient cycling / mass balance | N cycles plant→detritus→soil→plant with small open fluxes | stoichiometric `NUTRIENT_PER_BIOMASS` transfers |
| SIR(S) disease | Susceptible→Infected→Recovered(immune)→Susceptible; density-dependent β | `updateDisease` with immunity timer + disease mortality |
| Rescue effect | Connected landscapes avoid permanent local extinction | `supportMigration` (rescue-only, not pegging) |

## Probe metrics

- **Extinction rate** – fraction of seeds where a guild hits 0 post burn-in.
  Real guilds persist; routine whole-guild extinction signals instability.
- **Cap utilization** – mean population / its `maxCount`. A guild well below its
  cap is emergently regulated by resources/predation/disease; one pinned near
  the cap is still ceiling-limited (the report flags these). Every guild should
  be comfortably below its cap.
- **Energy pyramid** – predator standing somatic energy as a fraction of
  herbivore energy. Lindeman implies ~10-20%; values near 100% mean predators
  are over-fed (an inverted pyramid).
- **Predator–prey lag** – cross-correlation lag of predators vs herbivores.
  Should be positive (predators follow prey).
- **CV (coefficient of variation)** – population variability. ~0 means an
  artificial frozen equilibrium; >>1 means boom–bust toward extinction. Real
  populations sit in between.
- **Trophic pyramid** – producer ≫ herbivore > predator standing stock.
- **Matter pools** – soil-nutrient / detritus / plant drift over the tail.
  Near-zero drift means the cycle has reached steady state and conserves matter
  (vs. an unbounded source or sink).

Run: `node bench/probe.mjs [ticks] [seeds]` (parallel, one worker per seed).
Target a frozen build with `SIM_FILE=bench/baseline.sim.js`.
