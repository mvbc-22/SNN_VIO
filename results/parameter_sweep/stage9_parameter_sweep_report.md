# Stage 9 — Systematic parameter sweep

- Controlled configurations: 17 (baseline plus one-factor-at-a-time variants).
- APS segment: intervals 678 through 682.
- Every variant changes one parameter from the same baseline; this identifies local sensitivity, not parameter interactions or a global optimum.
- The event-driven and hybrid trackers use the exact Stage 8 grid association. Frame-only is retained as a reference and does not use event/grid parameters.

## Baseline

- EPE: 4.208541624890199
- Mean track lifetime: 122590.0 μs
- Mean survival: 0.5564
- Failure rate: 0.3464
- Events / successful update: 45.28866554997208
- Candidate features / event: 499.986
- Processing time: 36106.11 ms

## Accuracy sensitivity

- **Feature count (accuracy):** measured EPE span 4.001 px across tested values; lowest EPE was 800.0 (3.686 px).
- **Grid-cell size (accuracy):** measured EPE span 0.000 px across tested values; lowest EPE was 8.0 (4.209 px).
- **Radius R (accuracy):** measured EPE span 0.037 px across tested values; lowest EPE was 16.0 (4.214 px).
- **Threshold T (accuracy):** measured EPE span 0.028 px across tested values; lowest EPE was 10000.0 (4.220 px).
- **Alpha (accuracy):** measured EPE span 0.006 px across tested values; lowest EPE was 0.75 (4.220 px).
- **Beta (accuracy):** measured EPE span 0.033 px across tested values; lowest EPE was 0.25 (4.209 px).

## Resource and implementation sensitivity

- **Latency-sensitive (measured):** feature-count variants changed event-driven processing time from 9516 to 46353 ms; T variants changed it from 30012 to 62204 ms. Changing cell size changed time from 38551 to 39983 ms, but did not materially prune candidate checks in this run.
- **Memory-sensitive:** feature count controls fixed FeatureState storage (16500 bytes at baseline using the current NumPy state dtype). Grid-cell size changes logical cell count and grid index bookkeeping.
- **Hardware-oriented:** feature count/slots, integer cell indices, fixed grid dimensions, bounded R/T, and alpha/beta represented as quantized coefficients are natural fixed-width parameters. The present Python implementation does not establish fixed-point precision or FPGA timing.
- **Grid selectivity limitation:** online velocity estimates and the conservative velocity-expanded grid neighborhood yielded approximately the full active feature set per event (about 499.99/500.0 at baseline). The grid was correct but not selective in this five-interval sweep. Cell-size variation changed logical cell count, not measured candidate pruning; improving a selective, exact motion-aware index is a follow-up before claiming grid latency benefits for online tracking.

## Multi-metric operating-point candidates

A configuration is listed below if no other tested configuration is simultaneously no worse in EPE, lifetime, survival, failure rate, events/update, candidate count, processing time, and compact memory proxies. These are Pareto candidates; the sweep does not force a single weighted-score winner.

| ID | Features | Cell px | R px | T μs | α | β | EPE px | Lifetime μs | Survival | Failure | ms | Candidates/event |
|----|----------|---------|------|------|---|---|--------|-------------|----------|---------|----|------------------|
| cfg_000 | 500 | 16 | 8 | 50000 | 0.5 | 0.1 | 4.209 | 122590 | 0.556 | 0.346 | 36106.1 | 499.99 |
| cfg_001 | 100 | 16 | 8 | 50000 | 0.5 | 0.1 | 7.687 | 68301 | 0.310 | 0.554 | 9515.8 | 100.00 |
| cfg_002 | 300 | 16 | 8 | 50000 | 0.5 | 0.1 | 5.567 | 110164 | 0.500 | 0.404 | 23669.8 | 299.99 |
| cfg_003 | 800 | 16 | 8 | 50000 | 0.5 | 0.1 | 3.686 | 127653 | 0.579 | 0.323 | 46352.6 | 610.98 |
| cfg_004 | 500 | 8 | 8 | 50000 | 0.5 | 0.1 | 4.209 | 122590 | 0.556 | 0.346 | 39791.9 | 499.98 |
| cfg_005 | 500 | 24 | 8 | 50000 | 0.5 | 0.1 | 4.209 | 122590 | 0.556 | 0.346 | 38551.2 | 499.99 |
| cfg_006 | 500 | 32 | 8 | 50000 | 0.5 | 0.1 | 4.209 | 122590 | 0.556 | 0.346 | 39983.0 | 499.99 |
| cfg_007 | 500 | 16 | 4 | 50000 | 0.5 | 0.1 | 4.251 | 134047 | 0.608 | 0.275 | 39386.3 | 499.97 |
| cfg_008 | 500 | 16 | 12 | 50000 | 0.5 | 0.1 | 4.238 | 113248 | 0.514 | 0.411 | 39243.0 | 499.99 |
| cfg_009 | 500 | 16 | 16 | 50000 | 0.5 | 0.1 | 4.214 | 98883 | 0.449 | 0.502 | 40898.2 | 499.99 |
| cfg_010 | 500 | 16 | 8 | 10000 | 0.5 | 0.1 | 4.220 | 128671 | 0.584 | 0.302 | 30011.5 | 499.99 |
| cfg_011 | 500 | 16 | 8 | 25000 | 0.5 | 0.1 | 4.236 | 125763 | 0.571 | 0.325 | 35174.2 | 499.99 |
| cfg_013 | 500 | 16 | 8 | 50000 | 0.25 | 0.1 | 4.226 | 123912 | 0.562 | 0.339 | 38200.9 | 499.99 |
| cfg_014 | 500 | 16 | 8 | 50000 | 0.75 | 0.1 | 4.220 | 122854 | 0.558 | 0.347 | 39513.0 | 499.99 |
| cfg_015 | 500 | 16 | 8 | 50000 | 0.5 | 0.025 | 4.243 | 125939 | 0.572 | 0.332 | 39200.8 | 499.99 |

Do not treat the frontier as a final selection: choose among these only after repeating on additional segments/sequences and checking intermediate state behavior. The algorithm remains unfrozen.

## Limitations

APS detections are the endpoint reference, not ground-truth feature identities. EPE only includes successful one-to-one matches; failure rate and survival must be read alongside it. The sweep is one-factor-at-a-time, limited to this sequence and segment, and uses software runtime. It does not establish statistical significance or freeze the algorithm.
