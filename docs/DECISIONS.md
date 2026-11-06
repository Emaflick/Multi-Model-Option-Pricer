# Decision log

Format: `## D-NNN — title` · Context · Decision · Alternatives · Consequences.

## D-001 — Forward-based pricing kernels
**Context:** SPX has a discrete, uncertain dividend stream; r and q are not directly observed.
**Decision:** kernels take (F, DF, T); F and DF per expiry are implied from put-call parity on market quotes.
**Alternatives:** OIS curve + dividend-yield fit; futures-implied forward.
**Consequences:** carry assumptions do not leak into the vol surface; the parity fit doubles as a data-quality check.
