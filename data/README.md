# Data provenance

`sources.json` records exact download URLs, SHA-256 checksums, and byte counts.
GitHub sources are pinned to a commit. Each run verifies even cached files.
Raw downloads live in `data/raw/` and are excluded from version control.
Generated row-level COVID data live in `data/processed/` and are also excluded.
The first run needs internet; subsequent runs use the verified local cache.
Third-party datasets remain subject to their source terms; see the main README.
