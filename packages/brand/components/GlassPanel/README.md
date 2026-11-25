# GlassPanel

The StoneMind material: a frosted panel that floats over the board and table (`.sm-table`).

| Layer | Token |
|---|---|
| Backdrop | `backdrop-filter: blur(var(--glass-blur)) saturate(var(--glass-saturate))` |
| Tint | `glass-tint` (opaque enough to keep text AA over every backdrop) |
| Edge | 1px `glass-edge`, inner top `glass-rim` |
| Sheen | `glass-sheen` fading out over the top 42% |
| Shadow | `glass-shadow`, tinted by the theme |

- `.sm-glass` is a panel. `.sm-glass.sm-bar` is a header or toolbar.
- Inside glass, secondary text uses `text-2` (`.sm-secondary`), never `text-3`.
- `.sm-well` is an inset area inside a panel.
- Never put glass over glass, and never animate the blur.
- For a logo inside a panel, include both versions with `.sm-logo-light` and `.sm-logo-dark`; the one for the active theme shows.
