# @stonemind/brand

StoneMind's brand: the Lit Stone logos, the theme tokens (Washi & Shu, Hinoki & Indigo, Night), the glass material, and HTML/CSS component references. Read [BRAND.md](BRAND.md) first.

Specification: BRAND.md. The pattern follows [smalt-brand](https://github.com/stenhoor/smalt-brand).

## Depends on

None. No runtime or build dependencies: the token build and the tests use Node's own `node:test`.

## Used by

Nothing yet. Planned: `apps/web` (stonemind.app) and `apps/site` (stonemind.org). Both load `tokens.css` and `components.css` and use the logos.

## Deployment

Published as an npm package. Its files are plain CSS, JSON and SVG, so a site can also copy them unchanged. A consuming app copies them as they are and never edits its copy: changes are made here first.

## Development

Run these from this directory (`packages/brand`):

```bash
npm run build      # regenerate tokens/tokens.css from tokens/tokens.json
```

```bash
npm run check      # tokens.css up to date, then the contrast and asset tests
```

Open `components/*/preview.html` in a browser to see each component.

## Notes

- `tokens/tokens.json` is the source of truth. Never edit `tokens/tokens.css` by hand.
- The contrast tests are the readability rules in BRAND.md. A colour change that breaks one fails CI.
- The logo SVGs were generated once from Manrope 800 (Google Fonts, OFL) with fontTools, outside this repository. To regenerate them, outline the glyphs again; no font tooling is a dependency of this package.

See `docs/MONOREPO.md` for how this package fits with the other sub-projects.
