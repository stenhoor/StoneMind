# ThemePicker

Lets players choose their theme: System (follows the device's light or dark setting), Washi & Shu, Hinoki & Indigo, or Night.

- A `fieldset` of radio buttons with a `legend`, so it works with a keyboard and a screen reader as is.
- Choosing a theme sets `data-theme` on `<html>` and saves it under `stonemind.theme`. System removes the attribute, so `tokens.css` follows `prefers-color-scheme`.
- To avoid a flash of the wrong theme, apply the saved theme before first paint with this snippet in `<head>`, after `tokens.css`:

```html
<script>
  try { const t = localStorage.getItem('stonemind.theme'); if (t) document.documentElement.dataset.theme = t; } catch {}
</script>
```

The preview's script is the whole mechanism: no library and no framework.
