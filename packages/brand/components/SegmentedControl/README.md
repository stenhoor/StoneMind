# SegmentedControl

Two to four mutually exclusive views, such as Review, Study and Goals.

- A `role="group"` with an `aria-label`, holding `<button>`s.
- The selected button has `aria-pressed="true"`. The style follows the attribute, so the state and its look can't disagree.
- The page's own code sets `aria-pressed`. There is no script in the brand package.
