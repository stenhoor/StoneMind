# StoneMind Style Model Specification

Version: 0.1

Status: Research Draft

---

# Purpose

The Style Model defines how StoneMind understands the unique way a player approaches Go.

Style is separate from:

- Understanding
- Strength
- Weakness
- Rank
- Goals

A player may possess the same level of understanding as another player while expressing that understanding in a very different way.

The objective of the Style Model is to understand and preserve a player's individuality while supporting growth.

---

# Guiding Principle

StoneMind does not seek to eliminate style.

StoneMind seeks to understand style.

Improve style.

Support style.

Strengthen style.

A player's style is not a flaw.

A weakness is not automatically a style.

A style is not automatically a weakness.

---

# Style Definition

Style is defined as:

The collection of recurring decisions, priorities, risk tolerances, positional preferences, strategic tendencies, and aesthetic choices a player consistently exhibits.

---

# Style Categories

The following dimensions are proposed as initial style axes.

---

## Tactical ↔ Strategic

Questions:

- Does the player focus on local exchanges?
- Does the player prioritize long-term plans?

Indicators:

- Fighting frequency
- Whole-board exchanges
- Sacrifice behavior

---

## Aggressive ↔ Conservative

Questions:

- Does the player initiate conflict?
- Does the player seek stability?

Indicators:

- Invasions
- Attachments
- Contact play
- Reductions

---

## Territorial ↔ Influence

Questions:

- Does the player prefer secure territory?
- Does the player prefer future potential?

Indicators:

- Opening selection
- Joseki selection
- Framework usage

---

## Complexity ↔ Simplification

Questions:

- Does the player seek complications?
- Does the player simplify advantageous positions?

Indicators:

- Ko frequency
- Fighting acceptance
- Exchange decisions

---

## Local ↔ Whole Board

Questions:

- Does attention remain local?
- Does the player prioritize global considerations?

Indicators:

- Tenuki frequency
- Direction of play decisions

---

## Initiative ↔ Stability

Questions:

- Does the player seek sente?
- Does the player prioritize safety?

Indicators:

- Follow-up attacks
- Defensive choices

---

## Risk Seeking ↔ Risk Avoiding

Questions:

- How much uncertainty does the player tolerate?

Indicators:

- Deep invasions
- Ko threats
- Sacrifices
- Complicated fighting

---

# Style Emergence

Style is not selected.

Style is inferred.

StoneMind observes behavior over time.

Style emerges from evidence.

---

# Style Confidence

Each style dimension tracks:

```yaml
value:
confidence:
trend:
evidenceCount:
```

Example:

```yaml
dimension: Aggression

value: 78

confidence: 84

trend: Stable
```

---

# Stable Style

A style pattern observed consistently over long periods.

Examples:

- Three years of influence-oriented games
- Repeated preference for fighting

---

# Experimental Style

Temporary behavior.

Examples:

- Exploring new openings
- Studying territorial play

Style should not immediately shift due to experimentation.

---

# Style-Limiting Weaknesses

Definition:

A weakness that directly prevents successful expression of preferred style.

Example:

Style:
Aggressive Fighter

Weakness:
Life and Death

Impact:
Attacks frequently fail.

Priority:
High

---

# Style-Amplifying Strengths

Definition:

A strength that reinforces preferred style.

Example:

Style:
Strategic

Strength:
Whole Board Judgment

Impact:
Core identity-strength.

Priority:
Strengthen further.

