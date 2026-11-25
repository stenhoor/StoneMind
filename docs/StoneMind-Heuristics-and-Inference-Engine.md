# StoneMind Heuristics & Inference Engine Specification

Version: 0.1

Status: Research Draft

---

# Purpose

The Heuristics & Inference Engine is responsible for transforming evidence into understanding.

This is the core intelligence layer of StoneMind.

It interprets:

- Human games
- Tsumego performance
- Teaching games
- Study activity
- Professional game analysis

and updates:

- Player Graph
- Style Graph
- Goal Progress
- Recommendations

The purpose of this engine is not to identify the best move.

The purpose is to understand the player.

---

# Core Principle

StoneMind does not directly observe understanding.

StoneMind observes evidence.

Understanding is inferred.

Example:

StoneMind cannot observe:

    "Player misunderstands thickness."

StoneMind can observe:

    - Failed invasion
    - Multiple failed reductions
    - Consistently attacking from weak groups

and infer:

    Possible weakness in thickness evaluation.

---

# Inference Pipeline

```text
Evidence
     ↓

Observations
     ↓

Indicators
     ↓

Hypotheses
     ↓

Confidence Calculations
     ↓

Graph Updates
     ↓

Recommendations
```

---

# Evidence Sources

## Human vs Human Games

Highest value evidence source.

Measures:

- Real decisions
- Competitive performance
- Practical application

Weighting:

Very High

---

## Tsumego

Measures:

- Reading
- Tactical understanding
- Pattern recognition

Weighting:

High

---

## Teaching Games

Measures:

- Ability to learn
- Adaptation
- Instruction application

Weighting:

Medium-High

---

## Professional Study

Measures:

- Conceptual understanding

Weighting:

Medium

---

## Self Reflection

Measures:

- Self-awareness
- Motivation

Weighting:

Low

Never used as sole evidence.

---

# Observation Model

Observations are atomic evidence.

Examples:

```yaml
Observation:
  source: Game

  type:
    FailedInvasion

  confidence:
    High
```

```yaml
Observation:
  source: Tsumego

  type:
    RepeatedFailure

  concept:
    LifeAndDeath
```

Observations are facts.

No interpretation occurs here.

---

# Indicator Model

Indicators group observations.

Example:

```text
Failed Invasion

Failed Invasion

Failed Invasion

Incorrect Reduction

⇒ Indicator:
Poor Invasion Performance
```

Indicators represent patterns.

---

# Hypothesis Model

StoneMind forms hypotheses.

Examples:

```text
Hypothesis:
Weak Invasion Judgment
```

```text
Hypothesis:
Weak Thickness Evaluation
```

```text
Hypothesis:
Overly Aggressive Risk Taking
```

Hypotheses are provisional.

They require confidence.

---

# Confidence Framework

Every conclusion must have confidence.

---

## Low Confidence

```text
1-3 observations
```

StoneMind may say:

> Current evidence suggests...

---

## Moderate Confidence

```text
Repeated observations

Multiple evidence sources
```

StoneMind may say:

> Evidence indicates...

---

## High Confidence

```text
Long-term patterns

Cross-source confirmation
```

StoneMind may say:

> Strong evidence supports...
```

---

# Cross-Validation

Strongest conclusions require multiple sources.

Example:

```text
Game Reviews:
Invasion failures

Tsumego:
Invasion failures

Teaching Games:
Invasion failures
```

Confidence increases substantially.

---

# Understanding vs Application

StoneMind tracks both.

---

## Understanding

Can the player explain the concept?

Can they identify the concept?

Can they solve isolated examples?

---

## Application

Can the player use the concept during actual games?

Stress matters.

Time pressure matters.

Complexity matters.

---

# Example

```text
Life & Death

Understanding:
82

Application:
55
```

Conclusion:

Player understands the concept but struggles to apply it.

Coaching differs significantly from a player who scores:

```text
Understanding:
45

Application:
40
```

---

# Style Inference

Style is inferred.

Never declared.

---

# Style Evidence

Examples:

Repeated early fighting

Repeated invasions

High-risk joseki selection

Sacrificial play

Whole-board strategic exchanges
```

These are indicators.

Not conclusions.

---

# Style Hypotheses

Examples:

```text
Aggressive

Strategic

Influence-Oriented

Territorial

Complexity-Seeking

Simplification-Seeking
```

Each develops confidence.

---

# Weakness Identification

Weakness != Mistake

A weakness must be:

- Recurring
- Relevant
- Influential

---

## Not a Weakness

```text
One bad ladder.
```

---

## Potential Weakness

```text
Repeated ladder failures.
```

---

## Confirmed Weakness

```text
Repeated ladder failures

Across multiple games

Across multiple months

Supported by tsumego performance
```

---

# Strength Identification

Strength != Success

A strength must be:

- Repeatable
- Demonstrated
- Reliable

---

# Example

```text
Attack Planning

Strong in games

Strong in reviews

Strong in teaching games

⇒ Strength
```

---

# Style-Limiting Weaknesses

Special class of weakness.

Defined as:

A weakness that directly undermines
the player's preferred style.

---

# Example

Style:

Aggressive fighter

Weakness:

Life and death

Result:

Attacks fail to convert.

Priority:

Critical
```

StoneMind should prioritize these weaknesses.

---

# style-Amplifying Strengths

Special class of strength.

Defined as:

Strengths that reinforce the player's preferred style.

---

Example:

```text
Style:
Strategic

Strength:
Whole-board judgment
```

StoneMind should actively reinforce these strengths.

---

# Root Cause Analysis

StoneMind should search beyond symptoms.

---

## Example

Observed:

Failed invasion

Possible causes:

- Reading
- Life and death
- Thickness understanding
- Risk assessment
- Reduction timing
```

StoneMind should avoid assigning immediate blame.

Instead:

```text
Generate hypotheses

Collect evidence

Update confidence
```

---

# Recommendation Generation

Recommendations are evidence-based.

---

Formula:

```text
Player Graph

+
Style Graph

+
Goals

+
Evidence

↓

Recommendations
```

---

# Example

Goal:

Improve tournament performance

Evidence:

Weak invasion timing

Strong influence play

Style:

Strategic
```

Recommendation:

Do not become more aggressive.

Study:

- Reduction timing
- Framework evaluation
- Invasion judgment
```

---

# Recommendation Prioritization

Order:

1. Goal blockers
2. Style-limiting weaknesses
3. Major weaknesses
4. Strength reinforcement
5. Exploration opportunities

---

# Goal-Aware Inference

Identical evidence may produce different recommendations.

---

Player A Goal:

Reach 1 Dan

Recommendation:

Focused improvement plan.

---

Player B Goal:

Understand professional games

Recommendation:

Concept exploration.
```

Same evidence.

Different coaching.

---

# Uncertainty Principle

StoneMind should never represent assumptions as facts.

For every conclusion, track:

- Evidence count
- Confidence
- Contradictory evidence
- Last observation

---

# Explainability Requirement

Every recommendation must answer:

Why?

Every graph update must answer:

Why?

Every weakness must answer:

Why?

Every strength must answer:

Why?

StoneMind conclusions must always be traceable.

---

# Future Research

Topics requiring investigation:

- Weighting algorithms
- Bayesian confidence models
- Style emergence metrics
- Concept dependency scoring
- Contradictory evidence handling
- Longitudinal player development
- Recommendation effectiveness

---

# Success Criteria

The Heuristics & Inference Engine succeeds when StoneMind can answer:

- What does this player understand?
- What doesn't this player understand?
- How do they prefer to play?
- What evidence supports those conclusions?
- What weaknesses are most important?
- Which weaknesses inhibit their style?
- Which strengths define them?
- What should they study next?
- Why?

This engine is the reasoning layer that transforms StoneMind from a collection of tools into a genuine coaching system dedicated to helping players unlock their Go.
