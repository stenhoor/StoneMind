# StoneMind Ontology Design Specification

Version: 0.1

Status: Foundational Draft

---

# Purpose

The StoneMind Ontology exists to model the collective understanding of Go and provide a framework for understanding players.

The ontology is not a joseki dictionary.

The ontology is not a KataGo analysis database.

The ontology is a representation of:

- Go concepts
- Relationships between concepts
- Prerequisites
- Dependencies
- Strategic interactions
- Learning pathways

The ontology serves as the foundation for:

- Player modeling
- Style modeling
- Coaching
- Teaching
- Study recommendations
- Goal planning

---

# Core Principle

StoneMind does not seek to determine the best move.

KataGo already does that.

StoneMind seeks to understand:

- Why players choose moves
- What concepts drive those choices
- Where understanding is strong
- Where understanding is weak
- How style influences decisions

The ontology exists to make this possible.

---

# Ontology Structure

StoneMind models three interconnected domains.

## Domain 1 - Go Knowledge

Represents the game itself.

Answers:

"What is known about Go?"

---

## Domain 2 - Player Understanding

Represents the player's relationship with concepts.

Answers:

"What does the player understand?"

---

## Domain 3 - Player Style

Represents the player's preferred approach.

Answers:

"How does the player choose to play?"

---

# High-Level Concept Categories

The following categories are preliminary.

These categories exist to organize thought.

They are not intended to be rigid silos.

---

## Tactical Concepts

Examples:

- Reading
- Life and Death
- Tesuji
- Capturing Races
- Ko Fights
- Tactical Shape

Primary Question:

Can the player correctly evaluate local situations?

---

## Strategic Concepts

Examples:

- Fuseki
- Joseki
- Direction of Play
- Influence
- Thickness
- Initiative
- Whole Board Judgment
- Balance

Primary Question:

Can the player correctly evaluate global situations?

---

## Fighting Concepts

Examples:

- Attack
- Defense
- Sacrifice
- Sabaki
- Weak Groups
- Strong Groups

Primary Question:

Can the player manage conflict effectively?

---

## Territorial Concepts

Examples:

- Territory
- Reduction
- Invasion
- Frameworks
- Expansion

Primary Question:

Can the player manage space efficiently?

---

## Endgame Concepts

Examples:

- Sente
- Gote
- Counting
- Yose
- Endgame Prioritization

Primary Question:

Can the player convert advantages into wins?

---

## Meta Concepts

Examples:

- Efficiency
- Balance
- Timing
- Flexibility
- Judgment

Primary Question:

How effectively does the player connect concepts together?

---

# Relationship Types

Concepts do not exist independently.

StoneMind models relationships.

---

## Supports

Example:

Thickness supports Attack

```text
Thickness
    SUPPORTS
Attack
```

---

## Influences

Example:

Influence affects Reduction Timing

```text
Influence
    INFLUENCES
Reduction
```

---

## Prerequisite

Example:

Basic Life-and-Death understanding is a prerequisite to successful invasions.

```text
Life and Death
    PREREQUISITE_FOR
Invasion
```

---

## Contrasts With

Example:

Immediate Territory contrasts with Long-Term Influence.

```text
Territory
    CONTRASTS_WITH
Influence
```

---

## Reinforces

Example:

Sabaki reinforces Flexible Shape.

---

## Emerges From

Example:

Direction of Play emerges from understanding fuseki, influence, shape, and whole-board judgment.

---

# Ontology Nodes

Each concept node contains:

```yaml
id:
name:
description:

category:

difficulty:

related_concepts:

prerequisites:

examples:

common_misunderstandings:

learning_resources:
```

Example:

```yaml
name: Thickness

category: Strategic

supports:
  - Attack
  - Influence Development

related:
  - Shape
  - Weak Groups

common_misunderstandings:
  - Overconcentration
  - Passive Defense

difficulty: Intermediate
```

---

# Player Graph

The Player Graph overlays the ontology.

The ontology describes Go.

The Player Graph describes the player.

---

# Player Concept Node

Example:

```yaml
player:
concept: Thickness

understanding: 78

application: 65

confidence: 82

growth: improving

last_updated:
```

---

# Understanding vs Application

StoneMind tracks both.

A player may:

Understand a concept.

But fail to apply it.

Example:

```text
Understanding:
High

Application:
Low
```

This becomes coaching information.

---

# Evidence Model

Player assessments require evidence.

---

## Evidence Sources

### Human Games

Most important source.

Measures:

- Practical application
- Competitive decision making

---

### Tsumego

Measures:

- Tactical understanding
- Reading ability

---

### Teaching Games

Measures:

- Adaptability
- Learning response

---

### Professional Study

Measures:

- Conceptual understanding

---

# Confidence

All assessments include confidence.

Example:

```yaml
invasion_understanding:

score: 71

confidence: 32
```

Low confidence means more evidence is needed.

StoneMind should avoid strong conclusions with weak evidence.

---

# Style Ontology

Style is separate from understanding.

A player can understand a concept without preferring it.

Likewise, a player may prefer concepts they do not yet fully understand.

---

# Style Dimensions

Initial candidates:

---

## Tactical ↔ Strategic

Does the player focus on local fights or larger plans?

---

## Aggressive ↔ Conservative

How willing is the player to create conflict?

---

## Territorial ↔ Influence

Which form of compensation does the player seek?

---

## Complexity ↔ Simplicity

Does the player increase or reduce board complexity?

---

## Local ↔ Whole Board

Does attention stay local or shift globally?

---

## Initiative ↔ Stability

Does the player pursue action or security?

---

## Risk Seeking ↔ Risk Avoiding

How much uncertainty is tolerated?

---

# Style Attributes

Each style dimension contains:

```yaml
dimension:

value:

confidence:

trend:

evidence:
```

Example:

```yaml
dimension: Aggressive

value: High

confidence: 91

trend: Stable
```

---

# Strengths

Strengths are not simply high skill.

A strength is:

- demonstrated
- repeatable
- useful

Example:

```text
Attack Planning

Understanding: High
Application: High
Style Preference: High

=> Strength
```

---

# Weaknesses

Weaknesses are not simply mistakes.

A weakness is a recurring limitation affecting player goals.

Example:

```text
Invasion Timing

Understanding: Low
Application: Low
Evidence: Strong

=> Weakness
```

---

# Style-Limiting Weaknesses

Special category.

A weakness that directly hinders the player's preferred style.

Example:

Player Style:

Aggressive Fighting

Weakness:

Life-and-Death Reading

Result:

Frequent failed attacks.

Priority:

Critical
```

The player should not stop attacking.

The player should improve the skill supporting attacks.

---

# Coaching Priorities

StoneMind should optimize for:

1. Player goals.
2. Style preservation.
3. Weakness reduction.
4. Strength amplification.
5. Long-term understanding.

Not every weakness deserves equal attention.

---

# Future Research

Open questions:

## Ontology

- Concept granularity
- Dependency modeling
- Expert consensus process

## Style

- Measurement techniques
- Stability metrics
- Evolution modeling

## Evidence

- Weighting systems
- Confidence calculations
- Contradictory evidence handling

## Coaching

- Recommendation generation
- Goal adaptation
- Personalized learning plans

---

# Ultimate Objective

The ontology should eventually allow StoneMind to answer:

- What kind of player is this?
- What does this player understand?
- What doesn't this player understand?
- Why do they make the mistakes they make?
- What strengths define them?
- What weaknesses hold them back?
- What should they study next?
- How can they become a stronger version of the player they already are?

The purpose of the ontology is not to model Go.

The purpose of the ontology is to help players unlock their Go.

