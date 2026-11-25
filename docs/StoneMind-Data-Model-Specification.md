# StoneMind Data Model Specification

Version: 0.1

Status: Foundational Draft

---

# Purpose

The StoneMind Data Model defines the core entities, relationships, and information structures used by the platform.

It provides the bridge between:

- Project Philosophy
- Go Ontology
- Technical Architecture
- User Experience

The Data Model represents the stable conceptual foundation of StoneMind.

Technology choices may change.

AI providers may change.

Storage systems may change.

The core data model should remain stable and evolve slowly.

---

# Design Principles

## Player-Centric

The central entity in StoneMind is the Player.

Everything else exists to help understand and support the player's journey.

---

## Evidence-Based

All conclusions should be rooted in evidence.

StoneMind should not guess.

StoneMind should collect observations and build confidence over time.

---

## Explainable

Every recommendation should be traceable.

The system should be capable of answering:

- Why was this recommendation made?
- What evidence supports this conclusion?
- What assumptions were used?

---

## Extensible

The model should support:

- New concept types
- New AI providers
- New evidence sources
- New coaching strategies

without redesigning the system.

---

# Core Domains

StoneMind consists of five major domains:

```text
Player Domain
Go Ontology Domain
Style Domain
Evidence Domain
Resource Domain
```

All coaching behavior emerges from the interaction of these domains.

---

# Domain: Player

Represents an individual Go player.

## Player

```yaml
Player:
  id:
  displayName:

  createdDate:
  lastActiveDate:

  rankClaimed:
  rankEstimated:

  preferences:

  activeGoals:
  archivedGoals:

  profileMetadata:
```

---

## Player Profile

Represents long-term identity.

Contains:

- Preferences
- Learning history
- Milestones
- Achievements
- Study habits

This is descriptive information.

Not assessment information.

---

# Domain: Goals

Players define success.

---

## Goal

```yaml
Goal:
  id:

  title:

  description:

  category:

  status:

  priority:

  startDate:

  targetDate:

  successCriteria:
```

Examples:

```text
Reach 1 Dan

Improve invasions

Prepare for state tournament

Understand professional games

Improve fighting skill
```

---

## Goal Categories

Examples:

```text
Strength Improvement

Tournament Preparation

Opponent Preparation

Concept Mastery

Teaching

Exploration

Personal Enjoyment
```

---

# Domain: Go Ontology

Represents StoneMind's understanding of Go.

---

## Concept

```yaml
Concept:
  id:

  name:

  description:

  category:

  difficulty:

  metadata:
```

Examples:

```text
Thickness

Influence

Reduction

Invasion

Joseki

Fuseki

Shape

Sabaki
```

---

## Concept Relationship

```yaml
Relationship:
  id:

  sourceConcept:

  targetConcept:

  relationshipType:

  strength:
```

---

## Relationship Types

Examples:

```text
SUPPORTS

INFLUENCES

PREREQUISITE_FOR

REINFORCES

EMERGES_FROM

CONTRASTS_WITH

DEPENDS_ON
```

---

# Domain: Player Understanding

Represents the player's relationship with concepts.

---

## Concept Assessment

```yaml
ConceptAssessment:

  playerId:

  conceptId:

  understanding:

  application:

  confidence:

  trend:

  lastUpdated:
```

---

# Understanding

Measures:

```text
Conceptual understanding
```

Can the player demonstrate knowledge?

---

# Application

Measures:

```text
Practical execution
```

Can the player use the concept in games?

---

# Confidence

Measures:

```text
How certain StoneMind is.
```

High confidence requires substantial evidence.

---

# Domain: Style

Represents how the player chooses to play.

Style is separate from skill.

---

## Style Dimension

```yaml
StyleDimension:
  id:

  name:

  leftExtreme:

  rightExtreme:
```

Examples:

```text
Aggressive ↔ Conservative

Tactical ↔ Strategic

Territory ↔ Influence

Complicated ↔ Simplifying

Local ↔ Whole Board

Risk Seeking ↔ Risk Avoiding
```

---

## Style Assessment

```yaml
StyleAssessment:

  playerId:

  dimensionId:

  value:

  confidence:

  trend:

  evidenceCount:
```

---

## Style Evolution

Tracks changes over time.

Examples:

```text
Increasing aggression

Improving strategic focus

Experimenting with influence-based play
```

---

# Domain: Strengths

Represents demonstrated capabilities.

---

## Strength

```yaml
Strength:
  id:

  playerId:

  conceptId:

  evidenceWeight:

  confidence:

  impact:
```

---

# Domain: Weaknesses

Represents recurring limitations.

---

## Weakness

```yaml
Weakness:
  id:

  playerId:

  conceptId:

  evidenceWeight:

  confidence:

  impact:
```

---

# Style-Limiting Weakness

A special class of weakness.

Definition:

A weakness that directly prevents a player from successfully expressing their preferred style.

Example:

```text
Player Style:
Aggressive Fighting

Weakness:
Reading Accuracy

Impact:
Attacks often fail.
```

These should receive elevated coaching priority.

---

# Domain: Evidence

Evidence drives understanding.

---

## Observation

Atomic unit of evidence.

```yaml
Observation:
  id:

  source:

  timestamp:

  summary:

  associatedConcepts:

  confidence:
```

---

## Evidence Source Types

```text
Human Game

Teaching Game

Tsumego

Problem Set

Professional Study

Journal Entry

Teacher Review

User Feedback
```

---

## Evidence Record

```yaml
Evidence:
  id:

  playerId:

  observationId:

  concepts:

  weight:

  confidence:
```

---

# Domain: Games

Historical game records.

---

## Game

```yaml
Game:
  id:

  sgfReference:

  date:

  opponent:

  source:

  result:

  metadata:
```

---

## Game Analysis

```yaml
GameAnalysis:
  id:

  gameId:

  provider:

  findings:

  concepts:

  summary:
```

Examples:

```text
KataGo

OGS AI Review

Human Teacher Review
```

---

# Domain: Tsumego

Tracks problem-solving performance.

---

## Problem

```yaml
Problem:
  id:

  title:

  source:

  concepts:

  difficulty:
```

---

## Problem Attempt

```yaml
ProblemAttempt:
  id:

  problemId:

  playerId:

  result:

  attemptTime:

  hintsUsed:

  notes:
```

---

# Domain: Teaching Games

Interactive learning experiences.

---

## Teaching Session

```yaml
TeachingSession:
  id:

  playerId:

  provider:

  startDate:

  endDate:

  objectives:

  findings:
```

---

# Domain: Resources

Resources are external sources of knowledge.

---

## Resource

```yaml
Resource:
  id:

  title:

  type:

  author:

  location:
```

---

## Resource Types

Examples:

```text
Book

Article

Video

Professional Game

Course

Problem Collection

Study Note

Journal Entry
```

---

## Resource Relationship

Connects knowledge to learning resources.

```yaml
ResourceRelationship:

  resourceId:

  conceptId:

  relationshipType:
```

Examples:

```text
TEACHES

EXPLAINS

DEMONSTRATES

ILLUSTRATES
```

---

# Domain: Recommendations

Recommendations are coaching outputs.

---

## Recommendation

```yaml
Recommendation:
  id:

  playerId:

  goalId:

  category:

  rationale:

  supportingEvidence:

  confidence:

  priority:
```

---

## Recommendation Categories

```text
Study

Practice

Review

Reading

Teaching Game

Problem Set

Tournament Preparation

Opponent Preparation
```

---

# Domain: Study Plans

Structured coaching plans.

---

## Study Plan

```yaml
StudyPlan:
  id:

  playerId:

  goalId:

  createdDate:

  summary:
```

---

## Study Activity

```yaml
StudyActivity:
  id:

  studyPlanId:

  type:

  description:

  estimatedDuration:

  linkedConcepts:
```

---

# Entity Relationship Overview

```text
Player
    │
    ├── Goals
    │
    ├── Concept Assessments
    │
    ├── Style Assessments
    │
    ├── Strengths
    │
    ├── Weaknesses
    │
    ├── Recommendations
    │
    └── Study Plans

Concepts
    │
    ├── Relationships
    │
    └── Resources

Evidence
    │
    ├── Games
    ├── Problems
    ├── Teaching Sessions
    └── Studies

Evidence
    │
    ▼

Player Graph Updates

Player Graph
    +
Style Graph
    +
Goals

    ▼

Coaching Engine

    ▼

Recommendations
```

---

# Future Extensions

Potential future entities:

```text
Opponent Profile

Tournament

Teacher

Study Group

Opening Repertoire

AI Persona

Book Notes

Voice Sessions

Video Analysis
```

These should not be required for V1.

---

# Data Model Success Criteria

The Data Model succeeds when StoneMind can reliably answer:

- What kind of player is this?
- What are their goals?
- What are their strengths?
- What are their weaknesses?
- What evidence supports those conclusions?
- What style do they prefer?
- Which weaknesses most hinder that style?
- What should they study next?
- Why did StoneMind recommend that?
- How has the player evolved over time?

The ultimate purpose of the data model is to support StoneMind's mission:

**Help every player understand themselves, strengthen their game, and unlock their Go.**
