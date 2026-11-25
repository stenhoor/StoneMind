# StoneMind Knowledge Base

Version: 0.1

Status: Living Document

Purpose:
Capture project understanding, philosophy, design decisions, assumptions, open questions, and emerging knowledge that may not belong in formal specifications.

This document acts as the institutional memory of StoneMind.

---

# Why StoneMind Exists

Many Go tools analyze games.

Few seek to understand players.

StoneMind exists to help players unlock their Go.

The objective is not to teach a single correct way to play.

The objective is to help players understand:

- Their strengths
- Their weaknesses
- Their habits
- Their goals
- Their style
- Their relationship with Go

StoneMind helps players become stronger versions of themselves.

---

# Core Insight

Most tools answer:

"What move should have been played?"

StoneMind attempts to answer:

"Why does this player repeatedly encounter this problem?"

"What kind of player are they?"

"What should they study next?"

"What is preventing growth?"

"What strengths should be amplified?"

---

# Unlock Your Go

StoneMind rejects the assumption that all players should converge toward a single playing style.

Strong players often express Go differently.

Some thrive through:

- Fighting
- Influence
- Territory
- Reading
- Endgame
- Whole-board judgment
- Creativity

StoneMind seeks to preserve player identity while supporting improvement.

---

# Weaknesses vs Preferences

One of the most important distinctions in StoneMind.

A repeated behavior is not necessarily a weakness.

Examples:

Deep invasions may indicate:

- poor judgment

or

- a strong aggressive style

The system must distinguish between:

- weakness
- preference
- experimentation
- habit
- strategic choice

before making recommendations.

---

# Style as a First-Class Concept

Style is not a secondary attribute.

Style is one of the central models in StoneMind.

StoneMind models:

1. Go
2. The Player's Understanding of Go
3. The Player's Style of Go

All three interact.

---

# The Three Graphs

## Go Ontology

Represents:

The game's concepts and relationships.

Questions answered:

- What exists in Go?
- How do concepts interact?
- What concepts support other concepts?

---

## Player Graph

Represents:

The player's understanding and application.

Questions answered:

- What does the player understand?
- What can they apply?
- Where are strengths emerging?
- Where are weaknesses emerging?

---

## Style Graph

Represents:

The player's preferred way of approaching the game.

Questions answered:

- How does this player like to play?
- What tradeoffs do they prefer?
- What risks do they embrace?
- How is their style evolving?

---

# Future Fourth Graph

Possible future addition.

## Resource Graph

Represents:

Connections between knowledge resources and Go concepts.

Examples:

Book
    ->
Concept

Professional Game
    ->
Concept

Problem Collection
    ->
Concept

Journal Entry
    ->
Concept

The Resource Graph enables targeted study recommendations.

---

# The Role Model

StoneMind does not have a single role.

Its role depends on the player's needs.

---

## Coach

When reviewing performance.

---

## Tutor

When teaching a concept.

---

## Teacher

When delivering structured understanding.

---

## Trainer

When helping achieve goals.

---

## Analyst

When examining evidence.

---

## Practice Partner

When helping apply learning.

---

# Goal-Driven Behavior

The player defines success.

Not StoneMind.

Examples:

- Reach 1 Dan
- Prepare for tournament
- Improve fighting
- Understand professional games
- Beat a regular opponent
- Learn influence-based play

The same player may have different goals at different times.

Recommendations should adapt accordingly.

---

# Evidence Hierarchy

Not all evidence is equally valuable.

Current understanding:

1. Human vs Human Games
2. Tsumego Performance
3. Teaching Games
4. Professional Study
5. Self Assessment

This ordering should be revisited through research.

---

# Bring Your Own AI

StoneMind is not an AI provider.

StoneMind uses AI providers.

Supported approaches may include:

- Local LLMs
- OpenAI
- Anthropic
- Gemini
- Ollama
- Future systems

The platform remains AI-provider agnostic.

---

# Bring Your Own Knowledge

Players should be able to bring:

- SGF collections
- EPUB books
- Markdown notes
- Journals
- Teacher comments
- Problem collections

StoneMind should enhance these resources, not replace them.

---

# Open Standards

Preferred formats:

Games:
- SGF

Books:
- EPUB

Notes:
- Markdown

Configuration:
- JSON

Export:
- JSON / CSV / Markdown

---

# Player-Owned Data

A foundational principle.

The player owns:

- Profile
- Notes
- Journals
- Games
- Recommendations
- Style assessments
- Study history
- Knowledge graph

StoneMind acts as a steward rather than an owner.

---

# Integrate Before Replace

StoneMind should integrate with existing tools whenever practical.

Examples:

- OGS
- KataGo
- Sabaki
- KaTrain

The objective is to enrich workflows, not replace them.

---

# Local-First Philosophy

Preferred architecture:

Browser
    +
Local Storage
    +
Optional AI Services

The system should remain useful without cloud connectivity.

---

# Smaltworks Platform Decision

Current architectural direction:

Storage and analytics built on Smaltworks.

Benefits:

- SQL
- Cypher Graph
- DataFrames
- Browser execution
- OPFS persistence
- Local ownership

This aligns strongly with the project's principles.

---

# Research Areas

Several subsystems still require substantial research.

---

## Ontology Research

Questions:

- Concept granularity
- Relationship types
- Prerequisite mapping

---

## Style Research

Questions:

- What dimensions define style?
- How stable is style?
- How does style evolve?

---

## Heuristic Research

Questions:

- Evidence weighting
- Confidence scoring
- Root-cause analysis

---

## Coaching Research

Questions:

- Effective recommendation generation
- Training plan design
- Goal adaptation

---

# Open Questions

## Style Framework

Not fully defined.

Questions:

- Fundamental dimensions?
- Derived dimensions?
- Stable vs transient style?

---

## Resource Graph

Still conceptual.

Needs specification.

---

## Teaching Methodology

Still conceptual.

Needs formal design.

---

## Recommendation Prioritization

Needs empirical validation.

---

## Confidence Model

Needs formal mathematical framework.

---

# Architectural Invariants

The following should not change without deliberate project-wide review.

1. Players own their data.
2. Players define their goals.
3. Style is a first-class concept.
4. Open standards are preferred.
5. AI providers are replaceable.
6. Integrations are preferred over replacement.
7. Local-first operation is preferred.
8. Recommendations must be explainable.
9. Evidence drives conclusions.
10. StoneMind exists to help players unlock their Go.

---

# Project North Star

A successful StoneMind should be able to answer:

- What kind of player am I?
- What do I understand?
- What do I misunderstand?
- What strengths define me?
- What weaknesses limit me?
- How do I prefer to play?
- What should I study next?
- Why?
- Am I improving?
- How can I become the strongest version of the player I want to be?

Everything else exists in service of those questions.
