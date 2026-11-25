# StoneMind Charter

Version: 0.1
Status: Draft
Project Type: Open Research and Development Initiative

---

# Vision

StoneMind exists to help players unlock their Go.

Rather than treating improvement as a journey toward a single idealized style of play, StoneMind seeks to understand each player as an individual: their strengths, weaknesses, preferences, habits, goals, study history, and style of play.

StoneMind adapts to the player's needs, acting as a coach, tutor, teacher, trainer, analyst, practice partner, and study companion. Its purpose is not simply to teach Go, but to help every player become the strongest and most complete version of the player they aspire to be.

---

# Mission

Build an adaptive AI-powered Go companion that:

- Understands the game of Go through an expert-derived ontology.
- Understands the player through observation and analysis.
- Understands how the player approaches the game through style modeling.
- Helps players achieve goals that they define.
- Preserves player ownership of data, knowledge, and decisions.
- Integrates with existing Go tools and AI ecosystems.

---

# Core Philosophy

## Unlock Your Go

StoneMind does not attempt to teach a single "correct" way to play.

Players approach Go differently.

Some love fighting.
Some love influence.
Some love territorial play.
Some love professional game study.
Some seek competitive improvement.
Some seek deeper understanding.

StoneMind helps players discover and strengthen their own relationship with Go.

---

## Player-Defined Goals

The player defines success.

Examples:

- Become stronger
- Reach a specific rank
- Prepare for a tournament
- Beat a challenging opponent
- Improve a specific skill
- Understand professional games
- Develop a new style
- Deepen appreciation of Go

StoneMind adapts its recommendations to the player's goals.

---

## Strengths Matter

StoneMind studies:

- Weaknesses
- Strengths
- Preferences
- Tendencies
- Style

A weakness should be strengthened.

A strength should be amplified.

A style should be understood.

---

## Style Is Not Weakness

Differences in play style are not inherently defects.

StoneMind seeks to distinguish:

- Weaknesses
- Preferences
- Habits
- Strategies
- Personal expression

The goal is to refine a player's style, not replace it.

---

# Project Principles

## Player Ownership

Players own:

- Game records
- Reviews
- Notes
- Journals
- Knowledge graphs
- Study plans
- Style profiles
- Training history

StoneMind never assumes ownership.

---

## Bring Your Own AI

StoneMind integrates with:

- Local LLMs
- Cloud AI providers
- Self-hosted AI systems
- Future AI providers

StoneMind does not require a specific AI provider.

---

## Bring Your Own Knowledge

StoneMind embraces:

- SGF collections
- EPUB books
- Markdown notes
- Teaching materials
- Problem collections
- Professional game collections

Players bring the resources they already own.

---

## Open Formats First

Preferred formats include:

- SGF
- Markdown
- EPUB
- JSON
- CSV

Proprietary formats should be avoided when practical.

---

## Integrate Before Replacing

StoneMind prefers integrating with existing systems.

Examples:

- OGS
- KataGo
- Sabaki
- KaTrain
- Existing problem collections
- Existing study workflows

StoneMind complements tools rather than replacing them.

---

# Conceptual Architecture

StoneMind consists of three primary knowledge systems.

## 1. Go Ontology

Represents:

- Concepts
- Principles
- Relationships
- Dependencies
- Learning paths

Example concepts:

- Fuseki
- Joseki
- Shape
- Direction of Play
- Influence
- Thickness
- Reduction
- Invasion
- Attack
- Defense
- Sabaki
- Life and Death
- Tesuji
- Endgame

---

## 2. Player Understanding Graph

Represents:

- Demonstrated understanding
- Evidence
- Skill application
- Growth
- Confidence

This graph answers:

"What does the player understand?"

---

## 3. Player Style Graph

Represents:

- Strategic tendencies
- Tactical tendencies
- Risk appetite
- Positional preferences
- Long-term style evolution

This graph answers:

"How does this player choose to play?"

---

# Evidence Sources

StoneMind develops understanding through evidence.

## Human vs Human Games

Primary source of evidence.

Measures:

- Decision making
- Pattern application
- Strategic understanding
- Real-world performance

## Tsumego and Problems

Measures:

- Reading
- Pattern recognition
- Tactical skill
- Improvement over time

## Teaching Games

Measures:

- Learning behavior
- Adaptability
- Application of instruction

## Professional Study

Measures:

- Conceptual understanding
- Strategic comprehension
- Go appreciation

---

# Coaching Model

StoneMind adapts to player needs.

Possible roles include:

## Coach

Reviews games and identifies opportunities.

## Tutor

Explains specific concepts.

## Teacher

Provides instruction and structured learning.

## Trainer

Creates and manages study plans.

## Analyst

Evaluates games and trends.

## Practice Partner

Provides exercises and teaching games.

---

# Non-Goals

StoneMind is not:

- A Go server.
- A replacement for OGS.
- A social network.
- A ranking authority.
- A proprietary AI platform.
- A commercial data-harvesting service.
- A tool that enforces a single style of play.

---

# Technical Principles

## Local First

Whenever practical:

- Data remains local.
- Processing remains local.
- AI choice remains local.

---

## User Consent

Before sharing data externally:

- The player understands what is shared.
- The player understands where it is shared.
- The player approves transmission.

---

## AI Provider Independence

Any AI service should be replaceable.

No critical subsystem should depend on a specific AI provider.

---

# Initial Architecture

## Storage Layer

Smaltworks Platform

Components:

- Relational Database
- Graph Database
- DataFrame Analytics

Persistence:

- OPFS

---

## Integration Layer

Providers:

- KataGo
- OGS
- OpenAI
- Anthropic
- Ollama
- Gemini
- Future Providers

---

## Core Engine

Components:

- Ontology Engine
- Player Graph Engine
- Style Engine
- Coaching Engine
- Recommendation Engine

---

# Invariants

The following principles define StoneMind:

1. Players own their data.
2. Players define their goals.
3. Style diversity is valuable.
4. Open formats are preferred.
5. Integrations are preferred over replacement.
6. AI providers are replaceable.
7. Understanding the player is more important than analyzing moves.
8. StoneMind exists to help players unlock their Go.
