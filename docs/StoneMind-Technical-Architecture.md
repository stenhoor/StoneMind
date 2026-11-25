# StoneMind Technical Architecture

Version: 0.1

Status: Foundational Draft

---

# Purpose

This document defines the technical architecture of StoneMind.

The architecture exists to support the project's core principles:

- Unlock Your Go
- Player-Owned Data
- Bring Your Own AI
- Bring Your Own Knowledge
- Local-First Computing
- Open Formats
- Long-Term Portability

This document intentionally focuses on architectural principles rather than specific frameworks or libraries that may change over time.

---

# Guiding Principles

## Browser First

StoneMind should run in modern browsers without requiring server-side infrastructure.

Benefits:

- Easy deployment
- Zero installation option
- Accessibility
- Local execution
- Reduced operating costs

The browser is the primary runtime environment.

---

## Local First

Player data should remain local whenever possible.

Examples:

- Player Graph
- Style Graph
- Study History
- Journals
- Notes
- Resource Library
- Goals
- Recommendations

These should function without cloud services.

---

## TypeScript First

Primary implementation language:

```text
TypeScript
```

Rationale:

- Browser-native ecosystem
- Shared code between front-end and local services
- Excellent tooling
- Strong typing
- Good AI-assisted development support

Design preference:

```text
All major business logic should be implementable in TypeScript.
```

---

## Provider Independence

StoneMind should not depend on any single AI vendor.

Every external service is optional.

Players may choose:

- Cloud AI providers
- Local AI providers
- Self-hosted services
- No AI services at all

---

## Open Standards

Wherever practical, use open formats.

Preferred Formats:

| Purpose | Format |
|----------|----------|
| Games | SGF |
| Notes | Markdown |
| Books | EPUB |
| Configuration | JSON |
| Export | JSON |
| Tabular Data | CSV |
| Reports | Markdown / HTML |

---

## Privacy By Default

Player data belongs to the player.

No data should be transmitted externally without explicit consent.

---

# Architectural Overview

```text
+------------------------------------------------+
|                 User Experience                |
+------------------------------------------------+
|                                                |
| Coach                                          |
| Tutor                                          |
| Teacher                                        |
| Trainer                                        |
| Analyst                                        |
| Practice Partner                              |
|                                                |
+------------------------+-----------------------+
                         |
                         v
+------------------------------------------------+
|                 StoneMind Core                 |
+------------------------------------------------+
|                                                |
| Ontology Engine                                |
| Player Graph Engine                            |
| Style Engine                                   |
| Evidence Engine                                |
| Coaching Engine                                |
| Recommendation Engine                          |
| Study Planning Engine                          |
| Resource Engine                                |
|                                                |
+------------------------+-----------------------+
                         |
                         v
+------------------------------------------------+
|               Smaltworks Data Layer            |
+------------------------------------------------+
|                                                |
| SQL                                             |
| Graph (Cypher)                                  |
| DataFrames                                      |
|                                                |
+------------------------+-----------------------+
                         |
                         v
+------------------------------------------------+
|                  Persistence                   |
+------------------------------------------------+
|                                                |
| OPFS                                            |
| Local File Imports                              |
| Local Backups                                   |
|                                                |
+------------------------+-----------------------+
                         |
                         v
+------------------------------------------------+
|            Integrations & Providers            |
+------------------------------------------------+
|                                                |
| KataGo                                          |
| OGS                                             |
| OpenAI                                          |
| Claude                                          |
| Gemini                                          |
| Ollama                                          |
| Future Providers                               |
|                                                |
+------------------------------------------------+
```

---

# Runtime Architecture

StoneMind should support multiple deployment models.

---

## Mode 1 - Browser Only

```text
Browser

    +
OPFS

    +
Remote AI Services
```

Simplest deployment model.

Suitable for casual users.

---

## Mode 2 - Local AI

```text
Browser

    +
OPFS

    +
Local KataGo

    +
Local LLM
```

Maximum privacy.

No cloud dependency required.

---

## Mode 3 - Hybrid

```text
Browser

    +
Local Data

    +
Remote AI

    +
Local KataGo
```

Expected to be the most common configuration.

---

# Storage Architecture

---

## Primary Storage

Technology:

```text
OPFS
```

Purpose:

- Player ownership
- Offline operation
- Browser-native persistence

---

## Data Categories

### Relational Data

Stored in SQL.

Examples:

- Players
- Games
- Goals
- Study Plans
- Reviews
- Journals
- Resources

---

### Graph Data

Stored in Graph Database.

Examples:

- Go Ontology
- Player Graph
- Style Graph
- Resource Graph

Queries:

```cypher
Player -> Weakness

Concept -> Prerequisite

Resource -> Concept

Style -> Behavior
```

---

### Analytical Data

Stored in DataFrames.

Examples:

- Performance Trends
- Tsumego Success Rates
- Review Metrics
- Style Evolution
- Study Effectiveness

---

# Core System Components

---

## Ontology Engine

Responsibilities:

- Manage Go Ontology
- Concept relationships
- Learning dependencies
- Concept metadata

Inputs:

- Research
- Expert contributions

Outputs:

- Ontology queries
- Dependency information

---

## Player Graph Engine

Responsibilities:

- Track player understanding
- Track mastery
- Track application
- Track confidence

Outputs:

- Strengths
- Weaknesses
- Growth trends

---

## Style Engine

Responsibilities:

- Track style preferences
- Detect style shifts
- Differentiate preference from weakness

Outputs:

- Style assessments
- Style evolution reports

---

## Evidence Engine

Responsibilities:

- Collect observations
- Process reviews
- Process tsumego results
- Process teaching games

Outputs:

- Evidence records
- Graph updates

---

## Coaching Engine

Responsibilities:

- Interpret player state
- Interpret goals
- Generate coaching guidance

Possible roles:

- Coach
- Tutor
- Teacher
- Trainer
- Analyst

---

## Recommendation Engine

Responsibilities:

- Generate study activities
- Identify priorities
- Connect concepts to resources

Outputs:

- Study plans
- Reading suggestions
- Problem assignments

---

## Resource Engine

Responsibilities:

- Manage books
- Manage notes
- Manage articles
- Connect resources to concepts

Supported formats:

- EPUB
- Markdown
- HTML
- PDF (read-only support)

---

# Integration Architecture

---

## Data Source Providers

Purpose:

Provide evidence.

Examples:

- SGF Import
- OGS
- KataGo Analysis Files
- Problem Collections
- Study Journals

Interface:

```typescript
interface EvidenceProvider {
    import(): Promise<EvidenceRecord[]>;
}
```

---

## AI Providers

Purpose:

Provide reasoning.

Examples:

- OpenAI
- Anthropic
- Gemini
- Ollama
- Local Models

Interface:

```typescript
interface AIProvider {
    chat()
    summarize()
    explain()
}
```

---

## Analysis Providers

Purpose:

Provide game evaluation.

Examples:

- KataGo
- OGS Review
- Future Analyzers

Interface:

```typescript
interface AnalysisProvider {
    analyzeGame()
}
```

---

# External Service Policy

StoneMind should never assume availability of:

- OpenAI
- Anthropic
- Google
- Any specific provider

All services are optional.

No provider should become architecturally critical.

---

# Export Architecture

All information should be exportable.

---

## Supported Exports

### Full Archive

Contains:

- Games
- Notes
- Journals
- Goals
- Study Plans
- Graphs

---

### Player Profile

Contains:

- Assessments
- Style information
- Strengths
- Weaknesses

---

### Ontology Extensions

Contains:

- Custom concepts
- Custom relationships
- Local ontology additions

---

# Plugin Architecture

Everything external should be a plugin.

---

## Plugin Categories

### AI Providers

Examples:

- OpenAI
- Anthropic
- Gemini
- Ollama

---

### Analysis Engines

Examples:

- KataGo
- OGS

---

### Resource Providers

Examples:

- EPUB Library
- Problem Collections
- Pro Game Archives

---

### Export Providers

Examples:

- Markdown
- HTML
- PDF

---

# V1 Scope

StoneMind V1 should focus on proving the core concept.

---

## Included

- Browser-first operation
- TypeScript implementation
- OPFS storage
- Smaltworks integration
- SGF import
- KataGo integration
- Player Graph
- Goal management
- Weakness identification
- Strength identification
- Recommendations

---

## Deferred

- Cloud synchronization
- Tournament preparation
- Opponent modeling
- Multi-user collaboration
- Voice coaching
- Mobile applications
- Advanced teaching games
- Autonomous agent systems

---

# Architecture Success Criteria

The architecture succeeds when:

- The player owns their data.
- The system can operate offline.
- AI providers are replaceable.
- New integrations can be added without architectural redesign.
- Knowledge remains portable.
- Study history remains portable.
- The Player Graph remains the center of the system.

Most importantly:

The architecture should support StoneMind's mission to help players understand themselves, improve deliberately, and unlock their Go.

