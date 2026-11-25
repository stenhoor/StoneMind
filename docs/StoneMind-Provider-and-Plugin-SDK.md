# StoneMind Provider and Plugin SDK

Version: 0.1

---

# Purpose

StoneMind integrates with external services.

StoneMind should never become dependent upon them.

Everything external is treated as a provider.

---

# Core Philosophy

Bring Your Own AI.

Bring Your Own Knowledge.

Bring Your Own Tools.

---

# Provider Types

## AI Providers

Purpose:

Reasoning and explanation.

Examples:

- OpenAI
- Anthropic
- Gemini
- Ollama
- Future Models

---

## Analysis Providers

Purpose:

Game analysis.

Examples:

- KataGo
- OGS AI Reviews
- Future engines

---

## Resource Providers

Purpose:

Knowledge ingestion.

Examples:

- EPUB Libraries
- SGF Collections
- Tsumego Collections

---

## Export Providers

Purpose:

Data extraction.

Examples:

- Markdown
- HTML
- JSON
- PDF

---

# Design Rules

## Rule 1

Any provider may disappear.

StoneMind must continue operating.

---

## Rule 2

No provider may become mandatory.

---

## Rule 3

Players control provider selection.

---

## Rule 4

Provider use must be visible.

Users should know:

- what data is sent
- where it is sent
- why it is sent

---

# Example Interfaces

```typescript
interface AIProvider {
    summarize();
    explain();
    generateTeachingContent();
}
```

```typescript
interface AnalysisProvider {
    analyzeGame();
}
```

```typescript
interface ResourceProvider {
    importResource();
}
```

```typescript
interface ExportProvider {
    exportData();
}
```

---

# Future Plugin Categories

- Visualization Plugins
- Coaching Plugins
- Ontology Plugins
- Teaching Plugins
- Tournament Plugins
- Community Extensions
