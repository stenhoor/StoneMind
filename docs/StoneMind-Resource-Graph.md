i# StoneMind Resource Graph Specification

Version: 0.1

---

# Purpose

The Resource Graph connects learning resources to the Go Ontology.

The objective is to transform isolated content into connected knowledge.

Resources include:

- Books
- Games
- Articles
- Tsumego
- Videos
- Notes
- Journals

---

# Philosophy

Players already possess knowledge.

StoneMind should help organize and connect it.

Not replace it.

---

# Resource Types

## Books

Formats:

- EPUB
- PDF

Examples:

- Attack and Defense
- Lessons in the Fundamentals of Go

---

## Games

Formats:

- SGF

Examples:

- Professional games
- Teaching games
- Personal games

---

## Problems

Formats:

- SGF
- Custom collections

Examples:

- Life and Death
- Tesuji
- Invasion problems

---

## Notes

Formats:

- Markdown

Examples:

- Personal journals
- Study notes
- Teacher reviews

---

# Resource Relationships

Examples:

```text
Book
    EXPLAINS
Concept

Game
    DEMONSTRATES
Concept

Problem
    PRACTICES
Concept

Journal
    REFLECTS_ON
Concept
```

---

# Example

```text
Concept:
Thickness

Explained By:
Attack and Defense

Demonstrated In:
Lee Sedol Game #17

Practiced Through:
Problem Collection #4

Discussed In:
October Journal
```

---

# Recommendation Integration

Study recommendations should point to resources.

Not concepts alone.

Bad:

Study Thickness.

Good:

Review:
- Chapter 4 Attack and Defense
- Problem Set 17
- Lee Sedol Game 23
