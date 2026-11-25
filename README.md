# StoneMind

**Unlock your Go.**

StoneMind is an open, adaptive, player-centric Go companion. It learns how a player
understands the game (Player Graph), how they choose to play (Style Graph), and what
they want to achieve (Goals). It then acts as coach, tutor, teacher, trainer, analyst
and practice partner.

It is local-first and browser-first. The player owns their data. It works with
any AI provider, uses resources the player already owns, prefers open formats (SGF, Markdown,
EPUB, JSON, CSV), and integrates with existing tools (KataGo, OGS, Sabaki, KaTrain) rather than replacing them.

Status: **Milestone 0, Project Foundation.** The specifications are drafts and the code is an initial scaffold.

## Documentation

| Document | Topic |
|---|---|
| [Charter](docs/StoneMind-Charter.md) | Vision, mission, principles, invariants |
| [Knowledge Base](docs/StoneMind-Knowledge-Base.md) | Why StoneMind exists; core insights |
| [Technical Architecture](docs/StoneMind-Technical-Architecture.md) | Runtime, storage, engines, V1 scope |
| [Data Model](docs/StoneMind-Data-Model-Specification.md) | Core entities |
| [Ontology Design](docs/StoneMind-Ontology-Design.md) / [Seed](docs/StoneMind-Ontology-Seed.md) | Go concept graph |
| [Style Model](docs/StoneMind-Style-Model.md) | Style dimensions |
| [Heuristics & Inference](docs/StoneMind-Heuristics-and-Inference-Engine.md) | Evidence → understanding |
| [Goal Management](docs/StoneMind-Goal-Management.md) | Player-defined goals |
| [Resource Graph](docs/StoneMind-Resource-Graph.md) | Linking books, games, notes to concepts |
| [Provider & Plugin SDK](docs/StoneMind-Provider-and-Plugin-SDK.md) | Integrations |
| [Research & Roadmap](docs/StoneMind-Research-and-Roadmap.md) | Research tracks, milestones |
| [Monorepo & Interdependencies](docs/MONOREPO.md) | Sub-projects and how they depend on each other |
| [Open Questions](docs/OPEN-QUESTIONS.md) | Inconsistencies and gaps found in the specs |

## Repository layout

This is an npm-workspaces monorepo. Each sub-project is independent, with its own
package.json, version, README, build and tests. All interdependencies are documented in
[docs/MONOREPO.md](docs/MONOREPO.md).

| Sub-project | Purpose |
|---|---|
| [`packages/domain`](packages/domain) | Data model types |
| [`packages/storage`](packages/storage) | Storage ports (Smaltworks/OPFS adapter planned) |
| [`packages/ontology`](packages/ontology) | Ontology seed data, style dimensions, validation |
| [`packages/provider-sdk`](packages/provider-sdk) | Provider/plugin contracts and consent-enforcing registry |

## Getting started

Requires Node.js 20+ (see `.nvmrc`).

```bash
npm install
```

```bash
npm run check    # typecheck + test every sub-project
```

```bash
npm run build    # build every sub-project in dependency order
```

To work on a single sub-project, `cd packages/<name>` and run the same scripts there.

## License

MIT. See [LICENSE](LICENSE).
