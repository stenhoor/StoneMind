import type {
  Concept,
  ConceptCategory,
  ConceptRelationship,
  RelationshipType,
} from "@stonemind/domain";
import seedJson from "./seed.json" with { type: "json" };

/** Shape of src/seed.json: the seed document as written, before name resolution. */
export interface SeedConcept {
  id: string;
  name: string;
  category: ConceptCategory;
  description?: string;
  concerns?: string[];
  commonMisunderstandings?: string[];
  /** Relation type -> target concept NAMES as written in the seed doc. */
  relations?: Partial<Record<RelationshipType, string[]>>;
}

export interface SeedFile {
  version: string;
  source: string;
  aliases: Record<string, string[]>;
  rootDependencies: string[];
  concepts: SeedConcept[];
}

export interface UnresolvedReference {
  sourceConcept: string;
  relationshipType: RelationshipType;
  target: string;
}

export interface ResolvedOntology {
  version: string;
  concepts: Concept[];
  relationships: ConceptRelationship[];
  /** Relation targets that name no seed concept or alias. Candidates for new concepts or aliases. */
  unresolved: UnresolvedReference[];
}

export const SEED: SeedFile = seedJson as SeedFile;

/** Resolve seed relation targets (names) to concept ids, collecting anything unresolved. */
export function resolveSeed(seed: SeedFile = SEED): ResolvedOntology {
  const byName = new Map<string, string[]>();
  for (const c of seed.concepts) byName.set(c.name.toLowerCase(), [c.id]);
  for (const [alias, ids] of Object.entries(seed.aliases)) byName.set(alias.toLowerCase(), ids);

  const concepts: Concept[] = seed.concepts.map(({ relations: _relations, ...concept }) => concept);
  const relationships: ConceptRelationship[] = [];
  const unresolved: UnresolvedReference[] = [];

  for (const source of seed.concepts) {
    for (const [type, targets] of Object.entries(source.relations ?? {}) as [
      RelationshipType,
      string[],
    ][]) {
      for (const target of targets) {
        const ids = byName.get(target.toLowerCase());
        if (!ids) {
          unresolved.push({ sourceConcept: source.id, relationshipType: type, target });
          continue;
        }
        for (const targetId of ids) {
          relationships.push({
            id: `${source.id}--${type}--${targetId}`,
            sourceConcept: source.id,
            targetConcept: targetId,
            relationshipType: type,
          });
        }
      }
    }
  }

  return { version: seed.version, concepts, relationships, unresolved };
}
