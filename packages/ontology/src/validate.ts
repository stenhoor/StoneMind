import type { ConceptRelationship } from "@stonemind/domain";
import type { ResolvedOntology } from "./seed.js";

export interface ValidationIssue {
  kind: "duplicate-id" | "duplicate-name" | "dangling-reference" | "self-reference" | "dependency-cycle";
  message: string;
}

/**
 * Learning-dependency edges normalised to "prerequisite -> dependent".
 * `A DEPENDS_ON B` and `B PREREQUISITE_FOR A` both become B -> A.
 */
export function dependencyEdges(relationships: readonly ConceptRelationship[]): [string, string][] {
  const edges: [string, string][] = [];
  for (const r of relationships) {
    if (r.relationshipType === "DEPENDS_ON") edges.push([r.targetConcept, r.sourceConcept]);
    else if (r.relationshipType === "PREREQUISITE_FOR") edges.push([r.sourceConcept, r.targetConcept]);
  }
  return edges;
}

/** Find every elementary cycle reachable in the dependency graph (each reported once, rotated to its smallest id). */
export function findDependencyCycles(relationships: readonly ConceptRelationship[]): string[][] {
  const adj = new Map<string, Set<string>>();
  for (const [from, to] of dependencyEdges(relationships)) {
    if (!adj.has(from)) adj.set(from, new Set());
    adj.get(from)!.add(to);
  }

  const seen = new Set<string>();
  const cycles: string[][] = [];
  const path: string[] = [];
  const onPath = new Set<string>();

  const visit = (node: string): void => {
    path.push(node);
    onPath.add(node);
    for (const next of adj.get(node) ?? []) {
      if (onPath.has(next)) {
        const cycle = path.slice(path.indexOf(next));
        const start = cycle.indexOf([...cycle].sort()[0]!);
        const canonical = [...cycle.slice(start), ...cycle.slice(0, start)];
        const key = canonical.join(">");
        if (!seen.has(key)) {
          seen.add(key);
          cycles.push(canonical);
        }
      } else {
        visit(next);
      }
    }
    path.pop();
    onPath.delete(node);
  };

  for (const node of adj.keys()) visit(node);
  return cycles;
}

/** Structural checks on a resolved ontology. Unresolved seed names are reported separately by resolveSeed(). */
export function validateOntology(ontology: ResolvedOntology): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const ids = new Set<string>();
  const names = new Set<string>();

  for (const c of ontology.concepts) {
    if (ids.has(c.id)) issues.push({ kind: "duplicate-id", message: `Duplicate concept id "${c.id}".` });
    ids.add(c.id);
    const name = c.name.toLowerCase();
    if (names.has(name)) issues.push({ kind: "duplicate-name", message: `Duplicate concept name "${c.name}".` });
    names.add(name);
  }

  for (const r of ontology.relationships) {
    for (const end of [r.sourceConcept, r.targetConcept]) {
      if (!ids.has(end)) {
        issues.push({ kind: "dangling-reference", message: `Relationship "${r.id}" references unknown concept "${end}".` });
      }
    }
    if (r.sourceConcept === r.targetConcept) {
      issues.push({ kind: "self-reference", message: `Relationship "${r.id}" points at itself.` });
    }
  }

  for (const cycle of findDependencyCycles(ontology.relationships)) {
    issues.push({ kind: "dependency-cycle", message: `Dependency cycle: ${[...cycle, cycle[0]].join(" -> ")}.` });
  }

  return issues;
}
