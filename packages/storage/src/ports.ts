/**
 * Storage ports.
 *
 * The architecture places all persistence on the Smaltworks platform (SQL + Cypher graph +
 * DataFrames, persisted to OPFS). Smaltworks is not yet available as a dependency of this
 * repository, so core code depends only on these narrow ports. A Smaltworks adapter will
 * implement them; tests can use in-memory adapters.
 *
 * Keep these minimal until the Smaltworks API is known: they define what StoneMind needs,
 * not what Smaltworks offers.
 */

/** Relational data: players, games, goals, study plans, reviews, journals, resources. */
export interface RelationalStore {
  query<Row = Record<string, unknown>>(sql: string, params?: readonly unknown[]): Promise<Row[]>;
  execute(sql: string, params?: readonly unknown[]): Promise<void>;
}

/** Graph data: Go ontology, player graph, style graph, resource graph. */
export interface GraphStore {
  cypher<Row = Record<string, unknown>>(
    query: string,
    params?: Record<string, unknown>,
  ): Promise<Row[]>;
}

/** Analytical data: performance trends, tsumego success rates, style evolution. */
export interface AnalyticsStore {
  /** Placeholder until the Smaltworks DataFrame API is specified. */
  readonly kind: "analytics";
}

export interface StoneMindStorage {
  relational: RelationalStore;
  graph: GraphStore;
  analytics: AnalyticsStore;
}
