import type { Id } from "./common.js";

/** See docs/StoneMind-Resource-Graph.md. */
export type ResourceType =
  | "book"
  | "article"
  | "video"
  | "professional-game"
  | "course"
  | "problem-collection"
  | "study-note"
  | "journal-entry";

export interface Resource {
  id: Id;
  title: string;
  type: ResourceType;
  author?: string;
  /** Where the player's copy lives. Players bring resources they already own. */
  location: string;
}

export type ResourceRelationshipType = "TEACHES" | "EXPLAINS" | "DEMONSTRATES" | "ILLUSTRATES";

export interface ResourceRelationship {
  resourceId: Id;
  conceptId: Id;
  relationshipType: ResourceRelationshipType;
  /** Optional pointer into the resource (chapter, page, move number, ...). */
  locator?: string;
}
