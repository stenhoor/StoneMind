/**
 * Provider contracts. Everything external is a provider, and no provider may be mandatory.
 * See docs/StoneMind-Provider-and-Plugin-SDK.md and docs/StoneMind-Technical-Architecture.md.
 *
 * The SDK doc's example interfaces are untyped sketches; the signatures here are a first
 * concrete proposal and are expected to evolve.
 */
import type {
  EvidenceRecord,
  Game,
  GameAnalysis,
  Observation,
  Resource,
  ResourceRelationship,
} from "@stonemind/domain";

export type ProviderKind = "ai" | "analysis" | "evidence" | "resource" | "export";

/** Where a provider runs. Anything other than "in-browser" or "local-process" sends data off the device. */
export type ProviderLocality = "in-browser" | "local-process" | "self-hosted" | "remote";

/**
 * Design Rule 4: provider use must be visible.
 * Users must know what data is sent, where it is sent, and why.
 */
export interface DataDisclosure {
  /** Categories of player data sent, e.g. ["games", "concept-assessments"]. Empty if nothing leaves the device. */
  dataSent: string[];
  /** Destination, e.g. "https://api.example.com" or "localhost:8080". */
  destination: string;
  purpose: string;
}

export interface ProviderDescriptor {
  /** Stable id, e.g. "katago-local", "ollama", "anthropic". */
  id: string;
  kind: ProviderKind;
  displayName: string;
  locality: ProviderLocality;
  disclosure: DataDisclosure;
}

export interface Provider {
  readonly descriptor: ProviderDescriptor;
  /** Providers may disappear at any time (Rule 1). Callers must handle `false`. */
  isAvailable(): Promise<boolean>;
}

// ---------------------------------------------------------------- AI

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface AIProvider extends Provider {
  chat(messages: ChatMessage[], options?: { signal?: AbortSignal }): Promise<string>;
  summarize(text: string, options?: { signal?: AbortSignal }): Promise<string>;
  explain(topic: string, context?: string, options?: { signal?: AbortSignal }): Promise<string>;
  generateTeachingContent?(objective: string, context?: string): Promise<string>;
}

// ---------------------------------------------------------- Analysis

export interface AnalysisRequest {
  game: Game;
  /** Raw SGF text of the game. */
  sgf: string;
  /** Engine-specific options, e.g. visit count. */
  options?: Record<string, unknown>;
}

export interface AnalysisProvider extends Provider {
  analyzeGame(request: AnalysisRequest, options?: { signal?: AbortSignal }): Promise<GameAnalysis>;
}

// ---------------------------------------------------------- Evidence

export interface EvidenceImport {
  observations: Observation[];
  evidence: EvidenceRecord[];
  games?: Game[];
}

export interface EvidenceProvider extends Provider {
  import(options?: { signal?: AbortSignal }): Promise<EvidenceImport>;
}

// ---------------------------------------------------------- Resource

export interface ResourceImport {
  resources: Resource[];
  relationships: ResourceRelationship[];
}

export interface ResourceProvider extends Provider {
  importResource(source: Blob | string, options?: { signal?: AbortSignal }): Promise<ResourceImport>;
}

// ------------------------------------------------------------ Export

export interface ExportProvider extends Provider {
  /** Output format, e.g. "markdown", "html", "json". */
  readonly format: string;
  exportData(data: unknown): Promise<Blob | string>;
}

export interface ProviderByKind {
  ai: AIProvider;
  analysis: AnalysisProvider;
  evidence: EvidenceProvider;
  resource: ResourceProvider;
  export: ExportProvider;
}
