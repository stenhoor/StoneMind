import type { Provider, ProviderByKind, ProviderDescriptor, ProviderKind } from "./types.js";

/** True when the provider sends data off the player's device. */
export function isExternal(descriptor: ProviderDescriptor): boolean {
  return descriptor.locality === "remote" || descriptor.locality === "self-hosted";
}

export class ConsentRequiredError extends Error {
  constructor(readonly descriptor: ProviderDescriptor) {
    super(
      `Provider "${descriptor.id}" sends data to ${descriptor.disclosure.destination} ` +
        `and has not been approved by the player.`,
    );
    this.name = "ConsentRequiredError";
  }
}

/**
 * Holds the providers the player has configured.
 *
 * Invariants enforced here:
 * - Players control provider selection (Rule 3): nothing is registered implicitly.
 * - No external transmission without explicit consent (Charter, "User Consent"):
 *   `get` refuses an external provider until `grantConsent` is called for it.
 * - Providers are optional (Rule 2): lookups return `undefined` rather than throwing when absent.
 */
export class ProviderRegistry {
  readonly #providers = new Map<string, Provider>();
  readonly #consented = new Set<string>();

  register(provider: Provider): void {
    const { id } = provider.descriptor;
    if (this.#providers.has(id)) {
      throw new Error(`Provider "${id}" is already registered.`);
    }
    this.#providers.set(id, provider);
  }

  unregister(id: string): void {
    this.#providers.delete(id);
    this.#consented.delete(id);
  }

  /** Record the player's approval for an external provider after showing them its disclosure. */
  grantConsent(id: string): void {
    if (!this.#providers.has(id)) throw new Error(`Unknown provider "${id}".`);
    this.#consented.add(id);
  }

  revokeConsent(id: string): void {
    this.#consented.delete(id);
  }

  hasConsent(id: string): boolean {
    return this.#consented.has(id);
  }

  /** Descriptors of every registered provider, for display to the player. */
  list(kind?: ProviderKind): ProviderDescriptor[] {
    return [...this.#providers.values()]
      .map((p) => p.descriptor)
      .filter((d) => kind === undefined || d.kind === kind);
  }

  /**
   * Returns the provider if registered and usable. Throws ConsentRequiredError for an
   * external provider the player has not approved.
   */
  get<K extends ProviderKind>(kind: K, id: string): ProviderByKind[K] | undefined {
    const provider = this.#providers.get(id);
    if (!provider || provider.descriptor.kind !== kind) return undefined;
    if (isExternal(provider.descriptor) && !this.#consented.has(id)) {
      throw new ConsentRequiredError(provider.descriptor);
    }
    return provider as ProviderByKind[K];
  }
}
