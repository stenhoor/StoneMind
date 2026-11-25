import { describe, expect, it } from "vitest";
import {
  ConsentRequiredError,
  ProviderRegistry,
  type AIProvider,
  type ProviderDescriptor,
} from "../src/index.js";

function fakeAI(id: string, locality: ProviderDescriptor["locality"]): AIProvider {
  return {
    descriptor: {
      id,
      kind: "ai",
      displayName: id,
      locality,
      disclosure: { dataSent: ["game-summaries"], destination: `${id}.example`, purpose: "explanations" },
    },
    isAvailable: async () => true,
    chat: async () => "",
    summarize: async () => "",
    explain: async () => "",
  };
}

describe("ProviderRegistry", () => {
  it("returns local providers without consent", () => {
    const r = new ProviderRegistry();
    r.register(fakeAI("local-llm", "local-process"));
    expect(r.get("ai", "local-llm")).toBeDefined();
  });

  it("refuses remote providers until the player consents", () => {
    const r = new ProviderRegistry();
    r.register(fakeAI("cloud-llm", "remote"));
    expect(() => r.get("ai", "cloud-llm")).toThrow(ConsentRequiredError);
    r.grantConsent("cloud-llm");
    expect(r.get("ai", "cloud-llm")).toBeDefined();
    r.revokeConsent("cloud-llm");
    expect(() => r.get("ai", "cloud-llm")).toThrow(ConsentRequiredError);
  });

  it("returns undefined for absent providers or the wrong kind (providers are optional)", () => {
    const r = new ProviderRegistry();
    r.register(fakeAI("local-llm", "in-browser"));
    expect(r.get("ai", "missing")).toBeUndefined();
    expect(r.get("analysis", "local-llm")).toBeUndefined();
  });

  it("rejects duplicate registration and lists descriptors by kind", () => {
    const r = new ProviderRegistry();
    r.register(fakeAI("a", "in-browser"));
    expect(() => r.register(fakeAI("a", "in-browser"))).toThrow();
    expect(r.list("ai").map((d) => d.id)).toEqual(["a"]);
    expect(r.list("export")).toEqual([]);
  });

  it("clears consent on unregister", () => {
    const r = new ProviderRegistry();
    r.register(fakeAI("cloud", "remote"));
    r.grantConsent("cloud");
    r.unregister("cloud");
    r.register(fakeAI("cloud", "remote"));
    expect(r.hasConsent("cloud")).toBe(false);
  });
});
