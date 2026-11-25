import { expect, it } from "vitest";
import { DEPENDENCY_RELATIONSHIP_TYPES } from "../src/index.js";

it("treats only DEPENDS_ON and PREREQUISITE_FOR as learning dependencies", () => {
  expect([...DEPENDENCY_RELATIONSHIP_TYPES].sort()).toEqual(["DEPENDS_ON", "PREREQUISITE_FOR"]);
});
