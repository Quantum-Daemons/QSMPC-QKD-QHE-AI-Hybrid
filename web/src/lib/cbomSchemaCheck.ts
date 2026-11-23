// The CBOM checked against the CycloneDX 1.7 schema in the tab (Round 86, the owner's choice of 2026-11-23).
//
// The page's Content-Security-Policy has no 'unsafe-eval', so Ajv cannot compile a schema here; the schema was compiled
// ahead of time into ./generated/cdx17Validate.js (web/scripts/build_cdx_validator.mjs, Ajv 8.20.0, from the vendored
// schemas CI validates against). It is imported only when asked for, so its chunk is not part of any page's first load.

export const SCHEMA_CHECK = "the vendored CycloneDX 1.7 schema, compiled ahead of time by Ajv 8.20.0";

export interface SchemaCheck {
  valid: boolean;
  /** The first error, where the document is not valid: the path in the document, and what the schema wanted. */
  error?: string;
}

export async function checkCbom(doc: unknown): Promise<SchemaCheck> {
  const { validate } = await import("./generated/cdx17Validate.js");
  if (validate(doc)) return { valid: true };
  const e = validate.errors?.[0];
  return { valid: false, error: e ? `${e.instancePath || "the document"}: ${e.message ?? e.keyword}` : "not valid" };
}
