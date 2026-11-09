/**
 * Files fetched from the Hugging Face Hub, checked against what this site recorded for them (Round 86).
 *
 * A pin names a commit and says nothing about what a server returns for it, so a file altered on the Hub, in its CDN or in
 * transit would otherwise be run. `scripts/pin_hub_hashes.py` records the SHA-256 of every file the browser fetches at each
 * pin, the record ships with this site, and a file that does not match, or has no entry, is refused before anything reads it.
 */

/** Said in the run log and on the page, so a refusal is not mistaken for an outage. */
export class HubFileRefused extends Error {
  constructor(message: string) {
    super(message);
    this.name = "HubFileRefused";
  }
}

const hex = (bytes: ArrayBuffer): string => Array.from(new Uint8Array(bytes), (b) => b.toString(16).padStart(2, "0")).join("");

/** The file's digest when it matches `expected`; `HubFileRefused` when it does not, or when nothing is recorded for it. */
export async function checkPinned(bytes: ArrayBuffer | Uint8Array, expected: string | null | undefined, what: string): Promise<string> {
  if (!expected) throw new HubFileRefused(`${what}: no SHA-256 is recorded for this file at this revision, so it was not used`);
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const digest = hex(await crypto.subtle.digest("SHA-256", view as BufferSource));
  if (digest !== expected) {
    throw new HubFileRefused(`${what}: the file served hashes to ${digest.slice(0, 16)}..., not the ${expected.slice(0, 16)}... recorded for this `
      + "revision, so it was not used");
  }
  return digest;
}

export const asJson = <T>(bytes: ArrayBuffer): T => JSON.parse(new TextDecoder().decode(bytes)) as T;
