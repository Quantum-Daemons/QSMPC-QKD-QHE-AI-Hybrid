/**
 * The model each page's private-inference path encrypts, declared once (Round 86, batch 12; the owner's choice of
 * 2026-11-28 that every text naming it be held to one place, after finance's cards went on naming the linear student
 * for three days after F7b's square network had replaced it).
 *
 * A page whose encrypted model is its registry student reads that student. Finance's is F7b's square network, a site
 * file rather than a registry model, which replaced `finance_aml.student` on both of the page's encrypted paths on
 * 2026-11-25. Telecom's primary encrypted model is the QKD-layer student, which has no square network; its 5G-NIDD
 * figure has run the 5G-NIDD square network since batch 12, in place of batch 6's linear student, and is listed second.
 * `tests/test_r86_page_encrypted_models.py` holds every entry to the files, the registry and the run's metrics, and the
 * texts that name a replaced model to the fact that it was replaced.
 */
import { studentModel, studentParamCount } from "./config";

export type EncryptedKind = "square network" | "linear" | "convolutional";

export interface PageEncryptedModel {
  key: string;
  /** For a card or a row: what the model is, in a phrase. */
  label: string;
  params: number;
  kind: EncryptedKind;
  /** The figure or step it is encrypted in. */
  where: string;
  /** The model it replaced on this path, if it replaced one. */
  replaced?: string;
}

/** Site-file networks, which the registry does not hold; their parameter counts are held to the files by the test. */
const SITE_FILE_MODELS: Record<string, PageEncryptedModel[]> = {
  finance_aml: [{
    key: "finance_aml.square_network", label: "F7b's square network", params: 14978, kind: "square network",
    where: "the private-inference figure and the one-row CKKS step", replaced: "finance_aml.student",
  }],
  telecom_ids: [{
    key: "telecom_ids.square_network", label: "the 5G-NIDD square network", params: 2569, kind: "square network",
    where: "the 5G-NIDD figure", replaced: "telecom_ids.nidd_student",
  }],
};

/** Where each page encrypts its registry student (the first `t1_crypto` model), and what kind of model it is. */
const REGISTRY_STUDENT: Record<string, { kind: EncryptedKind; where: string }> = {
  medical_fl: { kind: "convolutional", where: "the one-row CKKS step and the MPC panel" },
  telecom_ids: { kind: "linear", where: "the IDS stage and the private-inference figure" },
  satellite_comms: { kind: "linear", where: "the forecast check, every forecast under CKKS" },
};

/** Every model the page encrypts, its primary first. */
export function pageEncryptedModels(usecase: string): PageEncryptedModel[] {
  const site = SITE_FILE_MODELS[usecase] ?? [];
  if (usecase === "finance_aml") return site;
  const s = studentModel(usecase);
  const n = studentParamCount(s);
  const how = REGISTRY_STUDENT[usecase];
  const registry: PageEncryptedModel[] = s && n != null && how ? [{ key: s.key, label: s.label, params: n, ...how }] : [];
  return [...registry, ...site];
}

/** The page's primary encrypted model, or null. */
export const primaryEncryptedModel = (usecase: string): PageEncryptedModel | null => pageEncryptedModels(usecase)[0] ?? null;
