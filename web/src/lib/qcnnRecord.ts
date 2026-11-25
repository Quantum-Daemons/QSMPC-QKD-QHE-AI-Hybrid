// The trained QCNN's headline as its weights file records it, for the published-figure table (checklist CS13). The
// file is the one a QCNN run loads (a few kB), so the table can show the circuit's figure before any run.
import { useEffect, useState } from "react";
import { dataDirFor } from "./clientCompute";
import type { QcnnRecord } from "./publishedComparisons";
import { hasTrainedQcnn } from "./usePipelineRun";

/** The weights file's field for each headline metric it can name. */
const FIELD: Record<string, string> = { minority_class_f1: "test_minority_f1", accuracy: "test_accuracy", macro_f1: "test_macro_f1" };

export function useQcnnRecord(usecase: string): QcnnRecord | null {
  const [record, setRecord] = useState<QcnnRecord | null>(null);
  useEffect(() => {
    setRecord(null);
    if (!hasTrainedQcnn(usecase)) return;
    let live = true;
    fetch(`${import.meta.env.BASE_URL}data/${dataDirFor(usecase)}/qcnn_weights.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((w: Record<string, unknown> | null) => {
        const metric = typeof w?.headline_metric === "string" ? w.headline_metric : null;
        const value = metric && FIELD[metric] ? w?.[FIELD[metric]] : null;
        if (live && metric && typeof value === "number") setRecord({ metric, value });
      })
      .catch(() => undefined);
    return () => { live = false; };
  }, [usecase]);
  return record;
}
