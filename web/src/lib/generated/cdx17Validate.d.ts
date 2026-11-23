// Types for cdx17Validate.js, which web/scripts/build_cdx_validator.mjs generates with Ajv's standalone code.
export interface CdxSchemaError {
  instancePath: string;
  schemaPath: string;
  keyword: string;
  params: Record<string, unknown>;
  message?: string;
}

export declare const validate: ((data: unknown) => boolean) & { errors?: CdxSchemaError[] | null };
export default validate;
