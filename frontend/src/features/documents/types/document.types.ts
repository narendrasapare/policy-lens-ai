export interface DocumentSummary {
  document_id: string;
  filename: string;
  page_count: number;
  character_count: number;
  chunk_count: number;
}

export type DocumentIngestionOutcome =
  | "processed"
  | "already_exists";

export interface DocumentUploadResponse {
  status: DocumentIngestionOutcome;
  document: DocumentSummary;
}

export type DocumentUploadStatus =
  | "idle"
  | "uploading"
  | "success"
  | "error";