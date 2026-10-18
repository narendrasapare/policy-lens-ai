import { API_BASE_URL } from "../../../api/apiConfig";

import type {
  DocumentUploadResponse,
} from "../types/document.types";

function isDocumentUploadResponse(
  value: unknown
): value is DocumentUploadResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const response = value as Partial<DocumentUploadResponse>;
  const document = response.document;

  return (
    (response.status === "processed" ||
      response.status === "already_exists") &&
    document !== undefined &&
    typeof document.document_id === "string" &&
    typeof document.filename === "string" &&
    typeof document.page_count === "number" &&
    typeof document.character_count === "number" &&
    typeof document.chunk_count === "number"
  );
}

export async function uploadDocument(
  file: File,
  signal?: AbortSignal
): Promise<DocumentUploadResponse> {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/v1/documents`,
    {
      method: "POST",
      body: formData,
      signal,
    }
  );

  const responseBody: unknown = await response
    .json()
    .catch(() => null);

  if (!response.ok) {
    let errorMessage = `Document upload failed: HTTP ${response.status}`;

    if (
      typeof responseBody === "object" &&
      responseBody !== null &&
      "detail" in responseBody &&
      typeof responseBody.detail === "string"
    ) {
      errorMessage = responseBody.detail;
    }

    throw new Error(errorMessage);
  }

  if (!isDocumentUploadResponse(responseBody)) {
    throw new Error("Invalid document upload response");
  }

  return responseBody;
}