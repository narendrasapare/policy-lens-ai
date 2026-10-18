import { API_BASE_URL } from "../../api/apiConfig";
import type { DocumentSummary } from "../documents/types/document.types";
export interface WorkspaceDocument extends DocumentSummary { indexed: boolean; preparation_status: "queued" | "indexing" | "ready" | "failed"; preparation_error: string | null }
export interface WorkspaceResponse { documents: WorkspaceDocument[]; total_vectors: number }
export async function workspaceRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof body?.detail === "string" ? body.detail : `Request failed (${response.status}).`);
  }
  if (body === null) throw new Error("The server returned an empty response.");
  return body as T;
}
