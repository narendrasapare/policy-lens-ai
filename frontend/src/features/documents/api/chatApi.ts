import { API_BASE_URL } from "../../../api/apiConfig";
import type { ChatRequest, ChatResponse } from "../types/chat.types";

export async function askPolicyQuestion(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body !== null && typeof body === "object" && "detail" in body &&
      typeof body.detail === "string"
        ? body.detail
        : `Chat request failed (HTTP ${response.status}).`;

    if (response.status === 409) {
      throw new Error(`${detail} Upload and index a PDF before chatting.`);
    }
    throw new Error(detail);
  }

  if (
    body === null || typeof body !== "object" ||
    !("answer" in body) || typeof body.answer !== "string" ||
    !("sources" in body) || !Array.isArray(body.sources)
  ) {
    throw new Error("Unexpected response from the chat API.");
  }

  return body as ChatResponse;
}
