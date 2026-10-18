import { API_BASE_URL } from "./apiConfig";

export interface HealthResponse {
  status: string;
  service: string;
}

export async function getHealthStatus(
  signal?: AbortSignal
): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    signal,
  });

  if (!response.ok) {
    throw new Error(
      `Backend health check failed: HTTP ${response.status}`
    );
  }

  const data: unknown = await response.json();

  if (
    typeof data !== "object" ||
    data === null ||
    !("status" in data) ||
    typeof data.status !== "string" ||
    !("service" in data) ||
    typeof data.service !== "string"
  ) {
    throw new Error("Invalid backend health response");
  }

  return {
    status: data.status,
    service: data.service,
  };
}