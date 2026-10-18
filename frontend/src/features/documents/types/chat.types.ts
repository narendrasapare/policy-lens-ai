export interface ChatRequest { question: string; top_k?: number }
export interface ChatSource {
  reference: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  page_number: number;
  excerpt: string;
  similarity_score: number;
  rerank_score: number | null;
  retrieved_for: string | null;
}
export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  retrieved_chunks: number;
  query_type: "factual" | "summary" | "multi_part";
  sub_questions: string[];
  reranking_applied: boolean;
}
