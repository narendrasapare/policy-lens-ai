import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { API_BASE_URL } from "../../api/apiConfig";
import { askPolicyQuestion } from "../documents/api/chatApi";
import type { ChatResponse } from "../documents/types/chat.types";
import { workspaceRequest } from "./workspace-api";
import type { WorkspaceResponse } from "./workspace-api";
import "./policy-workspace.css";

type Turn = { id: number; question: string; response: ChatResponse };
const suggestions = ["What leave benefits are available?", "Summarize the uploaded policies.", "What are the working hours and leave rules?"];

export default function PolicyWorkspace() {
  const [workspace, setWorkspace] = useState<WorkspaceResponse | null>(null);
  const [workspaceError, setWorkspaceError] = useState("");

  const [refreshing, setRefreshing] = useState(false);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [asking, setAsking] = useState(false);
  const [chatError, setChatError] = useState("");
  const request = useRef<AbortController | null>(null);
  const inFlight = useRef(false);
  const sequence = useRef(0);

  useEffect(() => () => request.current?.abort(), []);

  async function refreshDocuments() {
    setRefreshing(true);
    setWorkspaceError("");
    try {
      setWorkspace(await workspaceRequest<WorkspaceResponse>("/api/v1/workspace/documents"));
    } catch (error) {
      setWorkspaceError(error instanceof Error ? error.message : "Could not load documents.");
    } finally { setRefreshing(false); }
  }

  useEffect(() => {
    let active = true;
    let timer: number;
    let controller: AbortController | null = null;
    async function load() {
      controller?.abort();
      const current = new AbortController();
      controller = current;
      try {
        const result = await workspaceRequest<WorkspaceResponse>("/api/v1/workspace/documents", { signal: current.signal });
        if (active) { setWorkspace(result); setWorkspaceError(""); }
      } catch (error) {
        if (active && error instanceof Error && error.name !== "AbortError") setWorkspaceError("Could not load your documents. Check the connection and retry.");
      } finally {
        if (active && controller === current) timer = window.setTimeout(() => void load(), 2500);
      }
    }
    function uploaded() { window.clearTimeout(timer); void load(); }
    void load();
    window.addEventListener("policylens:document-uploaded", uploaded);
    return () => { active = false; window.clearTimeout(timer); controller?.abort(); window.removeEventListener("policylens:document-uploaded", uploaded); };
  }, []);

  async function retryPreparation(documentId: string) {
    try {
      await workspaceRequest(`/api/v1/workspace/documents/${documentId}/retry`, { method: "POST" });
      await refreshDocuments();
    } catch (error) { setWorkspaceError(error instanceof Error ? error.message : "Could not retry preparation."); }
  }

  const preparing = workspace?.documents.some(document => ["queued", "indexing"].includes(document.preparation_status)) ?? false;
  const canAsk = !!workspace?.documents.some(document => document.indexed) && !preparing && !workspaceError;

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = question.trim();
    if (!value || !canAsk || inFlight.current) return;
    inFlight.current = true;
    setAsking(true);
    setChatError("");
    const controller = new AbortController();
    request.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 180_000);
    try {
      const response = await askPolicyQuestion({ question: value, top_k: topK }, controller.signal);
      if (!controller.signal.aborted) {
        const turn = { id: ++sequence.current, question: value, response };
        setTurns(previous => [...previous, turn]);
        setQuestion("");
      }
    } catch (error) {
      setChatError(controller.signal.aborted ? "Request cancelled or timed out. You can retry." : error instanceof Error ? error.message : "Could not answer. Please retry.");
    } finally {
      window.clearTimeout(timeout);
      request.current = null;
      inFlight.current = false;
      setAsking(false);
    }
  }

  function downloadConversation() {
    const blob = new Blob([JSON.stringify(turns, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "policylens-conversation.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <section className="policy-workspace" aria-labelledby="workspace-heading">
    <p className="section-eyebrow">YOUR POLICY WORKSPACE</p>
    <h2 id="workspace-heading">Ask a question about your policies.</h2>
    <p>Your PDFs are prepared automatically. Answers include passages and page references from your documents.</p>
    <div className="workspace-toolbar">
      <button type="button" onClick={() => void refreshDocuments()} disabled={refreshing}>
        {refreshing ? "Loading documents…" : "Refresh documents"}
      </button>
      {workspace && <span role="status">{workspace.documents.length} documents · {workspace.total_vectors} passages ready</span>}
    </div>
    {workspaceError && <p role="alert" className="workspace-error">{workspaceError}</p>}
    {workspace?.documents.length === 0 && <p>No documents yet. Upload a text-based PDF above to get started.</p>}
    <ul className="workspace-documents">
      {workspace?.documents.map(document => <li key={document.document_id}>
        <div><strong>{document.filename}</strong><small>{document.page_count} pages · {document.chunk_count} chunks · {document.indexed ? "Ready for questions" : document.preparation_status === "failed" ? "Preparation failed" : "Preparing…"}</small></div>
        <div className="workspace-toolbar">
          <a href={`${API_BASE_URL}/api/v1/documents/${document.document_id}/file`} target="_blank" rel="noopener noreferrer">Open PDF</a>
          {document.preparation_status === "failed" && <><span role="alert">{document.preparation_error}</span><button type="button" onClick={() => void retryPreparation(document.document_id)}>Try again</button></>}

        </div>
      </li>)}
    </ul>
    {preparing && <p role="status">Preparing your documents in the background. Questions will be available when they’re ready.</p>}
    <hr />
    <div className="workspace-toolbar"><h3>Ask your documents</h3>
      <button type="button" disabled={!turns.length || asking} onClick={() => { setTurns([]); setChatError(""); }}>Clear conversation</button>
      <button type="button" disabled={!turns.length} onClick={downloadConversation}>Export answers</button>
    </div>
    <p>Ask about any ready document. Include the policy or topic in each question.</p>
    <div className="workspace-suggestions">{suggestions.map(value => <button key={value} type="button" disabled={asking} onClick={() => setQuestion(value)}>{value}</button>)}</div>
    <div className="workspace-turns">
      {turns.map(turn => <article className="workspace-answer" key={turn.id}>
        <h3>{turn.question}</h3><p className="workspace-answer-text">{turn.response.answer}</p>
        <p className="workspace-metadata">{turn.response.query_type === "summary" ? "Full document summary" : "Policy answer"} · {turn.response.retrieved_chunks} passages checked</p>
        {turn.response.sub_questions.length > 0 && <details><summary>Question breakdown</summary><ul>{turn.response.sub_questions.map(value => <li key={value}>{value}</li>)}</ul></details>}
        <details><summary>Sources and page references ({turn.response.sources.length})</summary>
          {turn.response.sources.map(source => <article key={`${source.reference}-${source.chunk_id}`} className="workspace-source">
            <a href={`${API_BASE_URL}/api/v1/documents/${encodeURIComponent(source.document_id)}/file#page=${source.page_number}`} target="_blank" rel="noopener noreferrer">[{source.reference}] {source.filename} · page {source.page_number}</a>
            <blockquote>{source.excerpt}</blockquote>

          </article>)}
        </details>
      </article>)}
    </div>
    <form onSubmit={event => void ask(event)} className="workspace-question-form">
      <label htmlFor="policy-question">Your question</label>
      <textarea id="policy-question" value={question} onChange={event => setQuestion(event.target.value)} maxLength={1000} rows={3} required disabled={asking} placeholder="What does the policy say about annual leave?" />
      <div className="workspace-toolbar">
        <label htmlFor="policy-top-k">Sources to retrieve</label>
        <select id="policy-top-k" value={topK} disabled={asking} onChange={event => setTopK(Number(event.target.value))}>{[1,3,5,8,10].map(value => <option key={value} value={value}>{value}</option>)}</select>
        <small>{question.length}/1000 · Summaries read all pages.</small>
        <button type="submit" disabled={asking || !question.trim() || !canAsk}>{asking ? "Finding an answer…" : "Ask PolicyLens"}</button>
        {asking && <button type="button" onClick={() => request.current?.abort()}>Cancel</button>}
      </div>
    </form>
    {asking && <p role="status">Reading your policies and writing an answer. Full summaries may take longer.</p>}
    {chatError && <p className="workspace-error" role="alert">{chatError}</p>}
  </section>;
}
