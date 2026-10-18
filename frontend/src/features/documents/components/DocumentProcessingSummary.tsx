import { useEffect, useState } from "react";
import { workspaceRequest } from "../../workspace/workspace-api";
import type { WorkspaceDocument, WorkspaceResponse } from "../../workspace/workspace-api";
import {
  AlertCircle,
  CheckCircle2,
  FileSearch,
  Info,
  Layers3,
  LoaderCircle,
} from "lucide-react";

import type {
  DocumentIngestionOutcome,
  DocumentSummary,
  DocumentUploadStatus,
} from "../types/document.types";

interface DocumentProcessingSummaryProps {
  status: DocumentUploadStatus;
  document: DocumentSummary | null;
  uploadOutcome: DocumentIngestionOutcome | null;
  error: string | null;
  hasSelectedFile: boolean;
}

const processingSteps = [
  {
    number: "01",
    title: "Validate",
    detail: "Check format and file size",
  },
  {
    number: "02",
    title: "Extract",
    detail: "Read PDF text page by page",
  },
  {
    number: "03",
    title: "Prepare",
    detail: "Create chunks with page references",
  },
];

function DocumentProcessingSummary({
  status,
  document,
  uploadOutcome,
  error,
  hasSelectedFile,
}: DocumentProcessingSummaryProps) {
  const [preparation, setPreparation] = useState<WorkspaceDocument | null>(null);
  const documentId = document?.document_id;
  useEffect(() => {
    if (!documentId) return;
    let active = true;
    let timer: number;
    const controller = new AbortController();
    async function check() {
      let ready = false;
      try {
        const workspace = await workspaceRequest<WorkspaceResponse>("/api/v1/workspace/documents", { signal: controller.signal });
        const current = workspace.documents.find(item => item.document_id === documentId) ?? null;
        if (active) setPreparation(current);
        ready = current?.indexed ?? false;
      } catch {
        if (active) setPreparation(null);
      } finally {
        if (active && !ready) timer = window.setTimeout(() => void check(), 2500);
      }
    }
    void check();
    return () => { active = false; window.clearTimeout(timer); controller.abort(); };
  }, [documentId]);
  const current = preparation?.document_id === documentId ? preparation : null;
  if (status === "uploading") {
    return (
      <aside className="processing-panel" aria-live="polite">
        <span className="processing-panel__symbol processing-panel__symbol--loading">
          <LoaderCircle
            size={25}
            className="processing-panel__spinner"
            aria-hidden="true"
          />
        </span>

        <p className="processing-panel__eyebrow">CHECKING DOCUMENT</p>

        <h3>Preparing your knowledge source</h3>

        <p className="processing-panel__description">
          Checking for an existing document and preparing new content when
          necessary.
        </p>

        <div className="processing-panel__activity" aria-hidden="true">
          <span />
        </div>

        <p className="processing-panel__footnote">
          Please keep this page open while processing finishes.
        </p>
      </aside>
    );
  }

  if (status === "error") {
    return (
      <aside className="processing-panel" role="alert">
        <span className="processing-panel__symbol processing-panel__symbol--error">
          <AlertCircle size={25} aria-hidden="true" />
        </span>

        <p className="processing-panel__eyebrow">UPLOAD FAILED</p>

        <h3>We couldn't process this PDF</h3>

        <p className="processing-panel__description">
          {error ?? "Please check your file and try again."}
        </p>

        <p className="processing-panel__footnote">
          Choose another file or retry the upload.
        </p>
      </aside>
    );
  }

  if (status === "success" && document) {
    const isDuplicate = uploadOutcome === "already_exists";

    const metrics = [
      {
        label: "Pages",
        value: document.page_count,
      },
      {
        label: "Chunks",
        value: document.chunk_count,
      },
      {
        label: "Characters",
        value: document.character_count,
      },
    ];

    return (
      <aside
        className="processing-panel"
        aria-label="Document processing result"
      >
        <span className="processing-panel__symbol processing-panel__symbol--success">
          <CheckCircle2 size={25} aria-hidden="true" />
        </span>

        <p className="processing-panel__eyebrow">
          {isDuplicate ? "DOCUMENT ALREADY EXISTS" : "UPLOAD COMPLETE"}
        </p>

        <h3 role="status">
          {isDuplicate
            ? "Already in your workspace"
            : "PDF uploaded"}
        </h3>

        <p className="processing-panel__description">
          {isDuplicate
            ? "An identical PDF already exists. We've reused the original document without adding duplicate chunks."
            : "Text extraction and chunking finished successfully."}
        </p>

        <div className="processing-panel__metrics">
          {metrics.map(({ label, value }) => (
            <div className="processing-panel__metric" key={label}>
              <span>{label}</span>
              <strong>{value.toLocaleString()}</strong>
            </div>
          ))}
        </div>

        <p className="processing-panel__footnote">
          Stored document: {document.filename}
        </p>

        <div className="processing-panel__note">
          <Info size={17} aria-hidden="true" />
          <span>
            {current?.indexed
              ? "Ready for questions. You can ask about this PDF below."
              : current?.preparation_status === "failed"
                ? "Preparation failed. Use Try again in the workspace below."
                : current
                  ? "Preparing your PDF in the background. Its status updates automatically below."
                  : "Upload saved. Checking whether your PDF is ready…"}
          </span>
        </div>
      </aside>
    );
  }

  return (
    <aside
      className="processing-panel"
      aria-label="Document processing workflow"
    >
      <span className="processing-panel__symbol">
        <FileSearch size={25} aria-hidden="true" />
      </span>

      <p className="processing-panel__eyebrow">WHAT HAPPENS NEXT</p>

      <h3>
        {hasSelectedFile ? "Ready to process" : "From PDF to searchable text"}
      </h3>

      <p className="processing-panel__description">
        Your document will go through three preparation steps.
      </p>

      <ol className="processing-panel__steps">
        {processingSteps.map((step) => (
          <li key={step.number}>
            <span className="processing-panel__step-number">{step.number}</span>

            <span>
              <strong>{step.title}</strong>
              <small>{step.detail}</small>
            </span>
          </li>
        ))}
      </ol>

      <div className="processing-panel__note">
        <Layers3 size={17} aria-hidden="true" />
        <span>Your PDF is prepared automatically after upload.</span>
      </div>
    </aside>
  );
}

export default DocumentProcessingSummary;
