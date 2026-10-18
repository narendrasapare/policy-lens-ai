import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { uploadDocument } from "../api/documentsApi";

import type {
  DocumentIngestionOutcome,
  DocumentSummary,
  DocumentUploadStatus,
} from "../types/document.types";

export function useDocumentUpload() {
  const [status, setStatus] =
    useState<DocumentUploadStatus>("idle");

  const [uploadedDocument, setUploadedDocument] =
    useState<DocumentSummary | null>(null);

  const [uploadOutcome, setUploadOutcome] =
    useState<DocumentIngestionOutcome | null>(null);

  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const reset = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    setStatus("idle");
    setUploadedDocument(null);
    setUploadOutcome(null);
    setError(null);
  }, []);

  const upload = useCallback(async (file: File) => {
    abortControllerRef.current?.abort();

    const controller = new AbortController();

    abortControllerRef.current = controller;

    setStatus("uploading");
    setUploadedDocument(null);
    setUploadOutcome(null);
    setError(null);

    try {
      const response = await uploadDocument(
        file,
        controller.signal
      );

      if (controller.signal.aborted) {
        return;
      }

      setUploadedDocument(response.document);
      setUploadOutcome(response.status);
      setStatus("success");
      window.dispatchEvent(new Event("policylens:document-uploaded"));
    } catch (uploadError: unknown) {
      if (controller.signal.aborted) {
        return;
      }

      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Unable to upload document"
      );

      setStatus("error");
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, []);

  return {
    status,
    uploadedDocument,
    uploadOutcome,
    error,
    upload,
    reset,
  };
}