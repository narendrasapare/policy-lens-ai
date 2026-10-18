import { useState } from "react";
import type { FormEvent } from "react";
import {
  AlertCircle,
  ArrowRight,
  Check,
  FileText,
  LoaderCircle,
  X,
} from "lucide-react";

import { useDocumentUpload } from "../hooks/useDocumentUpload";
import DocumentProcessingSummary from "./DocumentProcessingSummary";
import PdfDropzone from "./PdfDropzone";

import "./DocumentUploader.css";

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

function validateFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return "Only PDF files are supported.";
  }

  if (file.size === 0) {
    return "This file is empty. Choose a PDF with text content.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return "This PDF exceeds the 10 MB upload limit.";
  }

  return null;
}

function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  return `${Math.max(1, Math.ceil(bytes / 1024))} KB`;
}

function DocumentUploader() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [validationError, setValidationError] = useState<string | null>(null);

  const { status, uploadedDocument, uploadOutcome, error, upload, reset } =
    useDocumentUpload();

  const isProcessing = status === "uploading";
  const isComplete = status === "success";

  const handleFileSelect = (file: File) => {
    reset();

    const message = validateFile(file);

    setValidationError(message);
    setSelectedFile(message ? null : file);
  };

  const handleSelectionError = (message: string) => {
    reset();
    setSelectedFile(null);
    setValidationError(message);
  };

  const handleRemoveFile = () => {
    if (isProcessing) return;

    reset();
    setSelectedFile(null);
    setValidationError(null);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile || isProcessing || isComplete) {
      return;
    }

    void upload(selectedFile);
  };

  const buttonLabel = isProcessing
    ? "Checking document..."
    : isComplete
      ? uploadOutcome === "already_exists"
        ? "Already uploaded"
        : "Uploaded"
      : "Upload and process";

  return (
    <section
      className="document-uploader"
      aria-labelledby="document-upload-heading"
    >
      <div className="document-uploader__header">
        <div>
          <p className="document-uploader__eyebrow">DOCUMENT WORKSPACE</p>

          <h2 id="document-upload-heading">Upload a policy</h2>

          <p className="document-uploader__intro">
            Add an HR policy or handbook. We'll extract the text and prepare it
            for questions.
          </p>
        </div>

        <span className="document-uploader__limit">PDF only · 10 MB max</span>
      </div>

      <div className="document-uploader__layout">
        <form className="document-uploader__form" onSubmit={handleSubmit}>
          <PdfDropzone
            disabled={isProcessing}
            hasSelectedFile={selectedFile !== null}
            onFileSelect={handleFileSelect}
            onSelectionError={handleSelectionError}
          />

          {selectedFile && (
            <div className="document-uploader__file">
              <span className="document-uploader__file-icon">
                <FileText size={22} aria-hidden="true" />
              </span>

              <span className="document-uploader__file-info">
                <strong title={selectedFile.name}>{selectedFile.name}</strong>

                <small>
                  PDF document · {formatFileSize(selectedFile.size)}
                </small>
              </span>

              <button
                type="button"
                className="document-uploader__remove"
                onClick={handleRemoveFile}
                disabled={isProcessing}
                aria-label="Remove selected document"
                title="Remove selected document"
              >
                <X size={19} aria-hidden="true" />
              </button>
            </div>
          )}

          {validationError && (
            <p className="document-uploader__error" role="alert">
              <AlertCircle size={17} aria-hidden="true" />
              {validationError}
            </p>
          )}

          <div className="document-uploader__actions">
            <button
              type="submit"
              className="document-uploader__submit"
              disabled={!selectedFile || isProcessing || isComplete}
            >
              {isProcessing ? (
                <LoaderCircle
                  size={18}
                  className="document-uploader__spinner"
                  aria-hidden="true"
                />
              ) : isComplete ? (
                <Check size={18} aria-hidden="true" />
              ) : null}

              <span>{buttonLabel}</span>

              {!isProcessing && !isComplete && (
                <ArrowRight size={17} aria-hidden="true" />
              )}
            </button>

            <span className="document-uploader__helper">
              One document per upload
            </span>
          </div>
        </form>

        <DocumentProcessingSummary
          status={status}
          document={uploadedDocument}
          uploadOutcome={uploadOutcome}
          error={error}
          hasSelectedFile={selectedFile !== null}
        />
      </div>
    </section>
  );
}

export default DocumentUploader;
