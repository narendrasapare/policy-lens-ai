import { useId, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import { CloudUpload, FileText, FolderOpen } from "lucide-react";

interface PdfDropzoneProps {
  disabled: boolean;
  hasSelectedFile: boolean;
  onFileSelect: (file: File) => void;
  onSelectionError: (message: string) => void;
}

function PdfDropzone({
  disabled,
  hasSelectedFile,
  onFileSelect,
  onSelectionError,
}: PdfDropzoneProps) {
  const inputId = useId();
  const dragDepth = useRef(0);
  const [isDragActive, setIsDragActive] = useState(false);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];

    if (file) {
      onFileSelect(file);
    }

    event.currentTarget.value = "";
  };

  const handleDragEnter = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (disabled) return;

    dragDepth.current += 1;
    setIsDragActive(true);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (!disabled) event.dataTransfer.dropEffect = "copy";
  };

  const handleDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (disabled) return;

    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setIsDragActive(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    dragDepth.current = 0;
    setIsDragActive(false);

    if (disabled) return;

    const files = Array.from(event.dataTransfer.files);

    if (files.length !== 1) {
      onSelectionError("Please select one PDF document at a time.");
      return;
    }

    onFileSelect(files[0]);
  };

  return (
    <label
      htmlFor={inputId}
      className={[
        "pdf-dropzone",
        isDragActive ? "pdf-dropzone--active" : "",
        disabled ? "pdf-dropzone--disabled" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        id={inputId}
        className="pdf-dropzone__input"
        type="file"
        accept=".pdf,application/pdf"
        onChange={handleInputChange}
        disabled={disabled}
      />

      <span className="pdf-dropzone__icon" aria-hidden="true">
        <CloudUpload size={28} strokeWidth={1.7} />
      </span>

      <strong className="pdf-dropzone__title">
        {isDragActive
          ? "Drop to select your PDF"
          : hasSelectedFile
            ? "Want to use a different PDF?"
            : "Drag and drop your PDF here"}
      </strong>

      <span className="pdf-dropzone__subtitle">
        Or choose a file from your computer
      </span>

      <span className="pdf-dropzone__browse">
        <FolderOpen size={16} aria-hidden="true" />
        Browse files
      </span>

      <span className="pdf-dropzone__hint">
        <FileText size={14} aria-hidden="true" />
        PDF only <span aria-hidden="true">·</span> Up to 10 MB
      </span>
    </label>
  );
}

export default PdfDropzone;
