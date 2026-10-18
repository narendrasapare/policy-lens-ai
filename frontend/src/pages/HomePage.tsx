import { useEffect, useState } from "react";

import PolicyWorkspace from "../features/workspace/PolicyWorkspace";
import { getHealthStatus } from "../api/healthApi";
import DocumentUploader from "../features/documents/components/DocumentUploader";

import "./HomePage.css";

type ConnectionStatus = "checking" | "online" | "offline";

const statusLabels: Record<ConnectionStatus, string> = {
  checking: "Checking connection",
  online: "Connected",
  offline: "Service unavailable",
};

function HomePage() {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("checking");

  const [connectionError, setConnectionError] = useState<string | null>(null);

  const [checkAttempt, setCheckAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    const checkConnection = async () => {
      try {
        const response = await getHealthStatus(controller.signal);

        if (controller.signal.aborted) {
          return;
        }

        if (response.status !== "ok") {
          throw new Error("Backend reported an unexpected status");
        }

        setConnectionStatus("online");
        setConnectionError(null);
      } catch (error: unknown) {
        if (controller.signal.aborted) {
          return;
        }

        setConnectionStatus("offline");
        setConnectionError(
          error instanceof Error
            ? error.message
            : "Unable to connect to the backend",
        );
      }
    };

    void checkConnection();

    return () => {
      controller.abort();
    };
  }, [checkAttempt]);

  const handleRetry = () => {
    setConnectionStatus("checking");
    setConnectionError(null);
    setCheckAttempt((previous) => previous + 1);
  };

  return (
    <div className="application">
      <header className="application-header">
        <div className="brand">
          <div className="brand-logo">P</div>

          <div className="brand-details">
            <span className="brand-name">PolicyLens AI</span>
            <span className="brand-description">
              HR policy assistant
            </span>
          </div>
        </div>

        <div
          className={`connection-indicator ${connectionStatus}`}
          role="status"
        >
          <span className="connection-dot" />
          <span>{statusLabels[connectionStatus]}</span>
        </div>
      </header>

      <main className="page-content">
        <section className="hero-section">
          <span className="section-eyebrow">
            YOUR POLICIES, EXPLAINED
          </span>

          <h1>
            Understand your documents.
            <br />
            <span>Find answers faster.</span>
          </h1>

          <p className="hero-description">
            Upload a policy or employee handbook, then ask what you need to know.
            Every answer comes with sources you can check.
          </p>
        </section>

        <DocumentUploader />

        <PolicyWorkspace />
        {connectionStatus === "offline" && <section className="system-status-card" role="alert">
          <h2>We couldn’t connect to PolicyLens</h2>
          <p>{connectionError}. Try again in a moment.</p>
          <button type="button" className="primary-button" onClick={handleRetry}>Retry connection</button>
        </section>}
      </main>

      <footer className="application-footer">
        <span>PolicyLens AI</span>
        <span>Newton School · PolicyLens AI</span>
      </footer>
    </div>
  );
}

export default HomePage;
