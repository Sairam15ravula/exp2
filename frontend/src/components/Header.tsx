import React from "react";
import { Activity, AlertTriangle, CheckCircle, RefreshCw, Zap } from "lucide-react";

interface HeaderProps {
  appName: string;
  isBackendConnected: boolean;
  isSyntheticActive: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
  autoRefresh: boolean;
  toggleAutoRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  appName,
  isBackendConnected,
  isSyntheticActive,
  isRefreshing,
  onRefresh,
  autoRefresh,
  toggleAutoRefresh,
}) => {
  return (
    <header className="card" style={{ marginBottom: "1.25rem", padding: "1rem 1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        
        {/* Brand / Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{
            background: "linear-gradient(135deg, #0284c7, #10b981)",
            padding: "0.5rem",
            borderRadius: "0.5rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}>
            <Zap size={22} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: "1.2rem", fontWeight: 700, letterSpacing: "-0.01em" }}>{appName}</h1>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Hybrid Physics (1RC EKF) + Machine Learning Decision-Support BMS
            </p>
          </div>
        </div>

        {/* Status Indicators & Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
          
          {/* Rule 8: Global Synthetic Indicator Badge */}
          {isSyntheticActive ? (
            <div className="badge badge-synthetic" title="Rule 8: Synthetic Data Active">
              <AlertTriangle size={13} />
              <span>Synthetic Telemetry Active</span>
            </div>
          ) : (
            <div className="badge badge-real" title="NASA Battery Aging Lab Data">
              <CheckCircle size={13} />
              <span>NASA Li-ion Real Data</span>
            </div>
          )}

          {/* Backend Connection State */}
          <div className="badge" style={{
            background: isBackendConnected ? "rgba(16, 185, 129, 0.1)" : "rgba(244, 63, 94, 0.1)",
            color: isBackendConnected ? "#34d399" : "#fb7185",
            border: `1px solid ${isBackendConnected ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)"}`
          }}>
            <Activity size={13} />
            <span>{isBackendConnected ? "FastAPI Online" : "FastAPI Disconnected"}</span>
          </div>

          {/* Auto Refresh Toggle */}
          <button
            onClick={toggleAutoRefresh}
            className="btn"
            style={{
              borderColor: autoRefresh ? "var(--accent-blue)" : "var(--border-color)",
              color: autoRefresh ? "var(--accent-blue)" : "var(--text-muted)"
            }}
            title="Toggle 3s live polling"
          >
            <span>Auto: {autoRefresh ? "3s" : "OFF"}</span>
          </button>

          {/* Refresh Action */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="btn btn-primary"
          >
            <RefreshCw size={14} className={isRefreshing ? "animate-spin" : ""} style={{ animation: isRefreshing ? "spin 1s linear infinite" : "none" }} />
            <span>Sync</span>
          </button>
        </div>
      </div>
    </header>
  );
};
