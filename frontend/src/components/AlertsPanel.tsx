import React from "react";
import { AlertCircle, AlertTriangle, Check, CheckCircle2, ShieldAlert } from "lucide-react";
import { AlertItem } from "../types/battery";

interface AlertsPanelProps {
  alerts: AlertItem[];
  onAcknowledge: (id: number) => void;
  onResolve: (id: number) => void;
}

export const AlertsPanel: React.FC<AlertsPanelProps> = ({ alerts, onAcknowledge, onResolve }) => {
  const activeAlerts = alerts.filter((a) => a.status !== "RESOLVED");

  return (
    <div className="card" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <ShieldAlert size={18} color="var(--accent-amber)" />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Active Safety Alerts</h2>
        </div>
        <span className="badge" style={{ background: activeAlerts.length > 0 ? "rgba(244, 63, 94, 0.15)" : "rgba(16, 185, 129, 0.15)", color: activeAlerts.length > 0 ? "#fb7185" : "#34d399" }}>
          {activeAlerts.length} Active
        </span>
      </div>

      <div style={{ flex: 1, overflowY: "auto", maxHeight: "240px", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {activeAlerts.length === 0 ? (
          <div style={{ padding: "2rem 1rem", textAlign: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
            <CheckCircle2 size={32} color="var(--accent-emerald)" style={{ margin: "0 auto 0.5rem" }} />
            <div>All systems nominal. Zero unresolved critical or elevated alerts.</div>
          </div>
        ) : (
          activeAlerts.map((alert) => {
            const isCritical = alert.severity === "CRITICAL";
            return (
              <div
                key={alert.id}
                style={{
                  background: isCritical ? "rgba(244, 63, 94, 0.08)" : "rgba(245, 158, 11, 0.08)",
                  border: `1px solid ${isCritical ? "rgba(244, 63, 94, 0.3)" : "rgba(245, 158, 11, 0.3)"}`,
                  borderRadius: "0.5rem",
                  padding: "0.6rem 0.75rem",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.5rem", marginBottom: "0.3rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                    {isCritical ? (
                      <AlertCircle size={15} color="#fb7185" />
                    ) : (
                      <AlertTriangle size={15} color="#fbbf24" />
                    )}
                    <span style={{ fontSize: "0.75rem", fontWeight: 700, color: isCritical ? "#fb7185" : "#fbbf24" }}>
                      {alert.alert_type}
                    </span>
                    {alert.cell_id && (
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono" }}>
                        ({alert.cell_id})
                      </span>
                    )}
                  </div>
                  <span className="badge" style={{ fontSize: "0.65rem", padding: "0.15rem 0.4rem", background: "var(--bg-secondary)" }}>
                    {alert.status}
                  </span>
                </div>

                <div style={{ fontSize: "0.8rem", color: "var(--text-main)", marginBottom: "0.5rem", lineHeight: 1.35 }}>
                  {alert.message}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "0.65rem", color: "var(--text-dim)" }}>
                    {new Date(alert.created_at).toLocaleTimeString()}
                  </span>
                  <div style={{ display: "flex", gap: "0.4rem" }}>
                    {alert.status === "ACTIVE" && (
                      <button
                        onClick={() => onAcknowledge(alert.id)}
                        className="btn"
                        style={{ fontSize: "0.7rem", padding: "0.2rem 0.5rem" }}
                      >
                        Acknowledge
                      </button>
                    )}
                    <button
                      onClick={() => onResolve(alert.id)}
                      className="btn"
                      style={{ fontSize: "0.7rem", padding: "0.2rem 0.5rem", background: "rgba(16, 185, 129, 0.2)", borderColor: "rgba(16, 185, 129, 0.4)", color: "#34d399" }}
                    >
                      <Check size={11} />
                      Resolve
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
