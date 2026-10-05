import React from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BookOpen, ShieldCheck } from "lucide-react";
import { ExplainReport } from "../types/battery";

interface ExplanationPanelProps {
  report: ExplainReport | null;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({ report }) => {
  if (!report) {
    return (
      <div className="card" style={{ height: "100%", display: "flex", justifyContent: "center", alignItems: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Run inference or select a cell to view SHAP attributions and rule explanations.</p>
      </div>
    );
  }

  // Format SHAP data for horizontal bar chart
  const shapData = report.feature_contributions.slice(0, 6).map((c) => ({
    name: c.feature_name.replace(/_/g, " "),
    impact: +(c.shap_value * 100).toFixed(2),
    direction: c.direction,
    absImpact: Math.abs(c.shap_value),
  }));

  return (
    <div className="card" style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <BookOpen size={18} color="var(--accent-blue)" />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Explainable AI (TreeSHAP & Physics Grounding)</h2>
        </div>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          {report.is_grounded ? (
            <span className="badge badge-normal" title="All numbers verified against computed telemetry">
              <ShieldCheck size={12} />
              <span>Anti-Hallucination Verified</span>
            </span>
          ) : (
            <span className="badge badge-critical">
              <span>Unverified Claims Detected</span>
            </span>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
        
        {/* Left: SHAP Feature Attributions Bar Chart */}
        <div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Top Feature Drivers (Δ SOH % Impact)
          </div>
          <div style={{ width: "100%", height: 190 }}>
            <ResponsiveContainer>
              <BarChart
                data={shapData}
                layout="vertical"
                margin={{ top: 0, right: 20, left: 70, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                <XAxis type="number" stroke="#64748b" fontSize={10} tickLine={false} tickFormatter={(v) => `${v}%`} />
                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={10} tickLine={false} width={80} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", borderColor: "#334155", borderRadius: "0.375rem", fontSize: "0.75rem" }}
                  formatter={(val: number) => [`${val > 0 ? "+" : ""}${val}% impact`]}
                />
                <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                  {shapData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.impact >= 0 ? "#10b981" : "#f43f5e"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-dim)", marginTop: "0.2rem" }}>
            <span style={{ color: "#f43f5e" }}>◀ Accelerating Ageing (Negative)</span>
            <span style={{ color: "#10b981" }}>Preserving Health (Positive) ▶</span>
          </div>
        </div>

        {/* Right: Grounded Engineering Narrative */}
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Automated Diagnostic Summary (Zero Invention)
          </div>
          <div style={{
            flex: 1,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "0.5rem",
            padding: "0.75rem",
            fontSize: "0.8rem",
            lineHeight: 1.5,
            color: "var(--text-main)",
            overflowY: "auto",
            maxHeight: "190px",
          }}>
            <p>{report.narrative_explanation}</p>
          </div>
        </div>
      </div>

      {/* Domain Rule Findings List */}
      {report.rule_findings && report.rule_findings.length > 0 && (
        <div style={{ marginTop: "0.75rem", borderTop: "1px solid var(--border-subtle)", paddingTop: "0.5rem" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.4rem" }}>
            Electrochemical Rule Evaluations:
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
            {report.rule_findings.map((f, i) => (
              <span
                key={i}
                style={{
                  fontSize: "0.7rem",
                  padding: "0.2rem 0.5rem",
                  borderRadius: "0.25rem",
                  background: f.severity === "CRITICAL" ? "rgba(244, 63, 94, 0.15)" : f.severity === "WARNING" ? "rgba(245, 158, 11, 0.15)" : "rgba(30, 41, 59, 0.8)",
                  border: `1px solid ${f.severity === "CRITICAL" ? "rgba(244, 63, 94, 0.3)" : f.severity === "WARNING" ? "rgba(245, 158, 11, 0.3)" : "#334155"}`,
                  color: f.severity === "CRITICAL" ? "#fb7185" : f.severity === "WARNING" ? "#fbbf24" : "#94a3b8",
                }}
                title={f.explanation}
              >
                <strong>{f.rule_id}:</strong> {f.metric_name} = {f.measured_value}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
