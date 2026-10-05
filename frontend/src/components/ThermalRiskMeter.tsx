import React from "react";
import { Flame, ShieldAlert, Thermometer } from "lucide-react";
import { AnomalyReport } from "../types/battery";

interface ThermalRiskMeterProps {
  report: AnomalyReport | null;
}

export const ThermalRiskMeter: React.FC<ThermalRiskMeterProps> = ({ report }) => {
  const score = report ? report.thermal_risk_score : 18.5;
  const severity = report ? report.thermal_severity : "NORMAL";
  const peakTemp = report?.evidence?.peak_temperature_c ?? 28.4;
  const tempRise = report?.evidence?.temperature_rise_c ?? 4.2;
  const maxHeatingRate = report?.evidence?.max_heating_rate_c_per_s ?? 0.002;
  const irProxy = report?.evidence?.internal_resistance_proxy ?? 0.054;

  const getSeverityBadgeClass = (level: string) => {
    switch (level) {
      case "CRITICAL":
        return "badge-critical";
      case "ELEVATED":
        return "badge-elevated";
      default:
        return "badge-normal";
    }
  };

  const getScoreColor = (val: number) => {
    if (val >= 70) return "var(--accent-rose)";
    if (val >= 40) return "var(--accent-amber)";
    return "var(--accent-emerald)";
  };

  return (
    <div className="card" style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Flame size={18} color={getScoreColor(score)} />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Thermal Runaway Risk Gauge</h2>
        </div>
        <div className={`badge ${getSeverityBadgeClass(severity)}`}>
          {severity === "CRITICAL" ? <ShieldAlert size={12} /> : <Thermometer size={12} />}
          <span>{severity} RISK</span>
        </div>
      </div>

      {/* Score Progress Bar */}
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.4rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Continuous Escalation Index</span>
          <span style={{ fontSize: "1.5rem", fontWeight: 700, fontFamily: "JetBrains Mono", color: getScoreColor(score) }}>
            {score.toFixed(1)} <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>/ 100</span>
          </span>
        </div>

        {/* Multi-segment Gauge */}
        <div style={{ width: "100%", height: "10px", background: "var(--bg-secondary)", borderRadius: "9999px", overflow: "hidden", display: "flex" }}>
          <div
            style={{
              width: `${Math.min(100, Math.max(0, score))}%`,
              background: `linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #f43f5e 100%)`,
              borderRadius: "9999px",
              transition: "width 0.4s ease",
            }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-dim)", marginTop: "0.25rem" }}>
          <span>0 (Optimal)</span>
          <span>40 (Elevated)</span>
          <span>70 (Critical SEI Breakdown)</span>
          <span>100</span>
        </div>
      </div>

      {/* Sensor Drivers */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.1rem", fontFamily: "JetBrains Mono", fontWeight: 700, color: peakTemp > 45 ? "var(--accent-rose)" : "var(--text-main)" }}>
            {peakTemp.toFixed(1)} °C
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>Peak Cell Temp (&lt;35°C safe)</div>
        </div>

        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.1rem", fontFamily: "JetBrains Mono", fontWeight: 700 }}>
            +{tempRise.toFixed(1)} °C
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>Discharge ΔT (&lt;12°C limit)</div>
        </div>

        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.1rem", fontFamily: "JetBrains Mono", fontWeight: 700 }}>
            {(maxHeatingRate * 60).toFixed(2)} °C/min
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>Max Transient Rate (dT/dt)</div>
        </div>

        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.1rem", fontFamily: "JetBrains Mono", fontWeight: 700 }}>
            {irProxy.toFixed(3)} Ω
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>IR Proxy (Joule I²R Heat)</div>
        </div>
      </div>

      {/* Audit Evidence Notes */}
      {report?.evidence?.notes && report.evidence.notes.length > 0 && (
        <div style={{ background: "var(--bg-secondary)", borderRadius: "0.375rem", padding: "0.5rem 0.75rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <div style={{ fontWeight: 600, color: "var(--text-main)", marginBottom: "0.2rem" }}>Engineering Evidence:</div>
          <ul style={{ paddingLeft: "1rem", lineHeight: 1.4 }}>
            {report.evidence.notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
