import React from "react";
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity } from "lucide-react";
import { SOCEstimate } from "../types/battery";

interface SOCTrackerProps {
  socData: SOCEstimate | null;
  cellId: string;
}

export const SOCTracker: React.FC<SOCTrackerProps> = ({ socData, cellId }) => {
  if (!socData) {
    return (
      <div className="card" style={{ height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Awaiting telemetry stream to calculate SOC for {cellId}...</p>
      </div>
    );
  }

  // Format data points for Recharts
  const chartData = socData.soc_history_ekf.map((ekfVal, idx) => ({
    step: idx,
    ekf_soc: +(ekfVal * 100).toFixed(1),
    cc_soc: +(socData.soc_history_cc[idx] * 100).toFixed(1),
    upper_bound: +Math.min(100, (ekfVal + socData.error_bound) * 100).toFixed(1),
    lower_bound: +Math.max(0, (ekfVal - socData.error_bound) * 100).toFixed(1),
  }));

  return (
    <div className="card" style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Activity size={18} color="var(--accent-emerald)" />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>State of Charge (SOC) Observer</h2>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--accent-emerald)", fontWeight: 600 }}>● EKF (Physics 1RC)</span>
          <span style={{ fontSize: "0.75rem", color: "var(--accent-blue)", fontWeight: 600 }}>-- Coulomb Counting</span>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <div className="metric-box" style={{ padding: "0.5rem 0.75rem" }}>
          <div className="metric-val" style={{ color: "var(--accent-emerald)", fontSize: "1.3rem" }}>
            {(socData.final_soc_ekf * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: "0.7rem" }}>Final SOC (EKF)</div>
        </div>

        <div className="metric-box" style={{ padding: "0.5rem 0.75rem" }}>
          <div className="metric-val" style={{ color: "var(--accent-blue)", fontSize: "1.3rem" }}>
            {(socData.final_soc_cc * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: "0.7rem" }}>Coulomb Counting</div>
        </div>

        <div className="metric-box" style={{ padding: "0.5rem 0.75rem" }}>
          <div className="metric-val" style={{ fontSize: "1.3rem" }}>
            ±{(socData.error_bound * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: "0.7rem" }}>Observer Divergence</div>
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: "100%", height: 180 }}>
        <ResponsiveContainer>
          <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="step" stroke="#64748b" fontSize={11} tickLine={false} />
            <YAxis domain={[0, 100]} stroke="#64748b" fontSize={11} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#0f172a", borderColor: "#334155", borderRadius: "0.375rem", fontSize: "0.8rem" }}
              formatter={(value: number) => [`${value}%`]}
            />
            <Line type="monotone" dataKey="ekf_soc" name="EKF SOC" stroke="#10b981" strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="cc_soc" name="Coulomb Counting" stroke="#38bdf8" strokeWidth={1.5} strokeDasharray="4 4" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
