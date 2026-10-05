import React from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Radio } from "lucide-react";
import { TelemetryReading } from "../types/battery";

interface LiveTelemetryProps {
  telemetry: TelemetryReading[];
  cellId: string;
}

export const LiveTelemetry: React.FC<LiveTelemetryProps> = ({ telemetry, cellId }) => {
  if (telemetry.length === 0) {
    return (
      <div className="card" style={{ height: "100%", display: "flex", justifyContent: "center", alignItems: "center" }}>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>No telemetry streaming for {cellId}.</p>
      </div>
    );
  }

  // Format data for chart (reverse so chronological left-to-right)
  const chartData = [...telemetry].reverse().map((t, idx) => ({
    time: t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : `${idx}s`,
    voltage: +t.voltage_v.toFixed(3),
    current: +t.current_a.toFixed(2),
    temp: +t.temperature_c.toFixed(1),
  }));

  const latest = telemetry[0];

  return (
    <div className="card" style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Radio size={18} color="var(--accent-blue)" />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Live Sensor Telemetry ({cellId})</h2>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span className="badge" style={{ background: "rgba(56, 189, 248, 0.15)", color: "#38bdf8", fontSize: "0.65rem" }}>
            Rule 5: Physical Bounds Checked
          </span>
        </div>
      </div>

      {/* Latest Telemetry Readouts */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.5rem", marginBottom: "0.75rem" }}>
        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.2rem", fontFamily: "JetBrains Mono", fontWeight: 700, color: "var(--accent-blue)" }}>
            {latest.voltage_v.toFixed(3)} V
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>Cell Terminal Voltage</div>
        </div>

        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.2rem", fontFamily: "JetBrains Mono", fontWeight: 700, color: latest.current_a >= 0 ? "#fbbf24" : "#34d399" }}>
            {latest.current_a.toFixed(2)} A
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>
            {latest.current_a >= 0 ? "Discharging" : "Charging"} Current
          </div>
        </div>

        <div className="metric-box" style={{ padding: "0.4rem 0.6rem" }}>
          <div style={{ fontSize: "1.2rem", fontFamily: "JetBrains Mono", fontWeight: 700, color: latest.temperature_c > 35 ? "#fb7185" : "var(--accent-emerald)" }}>
            {latest.temperature_c.toFixed(1)} °C
          </div>
          <div className="metric-label" style={{ fontSize: "0.65rem" }}>Surface Temperature</div>
        </div>
      </div>

      {/* Voltage & Temp Rolling Chart */}
      <div style={{ width: "100%", height: 160 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickLine={false} />
            <YAxis yAxisId="left" domain={[2.4, 4.3]} stroke="#38bdf8" fontSize={10} tickLine={false} />
            <YAxis yAxisId="right" orientation="right" domain={[15, 60]} stroke="#f43f5e" fontSize={10} tickLine={false} />
            <Tooltip contentStyle={{ background: "#0f172a", borderColor: "#334155", borderRadius: "0.375rem", fontSize: "0.75rem" }} />
            <Line yAxisId="left" type="monotone" dataKey="voltage" name="Voltage (V)" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="temp" name="Temp (°C)" stroke="#f43f5e" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-dim)", marginTop: "0.25rem" }}>
        <span>Blue: Voltage [2.5V min, 4.2V max]</span>
        <span>Red: Temperature (right axis)</span>
      </div>
    </div>
  );
};
