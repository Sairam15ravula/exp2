import React from "react";
import { Battery, Layers } from "lucide-react";
import { BatteryPack, Cell } from "../types/battery";

interface PackOverviewProps {
  packs: BatteryPack[];
  selectedPack: BatteryPack | null;
  onSelectPack: (pack: BatteryPack) => void;
  selectedCell: Cell | null;
  onSelectCell: (cell: Cell) => void;
}

export const PackOverview: React.FC<PackOverviewProps> = ({
  packs,
  selectedPack,
  onSelectPack,
  selectedCell,
  onSelectCell,
}) => {
  if (!selectedPack) {
    return (
      <div className="card" style={{ textAlign: "center", padding: "2rem" }}>
        <p style={{ color: "var(--text-muted)" }}>No battery packs found. Register a pack to begin monitoring.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Layers size={18} color="var(--accent-blue)" />
          <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Pack Architecture & Cell Balancing</h2>
        </div>

        {/* Pack Selector Dropdown */}
        <select
          value={selectedPack.id}
          onChange={(e) => {
            const found = packs.find((p) => p.id === e.target.value);
            if (found) onSelectPack(found);
          }}
          style={{
            background: "var(--bg-secondary)",
            color: "var(--text-main)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.375rem",
            padding: "0.3rem 0.6rem",
            fontSize: "0.85rem",
            outline: "none",
          }}
        >
          {packs.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.id}) {p.is_synthetic ? "[SYNTH]" : ""}
            </option>
          ))}
        </select>
      </div>

      {/* Pack Top-Level Specifications (Rule 5: Series/Parallel scaling) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
        <div className="metric-box">
          <div className="metric-val" style={{ color: "var(--accent-blue)" }}>{selectedPack.total_voltage_v.toFixed(1)} <span style={{ fontSize: "1rem" }}>V</span></div>
          <div className="metric-label">Pack Voltage (Ns={selectedPack.series_cells})</div>
        </div>

        <div className="metric-box">
          <div className="metric-val" style={{ color: "var(--accent-emerald)" }}>{selectedPack.total_capacity_ah.toFixed(1)} <span style={{ fontSize: "1rem" }}>Ah</span></div>
          <div className="metric-label">Pack Capacity (Np={selectedPack.parallel_strings})</div>
        </div>

        <div className="metric-box">
          <div className="metric-val">{selectedPack.chemistry}</div>
          <div className="metric-label">Cell Chemistry</div>
        </div>

        <div className="metric-box">
          <div className="metric-val">{selectedPack.cells ? selectedPack.cells.length : 0}</div>
          <div className="metric-label">Monitored Cells</div>
        </div>
      </div>

      {/* Rule 8: Explicit Synthetic Warning Banner */}
      {selectedPack.is_synthetic && (
        <div style={{
          background: "rgba(245, 158, 11, 0.1)",
          border: "1px solid rgba(245, 158, 11, 0.3)",
          borderRadius: "0.375rem",
          padding: "0.5rem 0.75rem",
          fontSize: "0.75rem",
          color: "#fbbf24",
          marginBottom: "1rem",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}>
          <strong>RULE 8 NOTICE:</strong> This pack and its telemetry are synthetic representations generated for hardware-in-the-loop and fault simulation.
        </div>
      )}

      {/* Cell Matrix Grid */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Series-Parallel Cell Matrix (Click cell to inspect telemetry)
          </span>
          {selectedCell && (
            <span style={{ fontSize: "0.8rem", color: "var(--accent-blue)", fontWeight: 600 }}>
              Active Cell: {selectedCell.id}
            </span>
          )}
        </div>

        <div className="cell-grid">
          {selectedPack.cells && selectedPack.cells.map((cell) => {
            const isSelected = selectedCell?.id === cell.id;
            return (
              <div
                key={cell.id}
                onClick={() => onSelectCell(cell)}
                className={`cell-node ${isSelected ? "selected" : ""}`}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.2rem" }}>
                  <Battery size={13} color={isSelected ? "var(--accent-blue)" : "var(--text-dim)"} />
                  {cell.is_synthetic && (
                    <span style={{ fontSize: "0.6rem", color: "#fbbf24", fontWeight: 700 }}>SYN</span>
                  )}
                </div>
                <div style={{ fontSize: "0.8rem", fontWeight: 700, fontFamily: "JetBrains Mono" }}>{cell.id}</div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{cell.nominal_voltage_v.toFixed(1)}V</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
