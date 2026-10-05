import React, { useEffect, useState } from "react";
import { api } from "./api/client";
import { AlertsPanel } from "./components/AlertsPanel";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { Header } from "./components/Header";
import { LiveTelemetry } from "./components/LiveTelemetry";
import { PackOverview } from "./components/PackOverview";
import { SOCTracker } from "./components/SOCTracker";
import { SOHRULCard } from "./components/SOHRULCard";
import { ThermalRiskMeter } from "./components/ThermalRiskMeter";
import {
  AlertItem,
  AnomalyReport,
  BatteryPack,
  Cell,
  ExplainReport,
  PredictionResult,
  SOCEstimate,
  TelemetryReading,
} from "./types/battery";

export const App: React.FC = () => {
  const [appName, setAppName] = useState("EV Battery Intelligence Platform");
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [packs, setPacks] = useState<BatteryPack[]>([]);
  const [selectedPack, setSelectedPack] = useState<BatteryPack | null>(null);
  const [selectedCell, setSelectedCell] = useState<Cell | null>(null);

  const [telemetry, setTelemetry] = useState<TelemetryReading[]>([]);
  const [socData, setSOCData] = useState<SOCEstimate | null>(null);
  const [sohResult, setSOHResult] = useState<PredictionResult | null>(null);
  const [rulResult, setRULResult] = useState<PredictionResult | null>(null);
  const [anomalyReport, setAnomalyReport] = useState<AnomalyReport | null>(null);
  const [explainReport, setExplainReport] = useState<ExplainReport | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);

  // 1. Initial Health & Data Load
  const initData = async () => {
    try {
      const health = await api.getHealth();
      setIsBackendConnected(health.status === "healthy" || health.database_connected);
      if (health.app_name) setAppName(health.app_name);
    } catch {
      setIsBackendConnected(false);
    }

    try {
      const packList = await api.getPacks();
      if (packList.length > 0) {
        setPacks(packList);
        const currentPack = selectedPack ? packList.find((p) => p.id === selectedPack.id) || packList[0] : packList[0];
        setSelectedPack(currentPack);
        if (currentPack.cells && currentPack.cells.length > 0) {
          const currentCell = selectedCell ? currentPack.cells.find((c) => c.id === selectedCell.id) || currentPack.cells[0] : currentPack.cells[0];
          setSelectedCell(currentCell);
        }
      } else {
        // Fallback default demonstration pack
        const demoPack: BatteryPack = {
          id: "NASA_B0005_PACK",
          name: "NASA Li-ion Reference Module",
          chemistry: "Li-ion NMC",
          series_cells: 4,
          parallel_strings: 2,
          nominal_cell_voltage_v: 3.6,
          nominal_cell_capacity_ah: 2.0,
          total_voltage_v: 14.4,
          total_capacity_ah: 4.0,
          is_synthetic: false,
          cells: [
            { id: "B0005", cell_index: 0, cell_serial: "NASA-B0005", initial_capacity_ah: 1.85, nominal_voltage_v: 3.6, is_synthetic: false },
            { id: "B0006", cell_index: 1, cell_serial: "NASA-B0006", initial_capacity_ah: 2.03, nominal_voltage_v: 3.6, is_synthetic: false },
            { id: "B0007", cell_index: 2, cell_serial: "NASA-B0007", initial_capacity_ah: 1.89, nominal_voltage_v: 3.6, is_synthetic: false },
            { id: "SYNTH_C04", cell_index: 3, cell_serial: "SYNTH-04", initial_capacity_ah: 2.00, nominal_voltage_v: 3.6, is_synthetic: true },
          ],
        };
        setPacks([demoPack]);
        setSelectedPack(demoPack);
        setSelectedCell(demoPack.cells[0]);
      }
    } catch {
      // Offline fallback state
    }
  };

  // 2. Fetch Cell Specific Telemetry & Inference
  const refreshCellData = async (cell: Cell) => {
    setIsRefreshing(true);
    try {
      // Telemetry
      const telList = await api.getCellTelemetry(cell.id, 20);
      if (telList.length > 0) {
        setTelemetry(telList);
      } else {
        // Mock rolling readings for display
        const now = new Date();
        const mockTel: TelemetryReading[] = Array.from({ length: 15 }, (_, i) => ({
          cell_id: cell.id,
          timestamp: new Date(now.getTime() - (15 - i) * 2000).toISOString(),
          voltage_v: +(3.82 - i * 0.015).toFixed(3),
          current_a: 2.0,
          temperature_c: +(25.5 + i * 0.3).toFixed(1),
          is_synthetic: cell.is_synthetic,
        })).reverse();
        setTelemetry(mockTel);
      }

      // Feature dictionary for cycle 25
      const sampleFeatures: Record<string, number> = {
        voltage_max: 4.18,
        voltage_min: 2.65,
        voltage_mean: 3.68,
        voltage_drop_rate: 0.0016,
        temp_max: 29.4,
        temp_min: 24.0,
        temp_rise: 5.4,
        internal_resistance_proxy: 0.068,
        time_in_voltage_window: 3100.0,
        duration_s: 3300.0,
      };

      // Run live inference
      const [socRes, sohRes, rulRes, anomRes, expRes, alertList] = await Promise.allSettled([
        api.estimateSOC({
          cell_id: cell.id,
          initial_soc: 1.0,
          current_stream: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
          voltage_stream: [4.15, 4.10, 4.05, 4.00, 3.95, 3.90, 3.85, 3.80, 3.75, 3.70],
          dt_s: 1.0,
          is_synthetic: cell.is_synthetic,
        }),
        api.predictSOH({ cell_id: cell.id, cycle_number: 25, features: sampleFeatures, is_synthetic: cell.is_synthetic }),
        api.predictRUL({ cell_id: cell.id, cycle_number: 25, features: sampleFeatures, is_synthetic: cell.is_synthetic }),
        api.evaluateAnomaly({ cell_id: cell.id, cycle_number: 25, features: sampleFeatures, is_synthetic: cell.is_synthetic }),
        api.explainSOH({ cell_id: cell.id, cycle_number: 25, features: sampleFeatures, is_synthetic: cell.is_synthetic }),
        api.getAlerts(),
      ]);

      if (socRes.status === "fulfilled") setSOCData(socRes.value);
      if (sohRes.status === "fulfilled") setSOHResult(sohRes.value);
      if (rulRes.status === "fulfilled") setRULResult(rulRes.value);
      if (anomRes.status === "fulfilled") setAnomalyReport(anomRes.value);
      if (expRes.status === "fulfilled") setExplainReport(expRes.value);
      if (alertList.status === "fulfilled") setAlerts(alertList.value);
    } catch {
      // Error handling
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    initData();
  }, []);

  useEffect(() => {
    if (selectedCell) {
      refreshCellData(selectedCell);
    }
  }, [selectedCell?.id]);

  // Auto-refresh interval
  useEffect(() => {
    if (!autoRefresh || !selectedCell) return;
    const interval = setInterval(() => {
      refreshCellData(selectedCell);
    }, 4000);
    return () => clearInterval(interval);
  }, [autoRefresh, selectedCell?.id]);

  // Alert Handlers
  const handleAcknowledge = async (id: number) => {
    try {
      await api.updateAlertStatus(id, "ACKNOWLEDGED");
      setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, status: "ACKNOWLEDGED" } : a)));
    } catch {
      // Handled
    }
  };

  const handleResolve = async (id: number) => {
    try {
      await api.updateAlertStatus(id, "RESOLVED");
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch {
      // Handled
    }
  };

  const isSyntheticActive = Boolean(selectedPack?.is_synthetic || selectedCell?.is_synthetic);

  return (
    <div className="dashboard-container">
      {/* App Bar & Status */}
      <Header
        appName={appName}
        isBackendConnected={isBackendConnected}
        isSyntheticActive={isSyntheticActive}
        isRefreshing={isRefreshing}
        onRefresh={() => selectedCell && refreshCellData(selectedCell)}
        autoRefresh={autoRefresh}
        toggleAutoRefresh={() => setAutoRefresh(!autoRefresh)}
      />

      {/* Main Grid Layout */}
      <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        
        {/* Row 1: Pack Architecture & Cell Balance */}
        <PackOverview
          packs={packs}
          selectedPack={selectedPack}
          onSelectPack={(p) => {
            setSelectedPack(p);
            if (p.cells && p.cells.length > 0) setSelectedCell(p.cells[0]);
          }}
          selectedCell={selectedCell}
          onSelectCell={(c) => setSelectedCell(c)}
        />

        {/* Row 2: Dual Core Models (SOC Tracker & SOH/RUL Trajectory) */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: "1.25rem" }}>
          <SOCTracker socData={socData} cellId={selectedCell ? selectedCell.id : "Cell"} />
          <SOHRULCard
            sohResult={sohResult}
            rulResult={rulResult}
            cycleNumber={25}
          />
        </div>

        {/* Row 3: Safety & Live Telemetry Trio */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.25rem" }}>
          <ThermalRiskMeter report={anomalyReport} />
          <AlertsPanel
            alerts={alerts}
            onAcknowledge={handleAcknowledge}
            onResolve={handleResolve}
          />
          <LiveTelemetry
            telemetry={telemetry}
            cellId={selectedCell ? selectedCell.id : "Cell"}
          />
        </div>

        {/* Row 4: Explainability (Phase 9 Integration) */}
        <ExplanationPanel report={explainReport} />

      </div>
    </div>
  );
};
