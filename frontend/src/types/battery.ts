/**
 * Core type definitions for EV Battery Intelligence Platform.
 * Matches FastAPI schemas and SQLAlchemy models.
 */

export interface BatteryPack {
  id: string;
  name: string;
  chemistry: string;
  series_cells: number;
  parallel_strings: number;
  nominal_cell_voltage_v: number;
  nominal_cell_capacity_ah: number;
  total_voltage_v: number;
  total_capacity_ah: number;
  is_synthetic: boolean;
  cells: Cell[];
}

export interface Cell {
  id: string;
  pack_id?: string;
  cell_index: number;
  cell_serial: string;
  initial_capacity_ah: number;
  nominal_voltage_v: number;
  is_synthetic: boolean;
  created_at?: string;
}

export interface TelemetryReading {
  id?: number;
  cell_id: string;
  timestamp: string;
  voltage_v: number;
  current_a: number;
  temperature_c: number;
  soc_reported?: number;
  cycle_count?: number;
  is_synthetic: boolean;
}

export interface SOCEstimate {
  cell_id: string;
  soc_history_ekf: number[];
  soc_history_cc: number[];
  final_soc_ekf: number;
  final_soc_cc: number;
  error_bound: number;
  is_synthetic: boolean;
}

export interface PredictionResult {
  cell_id: string;
  target_type: "SOH" | "RUL";
  predicted_value: number;
  lower_bound?: number;
  upper_bound?: number;
  confidence_interval?: number;
  unit: string;
  model_name: string;
  is_synthetic: boolean;
}

export interface AnomalyReport {
  cell_id: string;
  cycle_number: number;
  is_anomaly: boolean;
  thermal_risk_score: number;
  thermal_severity: "NORMAL" | "ELEVATED" | "CRITICAL";
  evidence: {
    peak_temperature_c?: number;
    temperature_rise_c?: number;
    max_heating_rate_c_per_s?: number;
    internal_resistance_proxy?: number;
    notes?: string[];
  };
  is_synthetic: boolean;
}

export interface AlertItem {
  id: number;
  cell_id?: string;
  pack_id?: string;
  alert_type: string;
  severity: "WARNING" | "CRITICAL";
  message: string;
  evidence: Record<string, unknown>;
  status: "ACTIVE" | "ACKNOWLEDGED" | "RESOLVED";
  created_at: string;
  resolved_at?: string;
}

export interface FeatureImpact {
  feature_name: string;
  feature_value: number;
  shap_value: number;
  direction: "increases_prediction" | "decreases_prediction";
}

export interface RuleFinding {
  rule_id: string;
  category: "IMPEDANCE" | "THERMAL" | "VOLTAGE" | "DEGRADATION";
  severity: "INFO" | "WARNING" | "CRITICAL";
  metric_name: string;
  measured_value: number;
  threshold_value: number;
  explanation: string;
}

export interface ExplainReport {
  cell_id: string;
  cycle_number: number;
  target_type: "SOH" | "RUL";
  predicted_value: number;
  lower_bound?: number;
  upper_bound?: number;
  base_value: number;
  feature_contributions: FeatureImpact[];
  top_positive_drivers: FeatureImpact[];
  top_negative_drivers: FeatureImpact[];
  rule_findings: RuleFinding[];
  narrative_explanation: string;
  is_grounded: boolean;
  grounding_audit: string[];
  is_synthetic: boolean;
}
