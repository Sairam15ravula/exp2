/**
 * Direct REST API client talking to FastAPI backend.
 * Rule 10: No separate Node gateway. Talks directly to FastAPI.
 */

import {
  AlertItem,
  AnomalyReport,
  BatteryPack,
  ExplainReport,
  PredictionResult,
  SOCEstimate,
  TelemetryReading,
} from "../types/battery";

const BASE_URL = "/api/v1";

async function ensureAuthToken(): Promise<string | null> {
  let token = localStorage.getItem("bms_auth_token");
  if (!token) {
    try {
      const res = await fetch(`${BASE_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: "engineer", password: "engineer123" }),
      });
      if (res.ok) {
        const data = await res.json();
        token = data.access_token;
        if (token) localStorage.setItem("bms_auth_token", token);
      }
    } catch {
      // Offline fallback
    }
  }
  return token;
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = await ensureAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let errorDetail = "API Error";
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      errorDetail = res.statusText;
    }
    throw new Error(`[${res.status}] ${errorDetail}`);
  }

  return res.json();
}

export const api = {
  // System Health
  async getHealth(): Promise<{ status: string; app_name: string; database_connected: boolean }> {
    return request("/health");
  },

  // Battery Packs & Cells
  async getPacks(): Promise<BatteryPack[]> {
    return request("/packs");
  },

  async getPack(packId: string): Promise<BatteryPack> {
    return request(`/packs/${packId}`);
  },

  // Telemetry
  async getCellTelemetry(cellId: string, limit = 50): Promise<TelemetryReading[]> {
    return request(`/telemetry/cells/${cellId}?limit=${limit}`);
  },

  async ingestCellTelemetry(readings: TelemetryReading[]): Promise<unknown> {
    return request("/telemetry/ingest", {
      method: "POST",
      body: JSON.stringify({ readings }),
    });
  },

  // Safety Alerts
  async getAlerts(statusFilter?: string): Promise<AlertItem[]> {
    const query = statusFilter ? `?status=${statusFilter}` : "";
    return request(`/alerts${query}`);
  },

  async updateAlertStatus(alertId: number, status: "ACTIVE" | "ACKNOWLEDGED" | "RESOLVED"): Promise<AlertItem> {
    return request(`/alerts/${alertId}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    });
  },

  // Inference (Phases 3, 4, 5, 6)
  async estimateSOC(payload: {
    cell_id: string;
    initial_soc: number;
    current_stream: number[];
    voltage_stream: number[];
    dt_s: number;
    nominal_capacity_ah?: number;
    is_synthetic?: boolean;
  }): Promise<SOCEstimate> {
    return request("/inference/soc", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async predictSOH(payload: {
    cell_id: string;
    cycle_number: number;
    features: Record<string, number>;
    is_synthetic?: boolean;
  }): Promise<PredictionResult> {
    return request("/inference/soh", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async predictRUL(payload: {
    cell_id: string;
    cycle_number: number;
    features: Record<string, number>;
    is_synthetic?: boolean;
  }): Promise<PredictionResult> {
    return request("/inference/rul", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async evaluateAnomaly(payload: {
    cell_id: string;
    cycle_number: number;
    features: Record<string, number>;
    max_temp_c?: number;
    temp_rise_rate?: number;
    is_synthetic?: boolean;
  }): Promise<AnomalyReport> {
    return request("/inference/anomaly", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  // Explainability (Phase 9)
  async explainSOH(payload: {
    cell_id: string;
    cycle_number: number;
    features: Record<string, number>;
    is_synthetic?: boolean;
  }): Promise<ExplainReport> {
    return request("/explain/soh", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async explainRUL(payload: {
    cell_id: string;
    cycle_number: number;
    features: Record<string, number>;
    is_synthetic?: boolean;
  }): Promise<ExplainReport> {
    return request("/explain/rul", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
