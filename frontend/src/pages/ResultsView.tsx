import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Metrics, Run } from "../services/api";

interface ResultsProps {
  run: Run;
  onBatch: () => void;
}

const labels: Record<string, string> = {
  accuracy: "Accuracy",
  precision_macro: "Precision",
  recall_macro: "Recall",
  f1_macro: "Macro F1",
  roc_auc: "ROC-AUC",
  log_loss: "Log loss",
  rmse: "RMSE",
  mae: "MAE",
  r2: "R²",
};

function metricValue(key: string, value: number): string {
  if (key === "log_loss" || key === "rmse" || key === "mae") return value.toFixed(3);
  return value.toFixed(3);
}

function ConfusionMatrix({ metrics }: { metrics: Metrics }) {
  const matrix = metrics.visualization.confusion_matrix ?? [];
  const classLabels = metrics.visualization.class_labels ?? [];
  const maximum = Math.max(...matrix.flat(), 1);
  return (
    <div className="confusion-wrap">
      <div
        className="confusion-grid"
        style={{ gridTemplateColumns: `64px repeat(${classLabels.length}, minmax(54px, 1fr))` }}
      >
        <span />
        {classLabels.map((label) => (
          <small key={`head-${label}`}>{label}</small>
        ))}
        {matrix.map((row, rowIndex) => (
          <div className="matrix-row" key={`row-${classLabels[rowIndex]}`}>
            <small>{classLabels[rowIndex]}</small>
            {row.map((value, columnIndex) => (
              <span
                key={`${rowIndex}-${columnIndex}`}
                style={{ backgroundColor: `rgba(25, 181, 133, ${0.12 + (value / maximum) * 0.75})` }}
              >
                {value}
              </span>
            ))}
          </div>
        ))}
      </div>
      <p className="chart-note">Rows are actual classes; columns are predicted classes.</p>
    </div>
  );
}

function RegressionPlot({ metrics }: { metrics: Metrics }) {
  const points = (metrics.visualization.actual ?? []).map((actual, index) => ({
    actual,
    predicted: metrics.visualization.predicted?.[index] ?? 0,
  }));
  return (
    <div className="chart-height">
      <ResponsiveContainer height="100%" width="100%">
        <ScatterChart margin={{ top: 12, right: 18, bottom: 12, left: 4 }}>
          <CartesianGrid stroke="#dfe7e3" strokeDasharray="3 3" />
          <XAxis dataKey="actual" name="Actual" tick={{ fill: "#66716d", fontSize: 12 }} type="number" />
          <YAxis dataKey="predicted" name="Predicted" tick={{ fill: "#66716d", fontSize: 12 }} type="number" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Legend />
          <Scatter data={points} fill="#19b585" name="LimiX predictions" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ResultsView({ run, onBatch }: ResultsProps) {
  const metrics = run.metrics!;
  const primaryEntries = Object.entries(metrics.limix);
  return (
    <section>
      <div className="page-heading results-heading">
        <div>
          <span className="eyebrow">Experiment complete</span>
          <h1>Results, without the guesswork.</h1>
          <p>LimiX-2M compared against a reproducible sklearn baseline.</p>
        </div>
        <div className="heading-actions">
          <a className="button secondary" href={`/api/runs/${run.id}/download`}>
            Download test predictions
          </a>
          <button className="button primary" onClick={onBatch} type="button">
            Predict new data
          </button>
        </div>
      </div>

      <div className="run-meta-strip">
        <span><small>DEVICE</small>{run.device?.toUpperCase()}</span>
        <span><small>INFERENCE</small>{run.inference_seconds?.toFixed(2)} s</span>
        <span><small>TEST ROWS</small>{metrics.split.test_rows}</span>
        <span><small>MODEL FEATURES</small>{metrics.split.model_features}</span>
      </div>

      <div className="metric-grid">
        {primaryEntries.map(([key, value]) => {
          const baseline = metrics.baseline[key];
          const lowerIsBetter = ["log_loss", "rmse", "mae"].includes(key);
          const delta = baseline === undefined ? null : lowerIsBetter ? baseline - value : value - baseline;
          return (
            <article className="metric-card" key={key}>
              <small>{labels[key] ?? key}</small>
              <strong>{metricValue(key, value)}</strong>
              {baseline !== undefined && (
                <span className={(delta ?? 0) >= 0 ? "positive" : "negative"}>
                  {(delta ?? 0) >= 0 ? "+" : ""}{delta?.toFixed(3)} vs baseline
                </span>
              )}
            </article>
          );
        })}
      </div>

      <div className="results-grid">
        <article className="panel chart-panel">
          <div className="panel-title">
            <div><span className="eyebrow">Model diagnostics</span><h2>{metrics.task_type === "classification" ? "Confusion matrix" : "Actual vs predicted"}</h2></div>
          </div>
          {metrics.task_type === "classification" ? <ConfusionMatrix metrics={metrics} /> : <RegressionPlot metrics={metrics} />}
        </article>
        <article className="panel comparison-panel">
          <span className="eyebrow">Head-to-head</span>
          <h2>LimiX vs baseline</h2>
          <div className="comparison-list">
            {primaryEntries.map(([key, value]) => (
              <div key={`compare-${key}`}>
                <span>{labels[key] ?? key}</span>
                <div className="comparison-values"><strong>{metricValue(key, value)}</strong><small>{metrics.baseline[key] === undefined ? "—" : metricValue(key, metrics.baseline[key])}</small></div>
              </div>
            ))}
          </div>
          <div className="legend-inline"><span className="limix-dot" /> LimiX-2M <span className="baseline-dot" /> sklearn baseline</div>
        </article>
      </div>
    </section>
  );
}

