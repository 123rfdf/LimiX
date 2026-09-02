import { useCallback, useEffect, useRef, useState } from "react";

import { ArrowIcon, CheckIcon, DatabaseIcon, UploadIcon } from "./components/Icons";
import { Layout } from "./components/Layout";
import { ResultsView } from "./pages/ResultsView";
import {
  createProject,
  createRun,
  downloadBlob,
  getRun,
  inspectDataset,
  listProjects,
  listRuns,
  predictBatch,
  type Dataset,
  type Project,
  type Run,
  type RunConfig,
  type TaskType,
} from "./services/api";

const initialConfig: RunConfig = {
  project_id: "",
  target_column: "",
  feature_columns: [],
  task_type: "auto",
  test_size: 0.2,
  random_seed: 42,
};

function App() {
  const [step, setStep] = useState(0);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [projectName, setProjectName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [config, setConfig] = useState<RunConfig>(initialConfig);
  const [activeRun, setActiveRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<Run[]>([]);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [projectRows, runRows] = await Promise.all([listProjects(), listRuns()]);
      setProjects(projectRows);
      setHistory(runRows);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load local workspace.");
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void refresh(), 0);
    return () => {
      window.clearTimeout(initialLoad);
      if (pollRef.current) window.clearTimeout(pollRef.current);
    };
  }, [refresh]);

  const chooseProject = (project: Project, nextStep = 2) => {
    setSelectedProject(project);
    const columns = project.inspection.column_names;
    const target = columns.at(-1) ?? "";
    setConfig({
      ...initialConfig,
      project_id: project.id,
      target_column: target,
      feature_columns: columns.filter((column) => column !== target),
    });
    setStep(nextStep);
    setError(null);
  };

  const handleInspect = async () => {
    if (!uploadFile) return setError("Choose a CSV file first.");
    setBusy(true);
    setError(null);
    try {
      const inspected = await inspectDataset(uploadFile);
      setDataset(inspected);
      setProjectName(uploadFile.name.replace(/\.csv$/i, ""));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "CSV inspection failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleCreateProject = async () => {
    if (!dataset || !projectName.trim()) return setError("Give this project a name.");
    setBusy(true);
    try {
      const project = await createProject(projectName.trim(), dataset.id);
      await refresh();
      chooseProject({ ...project, inspection: dataset.inspection }, 2);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Project creation failed.");
    } finally {
      setBusy(false);
    }
  };

  const updateTarget = (target: string) => {
    const columns = selectedProject?.inspection.column_names ?? [];
    setConfig((value) => ({
      ...value,
      target_column: target,
      feature_columns: columns.filter((column) => column !== target),
    }));
  };

  const toggleFeature = (column: string) => {
    setConfig((value) => ({
      ...value,
      feature_columns: value.feature_columns.includes(column)
        ? value.feature_columns.filter((item) => item !== column)
        : [...value.feature_columns, column],
    }));
  };

  const pollRun = useCallback(async function poll(runId: string) {
    try {
      const run = await getRun(runId);
      setActiveRun(run);
      if (run.status === "completed") {
        await refresh();
        setStep(4);
      } else if (run.status === "failed") {
        setError(run.error_message ?? "Experiment failed.");
        await refresh();
      } else {
        pollRef.current = window.setTimeout(() => void poll(runId), 750);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read run status.");
    }
  }, [refresh]);

  const handleRun = async () => {
    if (!selectedProject) return setError("Select a project first.");
    if (!config.target_column || config.feature_columns.length === 0) {
      return setError("Choose a target and at least one feature.");
    }
    setBusy(true);
    setError(null);
    try {
      const run = await createRun(config);
      setActiveRun(run);
      void pollRun(run.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start the experiment.");
    } finally {
      setBusy(false);
    }
  };

  const handleBatch = async () => {
    if (!activeRun || !batchFile) return setError("Choose a completed run and a CSV file.");
    setBusy(true);
    setError(null);
    try {
      const blob = await predictBatch(activeRun.id, batchFile);
      downloadBlob(blob, `limix-batch-${activeRun.id.slice(0, 8)}.csv`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Batch prediction failed.");
    } finally {
      setBusy(false);
    }
  };

  const openHistoryRun = (run: Run) => {
    setActiveRun(run);
    const project = projects.find((item) => item.id === run.project_id) ?? null;
    setSelectedProject(project);
    if (run.status === "completed") setStep(4);
  };

  const page = (() => {
    if (step === 0) {
      return (
        <section>
          <div className="hero page-heading">
            <div>
              <span className="eyebrow">Private tabular intelligence</span>
              <h1>From raw CSV to a trustworthy model run.</h1>
              <p>Inspect, configure, compare, and export with LimiX-2M — entirely on your machine.</p>
              <button className="button primary" onClick={() => setStep(1)} type="button">Start a project <ArrowIcon /></button>
            </div>
            <div className="hero-visual" aria-label="Workbench overview">
              <div className="data-orbit orbit-one"><span>CSV</span></div>
              <div className="data-orbit orbit-two"><span>2M</span></div>
              <div className="model-core"><small>LOCAL</small><strong>LimiX</strong><span>classification · regression</span></div>
            </div>
          </div>
          <div className="section-title"><div><span className="eyebrow">Recent work</span><h2>Your projects</h2></div><button className="text-button" onClick={() => setStep(6)} type="button">View run history <ArrowIcon /></button></div>
          {projects.length === 0 ? (
            <div className="empty-state"><DatabaseIcon /><h3>No projects yet</h3><p>Upload a CSV to create your first local workspace.</p><button className="button secondary" onClick={() => setStep(1)} type="button">Upload data</button></div>
          ) : (
            <div className="project-grid">
              {projects.map((project) => (
                <button className="project-card" key={project.id} onClick={() => chooseProject(project)} type="button">
                  <div className="project-icon"><DatabaseIcon /></div>
                  <span className="status-pill">Ready</span>
                  <h3>{project.name}</h3><p>{project.dataset_filename}</p>
                  <div><span><strong>{project.inspection.rows}</strong> rows</span><span><strong>{project.inspection.columns}</strong> columns</span><span><strong>{project.run_count ?? 0}</strong> runs</span></div>
                </button>
              ))}
            </div>
          )}
        </section>
      );
    }
    if (step === 1) {
      return (
        <section>
          <div className="page-heading compact"><span className="eyebrow">Step 01</span><h1>Bring your data.</h1><p>CSV, UTF-8, up to 25 MB. Nothing leaves this computer.</p></div>
          <div className="upload-layout">
            <article className="panel upload-panel">
              <label className="dropzone">
                <input data-testid="dataset-file" onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)} type="file" accept=".csv,text/csv" />
                <span className="upload-circle"><UploadIcon /></span><strong>{uploadFile?.name ?? "Drop a CSV here"}</strong><p>{uploadFile ? `${(uploadFile.size / 1024).toFixed(1)} KB selected` : "or click to browse local files"}</p>
              </label>
              <button className="button primary full" disabled={!uploadFile || busy} onClick={() => void handleInspect()} type="button">{busy ? "Inspecting…" : "Inspect dataset"}</button>
            </article>
            <aside className="inspection-guide"><span className="eyebrow">Automatic checks</span><h2>Know what you have before modeling.</h2>{["Encoding and CSV structure", "Column types and missing values", "Duplicate rows and limits", "20-row private preview"].map((item) => <div key={item}><CheckIcon />{item}</div>)}</aside>
          </div>
          {dataset && <article className="panel inspection-panel"><div className="panel-title"><div><span className="eyebrow">Inspection passed</span><h2>{dataset.filename}</h2></div><div className="stat-pair"><span><strong>{dataset.inspection.rows}</strong> rows</span><span><strong>{dataset.inspection.columns}</strong> columns</span></div></div><div className="quality-grid"><span><small>NUMERIC</small>{dataset.inspection.numeric_columns.length}</span><span><small>CATEGORICAL</small>{dataset.inspection.categorical_columns.length}</span><span><small>MISSING CELLS</small>{Object.values(dataset.inspection.missing_values).reduce((a, b) => a + b, 0)}</span><span><small>DUPLICATE ROWS</small>{dataset.inspection.duplicate_rows}</span></div><label className="field"><span>Project name</span><input data-testid="project-name" value={projectName} onChange={(event) => setProjectName(event.target.value)} /></label><button className="button primary" data-testid="create-project" disabled={busy} onClick={() => void handleCreateProject()} type="button">Create project <ArrowIcon /></button></article>}
        </section>
      );
    }
    if (step === 2 && selectedProject) {
      return (
        <section>
          <div className="page-heading compact"><span className="eyebrow">Step 02</span><h1>Shape the experiment.</h1><p>Preprocessing is learned on training rows only. No leakage, no silent truncation.</p></div>
          <div className="config-grid">
            <article className="panel"><span className="eyebrow">Prediction target</span><h2>What should LimiX predict?</h2><label className="field"><span>Target column</span><select data-testid="target-column" value={config.target_column} onChange={(event) => updateTarget(event.target.value)}>{selectedProject.inspection.column_names.map((column) => <option key={column}>{column}</option>)}</select></label><span className="field-label">Task type</span><div className="segmented">{(["auto", "classification", "regression"] as TaskType[]).map((task) => <button className={config.task_type === task ? "selected" : ""} key={task} onClick={() => setConfig((value) => ({ ...value, task_type: task }))} type="button">{task}</button>)}</div></article>
            <article className="panel"><span className="eyebrow">Data split</span><h2>Hold out a fair test set.</h2><label className="field"><span>Test ratio <strong>{Math.round(config.test_size * 100)}%</strong></span><input min="0.1" max="0.8" step="0.05" type="range" value={config.test_size} onChange={(event) => setConfig((value) => ({ ...value, test_size: Number(event.target.value) }))} /></label><label className="field"><span>Random seed</span><input type="number" min="0" value={config.random_seed} onChange={(event) => setConfig((value) => ({ ...value, random_seed: Number(event.target.value) }))} /></label></article>
          </div>
          <article className="panel feature-panel"><div className="panel-title"><div><span className="eyebrow">Feature set</span><h2>{config.feature_columns.length} columns selected</h2></div><button className="text-button" onClick={() => setConfig((value) => ({ ...value, feature_columns: selectedProject.inspection.column_names.filter((column) => column !== value.target_column) }))} type="button">Select all</button></div><div className="feature-list">{selectedProject.inspection.column_names.filter((column) => column !== config.target_column).map((column) => <label key={column}><input checked={config.feature_columns.includes(column)} onChange={() => toggleFeature(column)} type="checkbox" /><span><strong>{column}</strong><small>{selectedProject.inspection.numeric_columns.includes(column) ? "numeric" : "categorical"}</small></span></label>)}</div><button className="button primary" data-testid="review-run" onClick={() => setStep(3)} type="button">Review experiment <ArrowIcon /></button></article>
        </section>
      );
    }
    if (step === 3 && selectedProject) {
      return (
        <section>
          <div className="page-heading compact"><span className="eyebrow">Step 03</span><h1>Ready when you are.</h1><p>LimiX is never trained or fine-tuned. Your training rows become its inference context.</p></div>
          <article className="panel run-panel"><div className="run-summary"><div><small>PROJECT</small><strong>{selectedProject.name}</strong></div><div><small>TASK</small><strong>{config.task_type}</strong></div><div><small>TARGET</small><strong>{config.target_column}</strong></div><div><small>FEATURES</small><strong>{config.feature_columns.length}</strong></div><div><small>SPLIT</small><strong>{Math.round((1 - config.test_size) * 100)} / {Math.round(config.test_size * 100)}</strong></div><div><small>SEED</small><strong>{config.random_seed}</strong></div></div><div className="run-callout"><div className="model-chip">L</div><div><span className="eyebrow">LimiX-2M · no retrieval</span><h2>One local foundation model, plus a transparent baseline.</h2><p>CUDA is preferred. If memory is exhausted, this run automatically retries on CPU and records the device used.</p></div></div>{activeRun && ["queued", "running"].includes(activeRun.status) ? <div className="progress-card"><span className="spinner" /><div><strong>{activeRun.status === "queued" ? "Queued" : "Running inference"}</strong><p>The page will update as soon as metrics and artifacts are ready.</p></div></div> : <button className="button primary large" data-testid="start-run" disabled={busy} onClick={() => void handleRun()} type="button">Run experiment <ArrowIcon /></button>}</article>
        </section>
      );
    }
    if (step === 4 && activeRun?.status === "completed" && activeRun.metrics) return <ResultsView run={activeRun} onBatch={() => setStep(5)} />;
    if (step === 5) {
      const completedRuns = history.filter((run) => run.status === "completed");
      return <section><div className="page-heading compact"><span className="eyebrow">Step 05</span><h1>Predict the next batch.</h1><p>Use the exact preprocessing and LimiX context from a completed experiment.</p></div><div className="upload-layout"><article className="panel upload-panel"><label className="field"><span>Completed run</span><select value={activeRun?.id ?? ""} onChange={(event) => setActiveRun(completedRuns.find((run) => run.id === event.target.value) ?? null)}><option value="">Select a run</option>{completedRuns.map((run) => <option key={run.id} value={run.id}>{run.config.target_column} · {run.id.slice(0, 8)}</option>)}</select></label><label className="dropzone compact-drop"><input onChange={(event) => setBatchFile(event.target.files?.[0] ?? null)} type="file" accept=".csv,text/csv" /><DatabaseIcon /><strong>{batchFile?.name ?? "Choose feature-only CSV"}</strong><p>Columns must match the completed experiment exactly.</p></label><button className="button primary full" disabled={!batchFile || !activeRun || busy} onClick={() => void handleBatch()} type="button">{busy ? "Predicting…" : "Predict and download CSV"}</button></article><aside className="inspection-guide"><span className="eyebrow">What gets exported</span><h2>Production-ready rows.</h2>{["Original feature columns preserved", "Prediction column appended", "Per-class probabilities when available", "UTF-8 CSV downloaded locally"].map((item) => <div key={item}><CheckIcon />{item}</div>)}</aside></div></section>;
    }
    if (step === 6) {
      return <section><div className="page-heading compact"><span className="eyebrow">Archive</span><h1>Experiment history.</h1><p>SQLite keeps every configuration, outcome, status, and artifact path across restarts.</p></div><article className="panel history-panel">{history.length === 0 ? <div className="empty-state"><DatabaseIcon /><h3>No experiments yet</h3></div> : <div className="history-table"><div className="history-head"><span>Created</span><span>Target / task</span><span>Status</span><span>Device</span><span /></div>{history.map((run) => <div className="history-row" key={run.id}><span>{new Date(run.created_at).toLocaleString()}</span><span><strong>{run.config.target_column}</strong><small>{run.config.task_type}</small></span><span><i className={`status-dot ${run.status}`} />{run.status}</span><span>{run.device ?? "—"}</span><span>{run.status === "completed" && <button className="text-button" onClick={() => openHistoryRun(run)} type="button">View results <ArrowIcon /></button>}</span></div>)}</div>}</article></section>;
    }
    return <div className="empty-state"><DatabaseIcon /><h3>Start with a project</h3><p>Select an existing project or upload a new CSV.</p><button className="button primary" onClick={() => setStep(0)} type="button">Back to projects</button></div>;
  })();

  return (
    <Layout currentStep={step} onNavigate={setStep} projectName={selectedProject?.name}>
      {error && <div className="error-banner" role="alert"><span>!</span><p>{error}</p><button aria-label="Dismiss error" onClick={() => setError(null)} type="button">×</button></div>}
      {page}
    </Layout>
  );
}

export default App;
