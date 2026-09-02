import type { ReactNode } from "react";

import {
  ChartIcon,
  ClockIcon,
  DatabaseIcon,
  GridIcon,
  PlayIcon,
  SlidersIcon,
  UploadIcon,
} from "./Icons";

const steps = [
  { label: "Projects", eyebrow: "Workspace", icon: GridIcon },
  { label: "Upload data", eyebrow: "01", icon: UploadIcon },
  { label: "Configure", eyebrow: "02", icon: SlidersIcon },
  { label: "Run experiment", eyebrow: "03", icon: PlayIcon },
  { label: "Results", eyebrow: "04", icon: ChartIcon },
  { label: "Batch prediction", eyebrow: "05", icon: DatabaseIcon },
  { label: "History", eyebrow: "Archive", icon: ClockIcon },
] as const;

interface LayoutProps {
  currentStep: number;
  onNavigate: (step: number) => void;
  projectName?: string;
  children: ReactNode;
}

export function Layout({ currentStep, onNavigate, projectName, children }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => onNavigate(0)} type="button">
          <span className="brand-mark">L</span>
          <span>
            <strong>LimiX</strong>
            <small>Workbench</small>
          </span>
        </button>
        <nav aria-label="Workbench steps">
          {steps.map((step, index) => {
            const Icon = step.icon;
            return (
              <button
                aria-current={currentStep === index ? "page" : undefined}
                className={`nav-item ${currentStep === index ? "active" : ""}`}
                key={step.label}
                onClick={() => onNavigate(index)}
                type="button"
              >
                <Icon />
                <span>
                  <small>{step.eyebrow}</small>
                  {step.label}
                </span>
              </button>
            );
          })}
        </nav>
        <div className="model-card">
          <span className="pulse" />
          <div>
            <small>LOCAL MODEL</small>
            <strong>LimiX-2M</strong>
            <span>No retrieval · CPU fallback</span>
          </div>
        </div>
      </aside>
      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Local workspace /</span>
            <strong>{projectName ?? "All projects"}</strong>
          </div>
          <span className="privacy-badge">Data stays on this machine</span>
        </header>
        <div className="content">{children}</div>
      </main>
    </div>
  );
}
