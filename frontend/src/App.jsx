import { useEffect, useMemo, useState } from "react";

const stageBlueprint = [
  {
    id: "01",
    title: "Dataset Spec",
    copy: "Lock the benchmark profile, split policy, quality floor, and runtime budget before any search begins.",
  },
  {
    id: "02",
    title: "Incumbent",
    copy: "Read or establish the current promoted run so every later mutation has a stable benchmark to beat.",
  },
  {
    id: "03",
    title: "Diagnosis",
    copy: "Convert current run behavior into explicit tags like overfit, underfit, plateau, or class imbalance.",
  },
  {
    id: "04",
    title: "Proposal",
    copy: "Choose one bounded mutation, with a plain hypothesis, expected upside, and explicit risk.",
  },
  {
    id: "05",
    title: "Evaluation Gate",
    copy: "Measure validation, holdout, runtime, and feature budget without touching the held-out test during search.",
  },
  {
    id: "06",
    title: "Promote Or Reject",
    copy: "Keep benchmark improvement separate from implementation readiness so the loop stays disciplined.",
  },
];

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed: ${response.status}`);
  }
  return response.json();
}

function App() {
  const [configs, setConfigs] = useState([]);
  const [selectedKey, setSelectedKey] = useState("");
  const [state, setState] = useState(null);
  const [journal, setJournal] = useState([]);
  const [glossary, setGlossary] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runDetail, setRunDetail] = useState(null);
  const [activeTab, setActiveTab] = useState("blueprint");
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function loadBootstrap() {
      try {
        const [configPayload, glossaryPayload] = await Promise.all([
          fetchJson("/api/configs"),
          fetchJson("/api/glossary"),
        ]);
        if (cancelled) {
          return;
        }
        setConfigs(configPayload);
        setGlossary(glossaryPayload);
        if (configPayload.length > 0) {
          setSelectedKey(configPayload[0].key);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(String(loadError.message || loadError));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    loadBootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedKey) {
      return;
    }
    let cancelled = false;
    async function loadSelectedConfig() {
      setError("");
      try {
        const [statePayload, journalPayload] = await Promise.all([
          fetchJson(`/api/configs/${selectedKey}/state`),
          fetchJson(`/api/configs/${selectedKey}/journal`),
        ]);
        if (cancelled) {
          return;
        }
        setState(statePayload);
        setJournal(journalPayload);
        const firstRunId = journalPayload[0]?.run_id ?? "";
        setSelectedRunId(firstRunId);
      } catch (loadError) {
        if (!cancelled) {
          setError(String(loadError.message || loadError));
        }
      }
    }
    loadSelectedConfig();
    return () => {
      cancelled = true;
    };
  }, [selectedKey]);

  useEffect(() => {
    if (!selectedRunId) {
      setRunDetail(null);
      return;
    }
    let cancelled = false;
    async function loadRun() {
      try {
        const payload = await fetchJson(`/api/runs/${selectedRunId}`);
        if (!cancelled) {
          setRunDetail(payload);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(String(loadError.message || loadError));
        }
      }
    }
    loadRun();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  async function refreshSelected() {
    if (!selectedKey) {
      return;
    }
    const [statePayload, journalPayload] = await Promise.all([
      fetchJson(`/api/configs/${selectedKey}/state`),
      fetchJson(`/api/configs/${selectedKey}/journal`),
    ]);
    setState(statePayload);
    setJournal(journalPayload);
    setSelectedRunId(journalPayload[0]?.run_id || "");
  }

  async function handleRunBaseline() {
    setBusyAction("baseline");
    setError("");
    try {
      await fetchJson(`/api/configs/${selectedKey}/baseline`, { method: "POST" });
      await refreshSelected();
      setActiveTab("journal");
    } catch (actionError) {
      setError(String(actionError.message || actionError));
    } finally {
      setBusyAction("");
    }
  }

  async function handleRunLoop() {
    setBusyAction("loop");
    setError("");
    try {
      await fetchJson(`/api/configs/${selectedKey}/loop`, {
        method: "POST",
        body: JSON.stringify({ steps: 1 }),
      });
      await refreshSelected();
      setActiveTab("journal");
    } catch (actionError) {
      setError(String(actionError.message || actionError));
    } finally {
      setBusyAction("");
    }
  }

  const selectedConfig = useMemo(
    () => configs.find((config) => config.key === selectedKey) ?? null,
    [configs, selectedKey],
  );

  const diagnosis = state?.diagnosis_preview?.diagnosis ?? null;
  const nextProposal = state?.diagnosis_preview?.next_proposal ?? null;
  const incumbent = state?.incumbent ?? null;
  const activeRun = runDetail?.entry ?? null;

  if (loading) {
    return <div className="app-shell"><div className="loading-card">Loading Treehouse Lab UI…</div></div>;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-kicker">Treehouse Lab</div>
          <h1>React Research Surface</h1>
          <p>
            A guided UI over the existing Python engine: benchmark pack, diagnosis, proposals,
            journal, and glossary.
          </p>
        </div>

        <section className="sidebar-section">
          <div className="section-label">Dataset Configs</div>
          <div className="config-list">
            {configs.map((config) => {
              const isActive = config.key === selectedKey;
              return (
                <button
                  key={config.key}
                  className={`config-card ${isActive ? "config-card--active" : ""}`}
                  onClick={() => setSelectedKey(config.key)}
                  type="button"
                >
                  <div className="config-card__title">{config.name}</div>
                  <div className="config-card__meta">
                    <span>{config.benchmark.profile}</span>
                    <span>{config.primary_metric}</span>
                  </div>
                  <div className="config-card__copy">{config.description}</div>
                </button>
              );
            })}
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-label">Legend</div>
          <ul className="legend-list">
            <li><strong>Benchmark status</strong> answers whether the run beat the benchmark.</li>
            <li><strong>Implementation readiness</strong> answers whether the run is credible enough to carry forward.</li>
            <li><strong>Diagnosis</strong> explains what the loop thinks is going wrong or right.</li>
          </ul>
        </section>
      </aside>

      <main className="main-panel">
        {error ? <div className="error-banner">{error}</div> : null}
        {selectedConfig && state ? (
          <>
            <header className="hero">
              <div>
                <div className="hero-kicker">Karpathy-style tabular autoresearch</div>
                <h2>{selectedConfig.name}</h2>
                <p>{selectedConfig.description}</p>
                <div className="pill-row">
                  <span className="pill">pack: {selectedConfig.benchmark.pack}</span>
                  <span className="pill">profile: {selectedConfig.benchmark.profile}</span>
                  <span className="pill">primary metric: {selectedConfig.primary_metric}</span>
                  <span className="pill">journal: {state.journal_count}</span>
                </div>
              </div>
              <div className="hero-actions">
                <button type="button" className="action-button" onClick={handleRunBaseline} disabled={busyAction !== ""}>
                  {busyAction === "baseline" ? "Running baseline…" : "Run Baseline"}
                </button>
                <button type="button" className="action-button action-button--secondary" onClick={handleRunLoop} disabled={busyAction !== ""}>
                  {busyAction === "loop" ? "Running loop…" : "Run 1-Step Loop"}
                </button>
              </div>
            </header>

            <nav className="tabs">
              {[
                ["blueprint", "Blueprint"],
                ["state", "Current State"],
                ["journal", "Journal"],
                ["glossary", "Glossary"],
              ].map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  className={`tab ${activeTab === key ? "tab--active" : ""}`}
                  onClick={() => setActiveTab(key)}
                >
                  {label}
                </button>
              ))}
            </nav>

            {activeTab === "blueprint" ? (
              <section className="panel-stack">
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <div className="section-label">System Blueprint</div>
                      <h3>Readable loop map</h3>
                    </div>
                    <div className="pill">selected config: {selectedConfig.key}.yaml</div>
                  </div>
                  <div className="stage-grid">
                    {stageBlueprint.map((stage) => (
                      <article key={stage.id} className="stage-tile">
                        <div className="stage-id">{stage.id}</div>
                        <h4>{stage.title}</h4>
                        <p>{stage.copy}</p>
                      </article>
                    ))}
                  </div>
                  <div className="signal-grid">
                    <SignalCard
                      label="Incumbent"
                      value={incumbent?.metric ? incumbent.metric.toFixed(4) : "n/a"}
                      copy="Current validation metric for the promoted run."
                    />
                    <SignalCard
                      label="Benchmark Status"
                      value={incumbent?.assessment?.benchmark_status ?? "none"}
                      copy="Whether the current best run established or improved the benchmark."
                    />
                    <SignalCard
                      label="Implementation Readiness"
                      value={incumbent?.assessment?.implementation_readiness ?? "not_started"}
                      copy="Whether the current best run clears the stricter readiness policy."
                    />
                    <SignalCard
                      label="Next Mutation"
                      value={nextProposal?.mutation_name ?? "baseline"}
                      copy="The next bounded move suggested by diagnosis-aware proposal selection."
                    />
                  </div>
                </div>
              </section>
            ) : null}

            {activeTab === "state" ? (
              <section className="panel-stack">
                <div className="state-grid">
                  <section className="panel">
                    <div className="section-label">Diagnosis</div>
                    <h3>{diagnosis?.primary_tag ?? "no_incumbent"}</h3>
                    <p className="lead-copy">{diagnosis?.summary ?? "Run the baseline first."}</p>
                    <div className="note-card">
                      <strong>Recommended direction</strong>
                      <p>{diagnosis?.recommended_direction ?? "Run the baseline first."}</p>
                    </div>
                    <div className="chip-group">
                      <span className="chip">preferred: {formatList(diagnosis?.preferred_mutations)}</span>
                      <span className="chip">avoid: {formatList(diagnosis?.avoided_mutations)}</span>
                    </div>
                  </section>

                  <section className="panel">
                    <div className="section-label">Suggested Proposal</div>
                    {nextProposal ? (
                      <>
                        <h3>{nextProposal.mutation_name}</h3>
                        <p className="lead-copy">{nextProposal.hypothesis}</p>
                        <dl className="detail-list">
                          <div>
                            <dt>Risk</dt>
                            <dd>{nextProposal.risk_level}</dd>
                          </div>
                          <div>
                            <dt>Expected upside</dt>
                            <dd>{nextProposal.expected_upside}</dd>
                          </div>
                          <div>
                            <dt>Override count</dt>
                            <dd>{Object.keys(nextProposal.params_override ?? {}).length}</dd>
                          </div>
                        </dl>
                        <details className="detail-box">
                          <summary>Parameter overrides</summary>
                          <pre>{JSON.stringify(nextProposal.params_override, null, 2)}</pre>
                        </details>
                      </>
                    ) : (
                      <p className="lead-copy">No proposal yet. Establish the baseline first.</p>
                    )}
                  </section>
                </div>
              </section>
            ) : null}

            {activeTab === "journal" ? (
              <section className="journal-layout">
                <section className="panel">
                  <div className="panel-header">
                    <div>
                      <div className="section-label">Run Journal</div>
                      <h3>{journal.length} runs</h3>
                    </div>
                  </div>
                  <div className="run-list">
                    {journal.map((entry) => (
                      <button
                        key={entry.run_id}
                        type="button"
                        className={`run-card ${selectedRunId === entry.run_id ? "run-card--active" : ""}`}
                        onClick={() => setSelectedRunId(entry.run_id)}
                      >
                        <div className="run-card__title">{entry.name}</div>
                        <div className="run-card__meta">
                          <span>{Number(entry.metric ?? 0).toFixed(4)}</span>
                          <span>{entry.assessment?.benchmark_status ?? "n/a"}</span>
                        </div>
                        <div className="run-card__copy">
                          {entry.diagnosis?.summary ?? entry.decision_reason ?? "No summary available."}
                        </div>
                      </button>
                    ))}
                  </div>
                </section>

                <section className="panel">
                  <div className="section-label">Run Inspector</div>
                  {activeRun ? (
                    <>
                      <h3>{activeRun.name}</h3>
                      <div className="signal-grid signal-grid--compact">
                        <SignalCard label="Metric" value={Number(activeRun.metric ?? 0).toFixed(4)} copy="Validation primary metric." />
                        <SignalCard label="Decision" value={activeRun.promoted ? "promote" : "reject"} copy={activeRun.decision_reason} />
                        <SignalCard label="Diagnosis" value={activeRun.diagnosis?.primary_tag ?? "n/a"} copy="Primary diagnosis tag." />
                        <SignalCard label="Readiness" value={activeRun.assessment?.implementation_readiness ?? "n/a"} copy="Current implementation-readiness label." />
                      </div>
                      <details className="detail-box" open>
                        <summary>Reason codes</summary>
                        <div className="code-pill-row">
                          {(activeRun.reason_codes ?? []).map((code) => (
                            <span key={code} className="code-pill">{code}</span>
                          ))}
                        </div>
                      </details>
                      <details className="detail-box">
                        <summary>Assessment</summary>
                        <pre>{JSON.stringify(activeRun.assessment, null, 2)}</pre>
                      </details>
                      <details className="detail-box">
                        <summary>Diagnosis</summary>
                        <pre>{JSON.stringify(activeRun.diagnosis, null, 2)}</pre>
                      </details>
                      <details className="detail-box">
                        <summary>Summary markdown</summary>
                        <pre>{runDetail?.artifacts?.summary_markdown ?? "No summary.md found."}</pre>
                      </details>
                    </>
                  ) : (
                    <p className="lead-copy">Select a run from the journal to inspect it.</p>
                  )}
                </section>
              </section>
            ) : null}

            {activeTab === "glossary" ? (
              <section className="panel">
                <div className="section-label">Glossary</div>
                <h3>Core terms</h3>
                <div className="glossary-grid">
                  {glossary.map((item) => (
                    <details key={item.term} className="glossary-item">
                      <summary>{item.term}</summary>
                      <p>{item.definition}</p>
                    </details>
                  ))}
                </div>
              </section>
            ) : null}
          </>
        ) : (
          <div className="loading-card">No dataset configs found.</div>
        )}
      </main>
    </div>
  );
}

function SignalCard({ label, value, copy }) {
  return (
    <article className="signal-card">
      <div className="signal-label">{label}</div>
      <div className="signal-value">{value}</div>
      <div className="signal-copy">{copy}</div>
    </article>
  );
}

function formatList(values) {
  if (!values || values.length === 0) {
    return "none";
  }
  return values.join(", ");
}

export default App;
