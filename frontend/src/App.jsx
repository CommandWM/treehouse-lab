import { useEffect, useMemo, useState } from "react";

const architectureFlow = [
  {
    id: "01",
    title: "Bring A Dataset",
    copy: "Start with a CSV plus an explicit target column. Intake inspects schema and label shape before a config exists.",
  },
  {
    id: "02",
    title: "Write The Spec",
    copy: "Treehouse Lab turns intake into a YAML dataset spec so split policy, metric, and budget remain reviewable.",
  },
  {
    id: "03",
    title: "Run Baseline",
    copy: "The baseline establishes the first incumbent and writes artifacts, metrics, and a readable summary.",
  },
  {
    id: "04",
    title: "Diagnose",
    copy: "The loop classifies the current run state into tags like no_incumbent, overfit, plateau, or imbalance.",
  },
  {
    id: "05",
    title: "Mutate Carefully",
    copy: "Exactly one bounded proposal is selected at a time from explicit mutation templates rather than free-form search.",
  },
  {
    id: "06",
    title: "Promote Or Reject",
    copy: "Validation wins, readiness gates, and journaled rationale decide whether the incumbent changes.",
  },
];

const operatingRules = [
  "Start from an explicit dataset spec, not hidden UI state.",
  "Keep the test set out of the search loop.",
  "Use bounded steps so the loop never feels unconstrained.",
  "Promote only meaningful wins over the incumbent.",
];

const workspaceViews = [
  ["intake", "1", "Intake"],
  ["state", "2", "Current State"],
  ["journal", "3", "Journal"],
];

const referenceViews = [
  ["architecture", "Architecture"],
  ["glossary", "Glossary"],
];

const intakeDefaults = {
  path: "",
  targetColumn: "",
  name: "",
  configKey: "",
  description: "",
  primaryMetric: "roc_auc",
  objective: "",
  validationSize: "0.2",
  testSize: "0.2",
  stratify: true,
};

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
  const [activeView, setActiveView] = useState("intake");
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState("");
  const [error, setError] = useState("");
  const [loopSteps, setLoopSteps] = useState("3");
  const [intakeForm, setIntakeForm] = useState(intakeDefaults);
  const [intakePreview, setIntakePreview] = useState(null);
  const [intakeBusy, setIntakeBusy] = useState("");
  const [templatesOpen, setTemplatesOpen] = useState(false);

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
        const firstUserConfig = configPayload.find((config) => config.benchmark?.pack === "user");
        setSelectedKey(firstUserConfig?.key ?? configPayload[0]?.key ?? "");
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
        setSelectedRunId(journalPayload[0]?.run_id ?? "");
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

  async function refreshSelected(configKey = selectedKey) {
    if (!configKey) {
      return;
    }
    const [statePayload, journalPayload] = await Promise.all([
      fetchJson(`/api/configs/${configKey}/state`),
      fetchJson(`/api/configs/${configKey}/journal`),
    ]);
    setState(statePayload);
    setJournal(journalPayload);
    setSelectedRunId(journalPayload[0]?.run_id || "");
  }

  async function refreshConfigs(nextSelectedKey = selectedKey) {
    const configPayload = await fetchJson("/api/configs");
    setConfigs(configPayload);
    if (nextSelectedKey) {
      setSelectedKey(nextSelectedKey);
    } else {
      const firstUserConfig = configPayload.find((config) => config.benchmark?.pack === "user");
      setSelectedKey(firstUserConfig?.key ?? configPayload[0]?.key ?? "");
    }
  }

  function updateIntakeField(field, value) {
    setIntakeForm((current) => ({ ...current, [field]: value }));
  }

  function applyIntakeSuggestions(pathValue) {
    const suggestion = suggestDatasetIdentity(pathValue);
    setIntakeForm((current) => ({
      ...current,
      name: current.name || suggestion.name,
      configKey: current.configKey || suggestion.key,
    }));
  }

  async function handleInspectDataset() {
    setIntakeBusy("inspect");
    setError("");
    try {
      const payload = await fetchJson("/api/intake/inspect", {
        method: "POST",
        body: JSON.stringify({
          path: intakeForm.path,
          target_column: intakeForm.targetColumn || undefined,
        }),
      });
      setIntakePreview(payload);
      applyIntakeSuggestions(intakeForm.path);
    } catch (actionError) {
      setError(String(actionError.message || actionError));
    } finally {
      setIntakeBusy("");
    }
  }

  async function createDatasetConfig(runBaseline = false) {
    setIntakeBusy(runBaseline ? "create-baseline" : "create");
    setError("");
    try {
      const created = await fetchJson("/api/intake/create", {
        method: "POST",
        body: JSON.stringify({
          path: intakeForm.path,
          target_column: intakeForm.targetColumn,
          name: intakeForm.name,
          config_key: intakeForm.configKey || undefined,
          description: intakeForm.description,
          primary_metric: intakeForm.primaryMetric,
          objective: intakeForm.objective,
          validation_size: Number(intakeForm.validationSize),
          test_size: Number(intakeForm.testSize),
          stratify: intakeForm.stratify,
        }),
      });
      setIntakePreview(created.inspection);
      await refreshConfigs(created.key);
      await refreshSelected(created.key);
      if (runBaseline) {
        await fetchJson(`/api/configs/${created.key}/baseline`, { method: "POST" });
        await refreshSelected(created.key);
        setActiveView("journal");
      } else {
        setActiveView("state");
      }
    } catch (actionError) {
      setError(String(actionError.message || actionError));
    } finally {
      setIntakeBusy("");
    }
  }

  async function handleRunBaseline() {
    setBusyAction("baseline");
    setError("");
    try {
      await fetchJson(`/api/configs/${selectedKey}/baseline`, { method: "POST" });
      await refreshSelected();
      setActiveView("journal");
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
        body: JSON.stringify({ steps: Number(loopSteps) }),
      });
      await refreshSelected();
      setActiveView("journal");
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
  const userConfigs = useMemo(
    () => configs.filter((config) => config.benchmark?.pack === "user"),
    [configs],
  );
  const templateConfigs = useMemo(
    () => configs.filter((config) => config.benchmark?.pack !== "user"),
    [configs],
  );

  const diagnosis = state?.diagnosis_preview?.diagnosis ?? null;
  const nextProposal = state?.diagnosis_preview?.next_proposal ?? null;
  const incumbent = state?.incumbent ?? null;
  const activeRun = runDetail?.entry ?? null;
  const hasSelectedDataset = Boolean(selectedConfig && state?.config?.key === selectedKey);
  const heroCopy = buildHeroCopy(activeView, hasSelectedDataset);

  if (loading) {
    return <div className="app-shell"><div className="loading-card">Loading Treehouse Lab UI…</div></div>;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-kicker">Treehouse Lab</div>
          <h1>Dataset-First Autoresearch</h1>
          <p>
            Start with intake. Built-in examples stay available as templates, but the workbench
            should always point you toward bringing your own data.
          </p>
        </div>

        <section className="sidebar-section">
          <div className="section-label">Workbench</div>
          <div className="sidebar-nav">
            {workspaceViews.map(([key, step, label]) => (
              <button
                key={key}
                type="button"
                className={`sidebar-tab ${activeView === key ? "sidebar-tab--active" : ""}`}
                onClick={() => setActiveView(key)}
              >
                <span className="sidebar-tab__step">{step}</span>
                <span>{label}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-label">Reference</div>
          <div className="sidebar-nav sidebar-nav--compact">
            {referenceViews.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`sidebar-tab sidebar-tab--secondary ${activeView === key ? "sidebar-tab--active" : ""}`}
                onClick={() => setActiveView(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        <section className="sidebar-section">
          <div className="section-label">Your Dataset Configs</div>
          {userConfigs.length > 0 ? (
            <div className="config-list">
              {userConfigs.map((config) => renderConfigCard(config, selectedKey, setSelectedKey))}
            </div>
          ) : (
            <div className="sidebar-note">
              Your imported dataset specs will appear here after you create them from intake.
            </div>
          )}
        </section>

        <section className="sidebar-section">
          <button
            type="button"
            className="section-toggle"
            onClick={() => setTemplatesOpen((current) => !current)}
          >
            <span className="section-label">Templates</span>
            <span>{templatesOpen ? "Hide" : `Show ${templateConfigs.length}`}</span>
          </button>
          {templatesOpen ? (
            <div className="config-list config-list--templates">
              {templateConfigs.map((config) => renderConfigCard(config, selectedKey, setSelectedKey))}
            </div>
          ) : (
            <div className="sidebar-note">
              Built-in examples stay tucked away here so intake remains the primary path.
            </div>
          )}
        </section>
      </aside>

      <main className="main-panel">
        {error ? <div className="error-banner">{error}</div> : null}

        <header className="hero">
          <div className="hero-copy">
            <div className="hero-kicker">{heroCopy.kicker}</div>
            <h2>{heroCopy.title}</h2>
            <p>{heroCopy.body}</p>
            <div className="pill-row">
              {heroCopy.pills.map((pill) => (
                <span key={pill} className="pill">{pill}</span>
              ))}
            </div>
          </div>
        </header>

        <section className="selection-card selection-card--standalone">
          <div className="section-label">Current Config</div>
          {hasSelectedDataset ? (
            <>
              <h3>{selectedConfig.name}</h3>
              <p className="lead-copy">{selectedConfig.description}</p>
              <div className="pill-row">
                <span className="pill">source: {selectedConfig.source?.kind ?? "n/a"}</span>
                <span className="pill">target: {selectedConfig.source?.target_column ?? "n/a"}</span>
                <span className="pill">metric: {selectedConfig.primary_metric}</span>
              </div>
              {selectedConfig.source?.path ? (
                <div className="selection-card__path">{selectedConfig.source.path}</div>
              ) : null}
              <div className="selection-card__controls">
                <label className="step-control">
                  <span>Loop steps</span>
                  <select value={loopSteps} onChange={(event) => setLoopSteps(event.target.value)}>
                    {["1", "3", "5", "10"].map((step) => (
                      <option key={step} value={step}>{step}</option>
                    ))}
                  </select>
                </label>
                <button type="button" className="action-button action-button--solid" onClick={handleRunBaseline} disabled={busyAction !== ""}>
                  {busyAction === "baseline" ? "Running baseline…" : "Run Baseline"}
                </button>
                <button type="button" className="action-button action-button--secondary" onClick={handleRunLoop} disabled={busyAction !== ""}>
                  {busyAction === "loop" ? `Running ${loopSteps}-step loop…` : `Run ${loopSteps}-Step Loop`}
                </button>
              </div>
            </>
          ) : (
            <>
              <h3>No dataset selected yet</h3>
              <p className="lead-copy">
                Start in intake, or open a template from the left side if you want a sample benchmark to inspect.
              </p>
            </>
          )}
        </section>

        {renderActiveView({
          activeView,
          hasSelectedDataset,
          intakeForm,
          intakePreview,
          intakeBusy,
          glossary,
          diagnosis,
          nextProposal,
          journal,
          selectedRunId,
          setSelectedRunId,
          activeRun,
          runDetail,
          incumbent,
          loopSteps,
          selectedConfig,
          updateIntakeField,
          handleInspectDataset,
          createDatasetConfig,
        })}
      </main>
    </div>
  );
}

function renderActiveView(context) {
  const {
    activeView,
    hasSelectedDataset,
    intakeForm,
    intakePreview,
    intakeBusy,
    glossary,
    diagnosis,
    nextProposal,
    journal,
    selectedRunId,
    setSelectedRunId,
    activeRun,
    runDetail,
    incumbent,
    loopSteps,
    selectedConfig,
    updateIntakeField,
    handleInspectDataset,
    createDatasetConfig,
  } = context;

  if (activeView === "intake") {
    return (
      <section className="panel-stack">
        <div className="intake-layout">
          <section className="panel">
            <div className="section-label">Dataset Intake</div>
            <h3>Bring your dataset in</h3>
            <p className="lead-copy">
              This is the first step. Point to a CSV, choose the target, inspect the schema, and let
              Treehouse Lab write the dataset spec for you.
            </p>
            <div className="intake-grid">
              <label className="field">
                <span>CSV path</span>
                <input
                  value={intakeForm.path}
                  onChange={(event) => updateIntakeField("path", event.target.value)}
                  placeholder="custom_datasets/my_dataset.csv"
                />
              </label>
              <label className="field">
                <span>Target column</span>
                <input
                  value={intakeForm.targetColumn}
                  onChange={(event) => updateIntakeField("targetColumn", event.target.value)}
                  placeholder="target"
                />
              </label>
              <label className="field">
                <span>Dataset name</span>
                <input
                  value={intakeForm.name}
                  onChange={(event) => updateIntakeField("name", event.target.value)}
                  placeholder="Developer Burnout"
                />
              </label>
              <label className="field">
                <span>Config key</span>
                <input
                  value={intakeForm.configKey}
                  onChange={(event) => updateIntakeField("configKey", event.target.value)}
                  placeholder="developer-burnout"
                />
              </label>
              <label className="field">
                <span>Primary metric</span>
                <select
                  value={intakeForm.primaryMetric}
                  onChange={(event) => updateIntakeField("primaryMetric", event.target.value)}
                >
                  <option value="roc_auc">roc_auc</option>
                  <option value="validation_accuracy">validation_accuracy</option>
                  <option value="validation_log_loss">validation_log_loss</option>
                </select>
              </label>
              <label className="field">
                <span>Stratify split</span>
                <select
                  value={String(intakeForm.stratify)}
                  onChange={(event) => updateIntakeField("stratify", event.target.value === "true")}
                >
                  <option value="true">true</option>
                  <option value="false">false</option>
                </select>
              </label>
              <label className="field">
                <span>Validation size</span>
                <input
                  value={intakeForm.validationSize}
                  onChange={(event) => updateIntakeField("validationSize", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Test size</span>
                <input
                  value={intakeForm.testSize}
                  onChange={(event) => updateIntakeField("testSize", event.target.value)}
                />
              </label>
              <label className="field field--full">
                <span>Description</span>
                <textarea
                  value={intakeForm.description}
                  onChange={(event) => updateIntakeField("description", event.target.value)}
                  placeholder="What this dataset represents and why it matters."
                  rows={3}
                />
              </label>
              <label className="field field--full">
                <span>Research objective</span>
                <textarea
                  value={intakeForm.objective}
                  onChange={(event) => updateIntakeField("objective", event.target.value)}
                  placeholder="Optional benchmark objective shown in the UI."
                  rows={3}
                />
              </label>
            </div>
            <div className="form-actions">
              <button type="button" className="action-button action-button--solid" onClick={handleInspectDataset} disabled={intakeBusy !== ""}>
                {intakeBusy === "inspect" ? "Inspecting…" : "Inspect CSV"}
              </button>
              <button type="button" className="action-button" onClick={() => createDatasetConfig(false)} disabled={intakeBusy !== ""}>
                {intakeBusy === "create" ? "Creating config…" : "Create Config"}
              </button>
              <button type="button" className="action-button action-button--secondary" onClick={() => createDatasetConfig(true)} disabled={intakeBusy !== ""}>
                {intakeBusy === "create-baseline" ? "Creating + running…" : "Create + Run Baseline"}
              </button>
            </div>
          </section>

          <section className="panel">
            <div className="section-label">Preview</div>
            <h3>Schema and target check</h3>
            {intakePreview ? (
              <>
                <div className="signal-grid signal-grid--compact">
                  <SignalCard label="Rows" value={intakePreview.row_count} copy="Observed rows in the CSV." />
                  <SignalCard label="Columns" value={intakePreview.column_count} copy="Total columns before selecting the target." />
                  <SignalCard
                    label="Feature count"
                    value={intakePreview.feature_count ?? "n/a"}
                    copy="Columns remaining after removing the target."
                  />
                  <SignalCard
                    label="Target check"
                    value={intakePreview.target?.binary_supported ? "binary ready" : "pending"}
                    copy={intakePreview.target?.error ?? "Set a target column to validate the current binary-classification scope."}
                  />
                </div>
                <div className="note-card">
                  <strong>Dataset path</strong>
                  <p>{intakePreview.path}</p>
                </div>
                {intakePreview.target ? (
                  <details className="detail-box" open>
                    <summary>Target encoding preview</summary>
                    {intakePreview.target.binary_supported ? (
                      <>
                        <div className="code-pill-row">
                          {(intakePreview.target.class_labels ?? []).map((item) => (
                            <span key={`${item.raw}-${item.encoded}`} className="code-pill">
                              {item.raw} → {item.encoded}
                            </span>
                          ))}
                        </div>
                        <p className="detail-copy">
                          Positive rate: {formatRate(intakePreview.target.positive_rate)} via {intakePreview.target.mapping_mode} mapping.
                        </p>
                      </>
                    ) : (
                      <p className="lead-copy">{intakePreview.target.error}</p>
                    )}
                  </details>
                ) : null}
                <details className="detail-box" open>
                  <summary>Columns</summary>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Type</th>
                          <th>Missing</th>
                          <th>Unique</th>
                        </tr>
                      </thead>
                      <tbody>
                        {intakePreview.columns.map((column) => (
                          <tr key={column.name}>
                            <td>{column.name}</td>
                            <td>{column.dtype}</td>
                            <td>{column.missing_count} ({formatRate(column.missing_rate)})</td>
                            <td>{column.unique_count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </>
            ) : (
              <p className="lead-copy">
                Inspect a CSV first. Templates stay available on the left, but this screen is the main
                path for bringing in your own dataset.
              </p>
            )}
          </section>
        </div>
      </section>
    );
  }

  if (activeView === "architecture") {
    return (
      <section className="panel-stack">
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="section-label">System Architecture</div>
              <h3>How the loop stays simple</h3>
            </div>
            <div className="pill">selected config: {selectedConfig?.key ?? "none"}.yaml</div>
          </div>
          <div className="flow-grid">
            {architectureFlow.map((stage, index) => (
              <article key={stage.id} className="flow-card">
                <div className="stage-id">{stage.id}</div>
                <h4>{stage.title}</h4>
                <p>{stage.copy}</p>
                {index < architectureFlow.length - 1 ? <div className="flow-arrow">→</div> : null}
              </article>
            ))}
          </div>
          <div className="signal-grid">
            <SignalCard
              label="Incumbent"
              value={incumbent?.metric ? incumbent.metric.toFixed(4) : "n/a"}
              copy="Current promoted validation metric."
            />
            <SignalCard
              label="Next Mutation"
              value={nextProposal?.mutation_name ?? "baseline"}
              copy="The next bounded move selected by diagnosis-aware proposal logic."
            />
            <SignalCard
              label="Benchmark Status"
              value={incumbent?.assessment?.benchmark_status ?? "none"}
              copy="Whether the current best run cleared the benchmark gate."
            />
            <SignalCard
              label="Readiness"
              value={incumbent?.assessment?.implementation_readiness ?? "not_started"}
              copy="Benchmark wins and implementation readiness remain separate labels."
            />
          </div>
        </div>
        <div className="state-grid">
          <section className="panel">
            <div className="section-label">Operating Rules</div>
            <h3>Guardrails</h3>
            <div className="rule-list">
              {operatingRules.map((rule) => (
                <div key={rule} className="rule-card">{rule}</div>
              ))}
            </div>
          </section>
          <section className="panel">
            <div className="section-label">Execution Modes</div>
            <h3>From baseline to short loop</h3>
            <div className="mode-grid">
              <article className="note-card">
                <strong>Baseline</strong>
                <p>Establish the first honest incumbent for a new dataset spec.</p>
              </article>
              <article className="note-card">
                <strong>{loopSteps}-step loop</strong>
                <p>Run a bounded research burst against the current incumbent and journal the outcome.</p>
              </article>
            </div>
            <details className="detail-box" open>
              <summary>Current benchmark objective</summary>
              <p className="detail-copy">{selectedConfig?.benchmark.objective || "No objective supplied."}</p>
            </details>
          </section>
        </div>
      </section>
    );
  }

  if (activeView === "glossary") {
    return (
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
    );
  }

  if (activeView === "state") {
    if (!hasSelectedDataset) {
      return (
        <section className="panel">
          <div className="section-label">Current State</div>
          <h3>No incumbent yet</h3>
          <p className="lead-copy">
            Create a dataset spec from intake and run the baseline first. Then diagnosis and the next bounded
            proposal will appear here.
          </p>
        </section>
      );
    }

    return (
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
    );
  }

  if (activeView === "journal") {
    if (!hasSelectedDataset) {
      return (
        <section className="panel">
          <div className="section-label">Run Journal</div>
          <h3>No runs logged</h3>
          <p className="lead-copy">
            The journal is empty until a dataset config exists and at least one baseline or loop step has run.
          </p>
        </section>
      );
    }

    return (
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
    );
  }

  return null;
}

function renderConfigCard(config, selectedKey, setSelectedKey) {
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
        <span>{config.source?.kind ?? "dataset"}</span>
        <span>{config.primary_metric}</span>
      </div>
      <div className="config-card__copy">{config.description}</div>
    </button>
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

function buildHeroCopy(activeView, hasSelectedDataset) {
  if (activeView === "state") {
    return {
      kicker: "Current State",
      title: hasSelectedDataset ? "Inspect the incumbent and next bounded move" : "State appears after baseline",
      body: hasSelectedDataset
        ? "Once a dataset spec exists, diagnosis and the next proposal become the main operational view."
        : "Start from intake, create the dataset spec, and establish the first incumbent before expecting diagnosis-aware guidance.",
      pills: ["1. intake", "2. baseline", "3. diagnosis"],
    };
  }
  if (activeView === "journal") {
    return {
      kicker: "Journal",
      title: "Review what the loop actually did",
      body: "The journal is the durable record of baseline, bounded mutations, and promote-or-reject decisions.",
      pills: ["artifacts", "reason codes", "promote or reject"],
    };
  }
  if (activeView === "architecture") {
    return {
      kicker: "Reference",
      title: "See how the system is put together",
      body: "Architecture stays available as an explainer, but it should not compete with intake for first attention.",
      pills: ["dataset spec", "incumbent", "bounded loop"],
    };
  }
  if (activeView === "glossary") {
    return {
      kicker: "Reference",
      title: "Keep the language readable",
      body: "Glossary terms stay one click away, separate from the workbench flow.",
      pills: ["diagnosis", "benchmark", "readiness"],
    };
  }
  return {
    kicker: "Dataset Intake",
    title: "Bring your dataset in first",
    body: "The product should start here: point to a CSV, validate the target, generate the spec, and then decide whether to run the baseline.",
    pills: ["inspect", "create config", "run baseline"],
  };
}

function formatList(values) {
  if (!values || values.length === 0) {
    return "none";
  }
  return values.join(", ");
}

function formatRate(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function suggestDatasetIdentity(pathValue) {
  const rawName = pathValue.split("/").filter(Boolean).pop() ?? "dataset";
  const stem = rawName.replace(/\.[^.]+$/, "");
  const prettyName = stem
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
  return {
    name: prettyName || "Dataset",
    key: stem
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "dataset",
  };
}

export default App;
