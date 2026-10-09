// TalentProof AI — Enterprise Frontend Controller
const API_BASE = "http://localhost:8000";

let currentLang = "AZ";
let blindScreening = true;
let requiredSkills = ["Python", "SQL", "Machine Learning", "Docker"];
let allSkillsPool = ["Python", "SQL", "Machine Learning", "Docker", "Tableau", "Git", "Power BI", "Pandas", "Scikit-learn"];

// Initial baseline candidates dataset (matches local ML engine evaluation)
let candidateDataset = [
  {
    id: "C-101",
    name: "Aysel Məmmədova",
    role: "Senior Data Analyst",
    exp_years: 4.5,
    resume_text: "Senior Data Analyst with 4.5 years experience. Strong background in Python and SQL for big data pipelines. Implemented Machine Learning models for churn prediction. Also experienced with Docker containers and Git workflows. Email: aysel.m@example.com Phone: +994 50 123 45 67",
    verified_skills: ["Python", "SQL", "Machine Learning", "Docker"],
    missing_skills: [],
    negated_skills: [],
    skill_match_ratio: 1.0,
    semantic_similarity: 0.88,
    exp_ratio: 2.25,
    suitability_score: 94.5,
    status: "QUALIFIED",
    evidence_breakdown: {
      "Python": "Strong background in Python and SQL for big data pipelines.",
      "SQL": "Strong background in Python and SQL for big data pipelines.",
      "Machine Learning": "Implemented Machine Learning models for churn prediction.",
      "Docker": "Also experienced with Docker containers and Git workflows."
    },
    masked_resume_text: "Senior Data Analyst with 4.5 years experience. Strong background in Python and SQL for big data pipelines. Implemented Machine Learning models for churn prediction. Also experienced with Docker containers and Git workflows. Email: [REDACTED_EMAIL] Phone: [REDACTED_PHONE]"
  },
  {
    id: "C-102",
    name: "Rauf Əliyev",
    role: "Junior BI Developer",
    exp_years: 1.5,
    resume_text: "Junior BI Developer with 1.5 years experience. Proficient in SQL query optimization and Tableau dashboards. Currently learning Python scripting. Has no experience in Docker deployment. Contact: rauf.aliyev@corp.az, +994 55 987 65 43",
    verified_skills: ["SQL", "Python"],
    missing_skills: ["Machine Learning"],
    negated_skills: ["Docker"],
    skill_match_ratio: 0.5,
    semantic_similarity: 0.52,
    exp_ratio: 0.75,
    suitability_score: 41.2,
    status: "NOT_QUALIFIED",
    evidence_breakdown: {
      "SQL": "Proficient in SQL query optimization and Tableau dashboards.",
      "Python": "Currently learning Python scripting.",
      "Docker": "[DISQUALIFIED: NEGATIVE CONTEXT] Has no experience in Docker deployment."
    },
    masked_resume_text: "Junior BI Developer with 1.5 years experience. Proficient in SQL query optimization and Tableau dashboards. Currently learning Python scripting. Has no experience in Docker deployment. Contact: [REDACTED_EMAIL], [REDACTED_PHONE]"
  },
  {
    id: "C-103",
    name: "Elmir Qasımov",
    role: "ML Engineer",
    exp_years: 3.0,
    resume_text: "Machine Learning Engineer with 3 years building predictive systems. Advanced Python, Scikit-learn, and SQL database tuning. Deploying containerized services with Docker. Email: elmir.q@work.io, Tel: +994 70 333 22 11",
    verified_skills: ["Python", "SQL", "Machine Learning", "Docker"],
    missing_skills: [],
    negated_skills: [],
    skill_match_ratio: 1.0,
    semantic_similarity: 0.91,
    exp_ratio: 1.5,
    suitability_score: 92.0,
    status: "QUALIFIED",
    evidence_breakdown: {
      "Python": "Advanced Python, Scikit-learn, and SQL database tuning.",
      "SQL": "Advanced Python, Scikit-learn, and SQL database tuning.",
      "Machine Learning": "Machine Learning Engineer with 3 years building predictive systems.",
      "Docker": "Deploying containerized services with Docker."
    },
    masked_resume_text: "Machine Learning Engineer with 3 years building predictive systems. Advanced Python, Scikit-learn, and SQL database tuning. Deploying containerized services with Docker. Email: [REDACTED_EMAIL], Tel: [REDACTED_PHONE]"
  },
  {
    id: "C-104",
    name: "Nigar Sultanova",
    role: "Database Administrator",
    exp_years: 5.0,
    resume_text: "Database Administrator with 5 years managing relational stores. Expert in SQL performance, indexing and ETL. Limited knowledge of Python. Never worked with Machine Learning or Docker. Contacts: n.sultanova@mail.ru",
    verified_skills: ["SQL"],
    missing_skills: ["Python", "Docker"],
    negated_skills: ["Machine Learning"],
    skill_match_ratio: 0.25,
    semantic_similarity: 0.44,
    exp_ratio: 2.5,
    suitability_score: 38.0,
    status: "NOT_QUALIFIED",
    evidence_breakdown: {
      "SQL": "Expert in SQL performance, indexing and ETL.",
      "Machine Learning": "[DISQUALIFIED: NEGATIVE CONTEXT] Never worked with Machine Learning or Docker."
    },
    masked_resume_text: "Database Administrator with 5 years managing relational stores. Expert in SQL performance, indexing and ETL. Limited knowledge of Python. Never worked with Machine Learning or Docker. Contacts: [REDACTED_EMAIL]"
  }
];

// Initialize DOM
document.addEventListener("DOMContentLoaded", () => {
  setupLanguageSwitcher();
  setupNavigationTabs();
  setupWhatIfControls();
  setupBlindScreening();
  setupUploadForm();
  setupModal();
  checkBackendHealth();
  renderScreeningResults();
});

// Internationalization
function setupLanguageSwitcher() {
  const langSelect = document.getElementById("lang-select");
  langSelect.value = currentLang;
  langSelect.addEventListener("change", (e) => {
    currentLang = e.target.value;
    updateTranslations();
    renderScreeningResults();
  });
  updateTranslations();
}

function updateTranslations() {
  const dict = TRANSLATIONS[currentLang] || TRANSLATIONS["AZ"];
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) {
      el.textContent = dict[key];
    }
  });
}

function t(key) {
  const dict = TRANSLATIONS[currentLang] || TRANSLATIONS["AZ"];
  return dict[key] || key;
}

// Tabs
function setupNavigationTabs() {
  const tabs = [
    { btn: "tab-btn-matrix", content: "tab-matrix-content" },
    { btn: "tab-btn-upload", content: "tab-upload-content" },
    { btn: "tab-btn-benchmark", content: "tab-benchmark-content" }
  ];

  tabs.forEach(({ btn, content }) => {
    document.getElementById(btn).addEventListener("click", () => {
      tabs.forEach(t => {
        const b = document.getElementById(t.btn);
        const c = document.getElementById(t.content);
        if (t.btn === btn) {
          b.classList.remove("border-transparent", "text-slate-500");
          b.classList.add("border-slate-900", "text-slate-900", "font-semibold");
          c.classList.remove("hidden");
        } else {
          b.classList.remove("border-slate-900", "text-slate-900", "font-semibold");
          b.classList.add("border-transparent", "text-slate-500");
          c.classList.add("hidden");
        }
      });
    });
  });
}

// What-If Skills Pool
function setupWhatIfControls() {
  const container = document.getElementById("skills-container");
  container.innerHTML = "";

  allSkillsPool.forEach(skill => {
    const isChecked = requiredSkills.includes(skill);
    const wrapper = document.createElement("label");
    wrapper.className = "flex items-center space-x-2 cursor-pointer";
    wrapper.innerHTML = `
      <input type="checkbox" value="${skill}" ${isChecked ? "checked" : ""} class="h-3.5 w-3.5 rounded border-slate-300 text-slate-900 focus:ring-0 skill-checkbox">
      <span class="text-xs text-slate-700">${skill}</span>
    `;
    container.appendChild(wrapper);
  });

  container.addEventListener("change", () => {
    requiredSkills = Array.from(container.querySelectorAll(".skill-checkbox:checked")).map(cb => cb.value);
    runWhatIfSimulation();
  });

  document.getElementById("required-exp-input").addEventListener("input", () => {
    runWhatIfSimulation();
  });

  document.getElementById("btn-re-screen").addEventListener("click", () => {
    triggerFullScreening();
  });
}

function setupBlindScreening() {
  const toggle = document.getElementById("blind-screening-toggle");
  toggle.addEventListener("change", (e) => {
    blindScreening = e.target.checked;
    renderScreeningResults();
  });
}

// Backend Health Check
async function checkBackendHealth() {
  const pill = document.getElementById("backend-status-pill");
  try {
    const res = await fetch(`${API_BASE}/`, { method: "GET" });
    if (res.ok) {
      pill.textContent = `${t("local_badge")} • ${t("server_status_online")}`;
      pill.className = "bg-emerald-950 text-emerald-300 border border-emerald-800 text-xs font-semibold px-3 py-1.5 rounded whitespace-nowrap flex-shrink-0 tracking-wider";
      return true;
    }
  } catch (err) {
    pill.textContent = `${t("local_badge")}`;
    pill.className = "bg-slate-800 text-slate-300 border border-slate-700 text-xs font-semibold px-3 py-1.5 rounded whitespace-nowrap flex-shrink-0 tracking-wider";
  }
  return false;
}

// What-If Simulation Logic
async function runWhatIfSimulation() {
  const reqExp = parseFloat(document.getElementById("required-exp-input").value) || 2.0;

  try {
    const payload = {
      required_exp_years: reqExp,
      required_skills: requiredSkills,
      candidates: candidateDataset.map(c => ({
        id: c.id,
        name: c.name,
        role: c.role,
        exp_years: c.exp_years,
        resume_text: c.resume_text
      }))
    };

    const res = await fetch(`${API_BASE}/api/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      applySimulationResults(data.results);
      return;
    }
  } catch (e) {
    // Fallback to local recalculation if API is in standalone mode
  }

  // Local simulation recalculation
  candidateDataset.forEach(c => {
    const verified = [];
    const missing = [];
    const negated = [];

    requiredSkills.forEach(s => {
      const lower = c.resume_text.toLowerCase();
      const sLower = s.toLowerCase();
      if (lower.includes(`no experience in ${sLower}`) || lower.includes(`never worked with ${sLower}`)) {
        negated.push(s);
      } else if (lower.includes(sLower)) {
        verified.push(s);
      } else {
        missing.push(s);
      }
    });

    c.verified_skills = verified;
    c.missing_skills = missing;
    c.negated_skills = negated;

    const matchRatio = requiredSkills.length > 0 ? verified.length / requiredSkills.length : 0;
    const expRatio = Math.min(2.0, c.exp_years / Math.max(0.5, reqExp));
    c.suitability_score = Math.round((matchRatio * 65 + Math.min(1.0, expRatio) * 35) * 10) / 10;
    c.status = (c.suitability_score >= 60 && negated.length === 0) ? "QUALIFIED" : "NOT_QUALIFIED";
  });

  renderScreeningResults();
}

function applySimulationResults(results) {
  results.forEach(res => {
    const cand = candidateDataset.find(c => c.id === res.id);
    if (cand) {
      cand.suitability_score = res.suitability_score;
      cand.status = res.status;
      cand.verified_skills = res.verified_skills;
      cand.missing_skills = res.missing_skills;
      cand.negated_skills = res.negated_skills;
      cand.evidence_breakdown = res.evidence_breakdown;
    }
  });
  renderScreeningResults();
}

async function triggerFullScreening() {
  const reqExp = parseFloat(document.getElementById("required-exp-input").value) || 2.0;
  const jobTitle = document.getElementById("job-title-input").value || "Data Analyst";

  try {
    const payload = {
      job_title: jobTitle,
      required_exp_years: reqExp,
      required_skills: requiredSkills,
      candidates: candidateDataset.map(c => ({
        id: c.id,
        name: c.name,
        role: c.role,
        exp_years: c.exp_years,
        resume_text: c.resume_text
      })),
      blind_screening: blindScreening
    };

    const res = await fetch(`${API_BASE}/api/screen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      applySimulationResults(data.results);
      return;
    }
  } catch (err) {
    // Local fallback
  }

  runWhatIfSimulation();
}

// Render Results Grid & Metrics
function renderScreeningResults() {
  const grid = document.getElementById("candidates-grid");
  grid.innerHTML = "";

  // Sort descending by suitability score
  const sorted = [...candidateDataset].sort((a, b) => b.suitability_score - a.suitability_score);

  // Update Metrics
  const total = sorted.length;
  const qualified = sorted.filter(c => c.status === "QUALIFIED").length;
  const qualRate = total > 0 ? Math.round((qualified / total) * 100) : 0;
  const meanScore = total > 0 ? (sorted.reduce((acc, c) => acc + c.suitability_score, 0) / total).toFixed(1) : "0.0";

  document.getElementById("metric-total").textContent = total;
  document.getElementById("metric-qualified").textContent = qualified;
  document.getElementById("metric-rate").textContent = `${qualRate}%`;
  document.getElementById("metric-mean").textContent = `${meanScore}%`;
  document.getElementById("results-count").textContent = `${t("total_candidates")}: ${total}`;

  // Render Candidate Cards
  sorted.forEach((c, index) => {
    const isQual = c.status === "QUALIFIED";
    const displayName = blindScreening ? `CANDIDATE-${c.id}` : c.name;
    const reqExp = parseFloat(document.getElementById("required-exp-input").value) || 2.0;

    const card = document.createElement("div");
    card.className = "card-enterprise p-4 space-y-3";

    card.innerHTML = `
      <div class="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-100">
        <div>
          <span class="text-xs font-mono text-slate-400 font-semibold mr-2">#${index + 1}</span>
          <span class="text-xs font-bold text-slate-900 tracking-wide uppercase">${displayName}</span>
          <span class="text-xs text-slate-500 ml-2">(${c.role || "Applicant"})</span>
        </div>
        <div class="flex items-center space-x-3">
          <span class="text-xs font-bold ${isQual ? "text-emerald-700" : "text-rose-700"}">
            ${c.suitability_score.toFixed(1)}%
          </span>
          <span class="${isQual ? "badge-qualified" : "badge-disqualified"}">
            ${isQual ? t("status_qualified") : t("status_disqualified")}
          </span>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-slate-600">
        <div>
          <strong>${t("exp_label")}:</strong> ${c.exp_years} ${t("years")} (${t("req_label")}: ${reqExp} ${t("years")})
        </div>
        <div>
          <strong>${t("comp_match")}:</strong> ${c.verified_skills.length} / ${requiredSkills.length}
        </div>
        <div class="flex justify-end">
          <button class="text-[11px] font-semibold text-slate-800 hover:underline uppercase inspect-btn" data-id="${c.id}">
            ${t("details_btn")} &rarr;
          </button>
        </div>
      </div>

      <!-- Skill Tag Lists -->
      <div class="pt-2 text-[11px] space-y-1">
        <div>
          <strong class="text-slate-700 mr-1">${t("sec_verified")}:</strong>
          ${c.verified_skills.length > 0 
            ? c.verified_skills.map(s => `<span class="tag-verified">${s}</span>`).join("")
            : `<span class="text-slate-400 italic">${t("no_verified")}</span>`}
        </div>

        ${c.missing_skills.length > 0 ? `
          <div>
            <strong class="text-slate-700 mr-1">${t("sec_missing")}:</strong>
            ${c.missing_skills.map(s => `<span class="tag-missing">${s}</span>`).join("")}
          </div>
        ` : ""}

        ${c.negated_skills.length > 0 ? `
          <div>
            <strong class="text-rose-700 mr-1">${t("sec_negated")}:</strong>
            ${c.negated_skills.map(s => `<span class="tag-negated">${s}</span>`).join("")}
          </div>
        ` : ""}
      </div>
    `;

    grid.appendChild(card);
  });

  // Attach Modal event listeners
  document.querySelectorAll(".inspect-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-id");
      openEvidenceModal(id);
    });
  });
}

// Modal handling
function setupModal() {
  const modal = document.getElementById("evidence-modal");
  const closeBtn = document.getElementById("modal-close-btn");
  closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });
}

function openEvidenceModal(candidateId) {
  const cand = candidateDataset.find(c => c.id === candidateId);
  if (!cand) return;

  const modal = document.getElementById("evidence-modal");
  const displayName = blindScreening ? `CANDIDATE-${cand.id}` : cand.name;

  document.getElementById("modal-cand-title").textContent = displayName;
  document.getElementById("modal-cand-subtitle").textContent = `${cand.role} • ${cand.exp_years} ${t("years")} • ${cand.status === "QUALIFIED" ? t("status_qualified") : t("status_disqualified")}`;

  const evidenceList = document.getElementById("modal-evidence-list");
  evidenceList.innerHTML = "";

  const breakdown = cand.evidence_breakdown || {};
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    evidenceList.innerHTML = `<p class="text-slate-400 italic">${t("no_context_spans")}</p>`;
  } else {
    entries.forEach(([skill, quote]) => {
      const isNegated = quote.includes("[DISQUALIFIED");
      const item = document.createElement("div");
      item.className = `p-2.5 border rounded ${isNegated ? "border-rose-200 bg-rose-50" : "border-slate-200 bg-slate-50"}`;
      item.innerHTML = `
        <div class="font-bold text-xs ${isNegated ? "text-rose-800" : "text-slate-900"} mb-1">${skill}</div>
        <div class="font-mono text-[11px] ${isNegated ? "text-rose-700" : "text-slate-700"}">"${quote}"</div>
      `;
      evidenceList.appendChild(item);
    });
  }

  const resumeTextEl = document.getElementById("modal-resume-text");
  resumeTextEl.textContent = blindScreening ? (cand.masked_resume_text || cand.resume_text) : cand.resume_text;

  modal.classList.remove("hidden");
}

// Form Upload Handler
function setupUploadForm() {
  const form = document.getElementById("ingest-form");
  const alertBox = document.getElementById("ingest-alert");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("cand-name-input").value.trim();
    const exp = parseFloat(document.getElementById("cand-exp-input").value) || 1.0;
    const fileInput = document.getElementById("resume-file-input");
    const rawText = document.getElementById("resume-text-input").value.trim();

    let resumeContent = rawText;

    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      if (file.name.endsWith(".txt")) {
        resumeContent = await file.text();
      } else {
        resumeContent = `${file.name} - Sənədin təhlili icra olunur...`;
      }
    }

    if (!resumeContent) {
      alert("Zəhmət olmasa CV mətni daxil edin və ya fayl seçin.");
      return;
    }

    const newId = `C-${candidateDataset.length + 101}`;
    const newCand = {
      id: newId,
      name: name,
      role: "Applicant",
      exp_years: exp,
      resume_text: resumeContent,
      verified_skills: [],
      missing_skills: [],
      negated_skills: [],
      skill_match_ratio: 0.5,
      semantic_similarity: 0.6,
      exp_ratio: 1.0,
      suitability_score: 50.0,
      status: "NOT_QUALIFIED",
      evidence_breakdown: {},
      masked_resume_text: resumeContent
    };

    candidateDataset.unshift(newCand);
    runWhatIfSimulation();

    alertBox.textContent = `${name} (${newId}) ${t("ingest_success")}`;
    alertBox.className = "p-3 border border-emerald-300 bg-emerald-50 text-emerald-800 text-xs rounded block";
    form.reset();
  });
}
