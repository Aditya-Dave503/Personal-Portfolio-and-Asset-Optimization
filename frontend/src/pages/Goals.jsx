import { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";
import Notice from "../components/Notice.jsx";
import { goalsApi } from "../services/api.js";
import { useAuth } from "../context/AuthContext.jsx";

const CATEGORIES = [
  { value: "retirement", label: "Retirement", icon: "◷", desc: "Long-term nest egg" },
  { value: "house_down_payment", label: "Home Down Payment", icon: "⌂", desc: "Property acquisition" },
  { value: "higher_education", label: "Higher Education", icon: "🎓", desc: "Tuition & academic funding" },
  { value: "other", label: "Other Milestone", icon: "↗", desc: "Personal life goal" },
];

const blankForm = {
  name: "",
  category: "retirement",
  target_amount: "",
  target_date: "",
};

function formatMoney(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(amount || 0));
}

function getCountdown(targetDateStr) {
  if (!targetDateStr) return { text: "", isUrgent: false };
  const now = new Date();
  const target = new Date(`${targetDateStr}T00:00:00`);
  const diffMs = target.getTime() - now.getTime();
  const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0) return { text: "Target date reached", isUrgent: true };
  if (diffDays <= 90) return { text: `${diffDays} days left`, isUrgent: true };

  const months = Math.floor(diffDays / 30.4375);
  const years = Math.floor(months / 12);
  const remMonths = months % 12;

  if (years > 0 && remMonths > 0) {
    return { text: `${years}y ${remMonths}m runway`, isUrgent: false };
  }
  if (years > 0) {
    return { text: `${years} year${years === 1 ? "" : "s"} runway`, isUrgent: false };
  }
  return { text: `${months} month${months === 1 ? "" : "s"} runway`, isUrgent: false };
}

export default function Goals() {
  const { token } = useAuth();
  const [goals, setGoals] = useState([]);
  const [form, setForm] = useState(blankForm);
  const [editingId, setEditingId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Filters and sorting
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("date_asc");

  const todayStr = useMemo(() => new Date().toISOString().slice(0, 10), []);

  async function loadGoals() {
    try {
      const result = await goalsApi.list(token);
      setGoals(Array.isArray(result) ? result : []);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    let active = true;
    setLoading(true);
    goalsApi
      .list(token)
      .then((result) => {
        if (active) setGoals(Array.isArray(result) ? result : []);
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token]);

  function startEdit(goal) {
    setEditingId(goal.id);
    setForm({
      name: goal.name,
      category: goal.category,
      target_amount: String(goal.target_amount),
      target_date: goal.target_date,
    });
    setError("");
    setSuccess("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function resetForm() {
    setEditingId(null);
    setForm(blankForm);
    setError("");
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSuccess("");
    setBusy(true);

    const payload = {
      name: form.name.trim(),
      category: form.category,
      target_amount: String(form.target_amount),
      target_date: form.target_date,
    };

    if (payload.target_date < todayStr) {
      setError("Target date cannot be in the past.");
      setBusy(false);
      return;
    }

    try {
      if (editingId !== null) {
        await goalsApi.update(token, editingId, payload);
        setSuccess(`Goal "${payload.name}" updated successfully.`);
      } else {
        await goalsApi.create(token, payload);
        setSuccess(`Goal "${payload.name}" added to your plan.`);
      }
      resetForm();
      await loadGoals();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function removeGoal(goal) {
    if (!window.confirm(`Are you sure you want to delete "${goal.name}"?`)) {
      return;
    }
    setError("");
    setSuccess("");
    try {
      await goalsApi.remove(token, goal.id);
      if (editingId === goal.id) resetForm();
      await loadGoals();
      setSuccess(`Goal "${goal.name}" removed.`);
    } catch (e) {
      setError(e.message);
    }
  }

  // Filtered and sorted goals list
  const filteredGoals = useMemo(() => {
    let list = [...goals];

    if (selectedCategory !== "all") {
      list = list.filter((g) => g.category === selectedCategory);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter((g) => g.name.toLowerCase().includes(q));
    }

    list.sort((a, b) => {
      if (sortBy === "date_asc") return new Date(a.target_date) - new Date(b.target_date);
      if (sortBy === "date_desc") return new Date(b.target_date) - new Date(a.target_date);
      if (sortBy === "amount_desc") return Number(b.target_amount) - Number(a.target_amount);
      if (sortBy === "amount_asc") return Number(a.target_amount) - Number(b.target_amount);
      if (sortBy === "name_asc") return a.name.localeCompare(b.name);
      return 0;
    });

    return list;
  }, [goals, selectedCategory, searchQuery, sortBy]);

  const totalAllTarget = goals.reduce((s, g) => s + Number(g.target_amount || 0), 0);
  const totalFilteredTarget = filteredGoals.reduce((s, g) => s + Number(g.target_amount || 0), 0);

  return (
    <div className="section-wrap inner-page goals-page">
      <div className="page-heading goals-heading">
        <div>
          <span className="eyebrow">SAVINGS &amp; INVESTMENT TARGETS</span>
          <h1>
            Financial <em>goals.</em>
          </h1>
          <p>
            Give the milestones that matter a name, a target amount, and a target date.
            Organize your future one milestone at a time.
          </p>
        </div>
        <div className="goal-total">
          <span>TOTAL PLANNED TARGET</span>
          <strong>{formatMoney(totalAllTarget)}</strong>
          <small>
            Across {goals.length} active milestone{goals.length === 1 ? "" : "s"}
          </small>
        </div>
      </div>

      <Notice type="error" onClose={() => setError("")}>
        {error}
      </Notice>
      <Notice type="success" onClose={() => setSuccess("")}>
        {success}
      </Notice>

      <div className="goals-layout">
        {/* Left Form Card */}
        <section className={`content-card goal-form-card ${editingId !== null ? "form-editing-mode" : ""}`}>
          <div className="form-card-header">
            <span className="eyebrow">
              {editingId !== null ? "UPDATE MILESTONE" : "NEW MILESTONE"}
            </span>
            {editingId !== null && <span className="editing-indicator-badge">Editing Mode</span>}
          </div>
          <h2>{editingId !== null ? "Edit your goal" : "Add a financial goal"}</h2>
          <p className="muted">
            {editingId !== null
              ? "Update the details of your milestone below and save changes."
              : "Set a clear financial milestone to guide your savings and portfolio allocation."}
          </p>

          <form className="form-stack" onSubmit={submit}>
            <label>
              Goal Name
              <span className="label-help">Give your goal a descriptive name</span>
              <input
                value={form.name}
                maxLength={100}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="e.g. Dream Home Downpayment"
                required
              />
            </label>

            <label>
              Category
              <div className="select-wrap">
                <select
                  value={form.category}
                  onChange={(e) => setForm({ ...form, category: e.target.value })}
                  required
                >
                  {CATEGORIES.map((c) => (
                    <option value={c.value} key={c.value}>
                      {c.label} ({c.desc})
                    </option>
                  ))}
                </select>
              </div>
            </label>

            <label>
              Target Amount
              <span className="label-help">How much do you need to reach this goal?</span>
              <div className="input-with-prefix">
                <span>₹</span>
                <input
                  type="number"
                  min="0.01"
                  step="0.01"
                  max="9999999999.99"
                  value={form.target_amount}
                  onChange={(e) => setForm({ ...form, target_amount: e.target.value })}
                  placeholder="e.g. 2500000"
                  required
                />
              </div>
            </label>

            <label>
              Target Date
              <span className="label-help">When do you want to accomplish this goal?</span>
              <input
                type="date"
                min={todayStr}
                value={form.target_date}
                onChange={(e) => setForm({ ...form, target_date: e.target.value })}
                required
              />
            </label>

            <button className="button button-full" disabled={busy} type="submit">
              {busy ? "Saving…" : editingId !== null ? "Save Changes" : "Add Goal"} <span>↗</span>
            </button>

            {editingId !== null && (
              <button type="button" className="button button-quiet button-full" onClick={resetForm}>
                Cancel Editing
              </button>
            )}
          </form>
        </section>

        {/* Right Goals List Section */}
        <section className="goal-list-section">
          {/* Controls: Search, Category Filter, and Sort */}
          <div className="goals-controls-card content-card">
            <div className="filter-pills-row">
              <button
                type="button"
                className={`filter-pill ${selectedCategory === "all" ? "filter-pill-active" : ""}`}
                onClick={() => setSelectedCategory("all")}
              >
                All Goals ({goals.length})
              </button>
              {CATEGORIES.map((c) => {
                const count = goals.filter((g) => g.category === c.value).length;
                return (
                  <button
                    key={c.value}
                    type="button"
                    className={`filter-pill ${selectedCategory === c.value ? "filter-pill-active" : ""}`}
                    onClick={() => setSelectedCategory(c.value)}
                  >
                    {c.icon} {c.label} ({count})
                  </button>
                );
              })}
            </div>

            <div className="search-and-sort-row">
              <div className="search-box">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  placeholder="Search goals by name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <button
                    type="button"
                    className="clear-search-btn"
                    onClick={() => setSearchQuery("")}
                    title="Clear search"
                  >
                    ×
                  </button>
                )}
              </div>

              <div className="sort-box">
                <label htmlFor="goal-sort-select">Sort by:</label>
                <select
                  id="goal-sort-select"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <option value="date_asc">Target Date (Soonest first)</option>
                  <option value="date_desc">Target Date (Farthest first)</option>
                  <option value="amount_desc">Target Amount (High to Low)</option>
                  <option value="amount_asc">Target Amount (Low to High)</option>
                  <option value="name_asc">Name (A-Z)</option>
                </select>
              </div>
            </div>

            {(selectedCategory !== "all" || searchQuery) && (
              <div className="filter-summary-row">
                <span>
                  Showing <strong>{filteredGoals.length}</strong> of <strong>{goals.length}</strong> goals
                  {selectedCategory !== "all" && ` in ${CATEGORIES.find((c) => c.value === selectedCategory)?.label}`}
                  {searchQuery && ` matching "${searchQuery}"`} (Subtotal: {formatMoney(totalFilteredTarget)})
                </span>
                <button
                  type="button"
                  className="reset-filters-btn"
                  onClick={() => {
                    setSelectedCategory("all");
                    setSearchQuery("");
                  }}
                >
                  Reset filters
                </button>
              </div>
            )}
          </div>

          {/* Goal List Display */}
          {loading ? (
            <div className="loading-card content-card">
              <span className="spinner" /> Loading your goals…
            </div>
          ) : filteredGoals.length === 0 ? (
            <div className="empty-state content-card">
              <span className="empty-icon">◎</span>
              <h3>No goals found</h3>
              <p>
                {goals.length === 0
                  ? "You haven't added any goals yet. Use the form on the left to create your first milestone."
                  : "No goals match your search or category filter."}
              </p>
              {goals.length > 0 && (
                <button
                  type="button"
                  className="button button-small button-quiet"
                  onClick={() => {
                    setSelectedCategory("all");
                    setSearchQuery("");
                  }}
                >
                  Show all goals
                </button>
              )}
            </div>
          ) : (
            <div className="goal-list">
              {filteredGoals.map((goal) => {
                const countdown = getCountdown(goal.target_date);
                const categoryObj = CATEGORIES.find((c) => c.value === goal.category);
                const isEditing = editingId === goal.id;

                return (
                  <article
                    className={`goal-item content-card ${isEditing ? "goal-card-active" : ""}`}
                    key={goal.id}
                  >
                    <div className="goal-item-top">
                      <span className={`goal-avatar goal-avatar-${goal.category}`}>
                        {categoryObj?.icon || "↗"}
                      </span>
                      <span className="category-label">
                        {categoryObj?.label || goal.category}
                      </span>
                      <div className="goal-menu">
                        <button
                          type="button"
                          className="edit-goal-btn"
                          aria-label={`Edit ${goal.name}`}
                          onClick={() => startEdit(goal)}
                          title="Edit goal"
                        >
                          ✎
                        </button>
                        <button
                          type="button"
                          className="delete-goal-btn"
                          aria-label={`Delete ${goal.name}`}
                          onClick={() => removeGoal(goal)}
                          title="Delete goal"
                        >
                          ×
                        </button>
                      </div>
                    </div>

                    <h3 title={goal.name}>{goal.name}</h3>
                    <strong className="goal-item-amount">{formatMoney(goal.target_amount)}</strong>

                    <div className="goal-item-foot">
                      <div className="target-date-box">
                        <span className="target-label">Target Date</span>
                        <time dateTime={goal.target_date}>
                          {new Date(`${goal.target_date}T00:00:00`).toLocaleDateString("en-IN", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                          })}
                        </time>
                      </div>
                      <span className={`countdown-badge ${countdown.isUrgent ? "countdown-urgent" : ""}`}>
                        {countdown.text}
                      </span>
                    </div>

                    <div className="goal-action-row" style={{ marginTop: "14px", paddingTop: "12px", borderTop: "1px dashed var(--line)" }}>
                      <Link
                        to={`/portfolio?amount=${goal.target_amount}&goalName=${encodeURIComponent(goal.name)}`}
                        className="button button-small button-quiet"
                        style={{ width: "100%", justifyContent: "center", fontSize: "11px", padding: "8px 10px" }}
                        title="Run ML optimizer targeting this goal amount"
                      >
                        Optimize Portfolio <span>↗</span>
                      </Link>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
