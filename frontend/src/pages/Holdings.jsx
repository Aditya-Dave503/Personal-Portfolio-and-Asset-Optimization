import { useEffect, useState, useMemo } from "react";
import { Link } from "react-router-dom";
import Notice from "../components/Notice.jsx";
import { holdingsApi } from "../services/api.js";
import { useAuth } from "../context/AuthContext.jsx";

const ASSET_TYPES = [
  { value: "STOCK", label: "Equities / Stocks", icon: "📈", badgeClass: "badge-stock" },
  { value: "MF", label: "Mutual Funds", icon: "💼", badgeClass: "badge-mf" },
  { value: "GOLD", label: "Sovereign Gold", icon: "🪙", badgeClass: "badge-gold" },
  { value: "SILVER", label: "Silver Commodity", icon: "🥈", badgeClass: "badge-silver" },
  { value: "BOND", label: "Government Bonds", icon: "🏛️", badgeClass: "badge-bond" },
];

const blankHolding = {
  asset_type: "STOCK",
  name: "",
  units: "",
  invested_value: "",
};

function formatMoney(amount) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(amount || 0));
}

export default function Holdings() {
  const { token } = useAuth();
  const [holdings, setHoldings] = useState([]);
  const [form, setForm] = useState(blankHolding);
  const [editingId, setEditingId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [serviceUnavailable, setServiceUnavailable] = useState(false);

  // Filters
  const [filterType, setFilterType] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  async function loadHoldings() {
    setLoading(true);
    setError("");
    setServiceUnavailable(false);
    try {
      const data = await holdingsApi.list(token);
      setHoldings(Array.isArray(data) ? data : []);
    } catch (err) {
      if (err.status === 503) {
        setServiceUnavailable(true);
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadHoldings();
  }, [token]);

  function startEdit(holding) {
    setEditingId(holding.id);
    setForm({
      asset_type: holding.asset_type || "STOCK",
      name: holding.name || "",
      units: holding.units ? String(holding.units) : "",
      invested_value: holding.invested_value ? String(holding.invested_value) : "",
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function cancelEdit() {
    setEditingId(null);
    setForm(blankHolding);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSuccess("");

    const unitsNum = Number(form.units);
    const valueNum = Number(form.invested_value);

    if (!form.name.trim()) {
      setError("Please specify the instrument or asset name.");
      return;
    }
    if (isNaN(unitsNum) || unitsNum <= 0) {
      setError("Units must be a positive number greater than 0.");
      return;
    }
    if (isNaN(valueNum) || valueNum <= 0) {
      setError("Invested value must be a positive amount greater than 0.");
      return;
    }

    setBusy(true);
    try {
      const payload = {
        asset_type: form.asset_type,
        name: form.name.trim(),
        units: unitsNum.toString(),
        invested_value: valueNum.toFixed(2),
      };

      if (editingId) {
        await holdingsApi.update(token, editingId, payload);
        setSuccess(`Holding "${form.name}" was successfully updated.`);
      } else {
        await holdingsApi.create(token, payload);
        setSuccess(`Holding "${form.name}" has been recorded.`);
      }
      setForm(blankHolding);
      setEditingId(null);
      await loadHoldings();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(id, name) {
    if (!window.confirm(`Are you sure you want to remove "${name}" from your recorded holdings?`)) {
      return;
    }
    setError("");
    setSuccess("");
    try {
      await holdingsApi.remove(token, id);
      setSuccess(`Holding "${name}" was removed.`);
      await loadHoldings();
    } catch (err) {
      setError(err.message);
    }
  }

  // Derived calculations
  const totalInvested = useMemo(
    () => holdings.reduce((sum, h) => sum + Number(h.invested_value || 0), 0),
    [holdings]
  );

  const assetTypesPresent = useMemo(
    () => new Set(holdings.map((h) => h.asset_type)).size,
    [holdings]
  );

  const filteredHoldings = useMemo(() => {
    return holdings.filter((item) => {
      const matchesType = filterType === "all" || item.asset_type === filterType;
      const matchesSearch =
        !searchQuery.trim() ||
        (item.name && item.name.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (item.asset_type && item.asset_type.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesType && matchesSearch;
    });
  }, [holdings, filterType, searchQuery]);

  return (
    <div className="section-wrap inner-page holdings-page">
      <div className="page-heading">
        <span className="eyebrow">PORTFOLIO INVENTORY</span>
        <h1>
          Your recorded <em>holdings.</em>
        </h1>
        <p>
          Track your active investments across equities, mutual funds, gold, bonds, and commodities.
          Your holdings form the real-world baseline compared against ML-optimized allocations.
        </p>
      </div>

      <Notice type="error" onClose={() => setError("")}>
        {error}
      </Notice>
      <Notice type="success" onClose={() => setSuccess("")}>
        {success}
      </Notice>

      {serviceUnavailable && (
        <div className="warning-banner-card">
          <span className="warning-icon">ℹ</span>
          <div>
            <strong>Holdings Service Integration in Progress:</strong> The backend holdings service
            is currently responding with HTTP 503 as service endpoints are being connected with the database.
            You can view the interface and forms below.
          </div>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="stats-grid holdings-stats-grid">
        <article className="stat-card stat-primary">
          <span className="stat-label">TOTAL INVESTED</span>
          <strong>{formatMoney(totalInvested)}</strong>
          <span className="stat-caption">Sum across all recorded holdings</span>
          <span className="stat-watermark">₹</span>
        </article>

        <article className="stat-card">
          <span className="stat-label">TOTAL HOLDINGS</span>
          <strong>{holdings.length.toString().padStart(2, "0")}</strong>
          <span className="stat-caption">Individual instruments tracked</span>
          <span className="stat-watermark">◎</span>
        </article>

        <article className="stat-card">
          <span className="stat-label">ASSET CLASSES</span>
          <strong>{assetTypesPresent.toString().padStart(2, "0")}</strong>
          <span className="stat-caption">Active asset categories</span>
          <span className="stat-watermark">⌁</span>
        </article>

        <article className="stat-card">
          <span className="stat-label">OPTIMIZE PORTFOLIO</span>
          <strong className="stat-word">Ready</strong>
          <span className="stat-caption">
            <Link to="/portfolio" className="small-text-link">
              Launch ML Optimizer →
            </Link>
          </span>
          <span className="stat-watermark">↗</span>
        </article>
      </div>

      <div className="goals-layout holdings-layout">
        {/* Left Column: Add / Edit Holding Form */}
        <div className="content-card goal-form-card">
          <span className="eyebrow">{editingId ? "UPDATE ASSET" : "RECORD ASSET"}</span>
          <h2>{editingId ? "Edit Holding" : "Add New Holding"}</h2>
          <p className="muted">
            {editingId
              ? "Modify the quantity, asset type, or purchase value."
              : "Enter details of any stock, fund, gold, bond, or asset you own."}
          </p>

          <form className="form-stack" onSubmit={handleSubmit}>
            <label>
              Asset Category
              <div className="select-wrap">
                <select
                  value={form.asset_type}
                  onChange={(e) => setForm({ ...form, asset_type: e.target.value })}
                  required
                >
                  {ASSET_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.icon} {t.label} ({t.value})
                    </option>
                  ))}
                </select>
              </div>
            </label>

            <label>
              Instrument / Asset Name
              <input
                type="text"
                placeholder="e.g. Nifty 50 ETF, HDFC Top 100, SGB 2028"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                required
                maxLength={120}
              />
            </label>

            <label>
              Units / Quantity
              <input
                type="number"
                step="any"
                min="0.000001"
                placeholder="e.g. 10 or 4.5"
                value={form.units}
                onChange={(e) => setForm({ ...form, units: e.target.value })}
                required
              />
            </label>

            <label>
              Total Invested Value
              <div className="input-with-prefix">
                <span>₹</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  placeholder="e.g. 50000"
                  value={form.invested_value}
                  onChange={(e) => setForm({ ...form, invested_value: e.target.value })}
                  required
                />
              </div>
            </label>

            <div className="form-actions-row">
              <button className="button button-full" type="submit" disabled={busy}>
                {busy
                  ? "Saving…"
                  : editingId
                  ? "Update Holding"
                  : "+ Record Holding"} <span>↗</span>
              </button>
              {editingId && (
                <button
                  type="button"
                  className="button button-quiet button-full"
                  onClick={cancelEdit}
                >
                  Cancel Edit
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Right Column: Holdings List and Filters */}
        <section className="goal-list-section">
          <div className="goals-controls-card">
            <div className="filter-controls-row">
              <div className="search-box-wrap">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  placeholder="Filter by asset name or ticker…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              <div className="category-filter-pills">
                <button
                  type="button"
                  className={`filter-pill ${filterType === "all" ? "filter-pill-active" : ""}`}
                  onClick={() => setFilterType("all")}
                >
                  All ({holdings.length})
                </button>
                {ASSET_TYPES.map((t) => {
                  const count = holdings.filter((h) => h.asset_type === t.value).length;
                  return (
                    <button
                      type="button"
                      key={t.value}
                      className={`filter-pill ${filterType === t.value ? "filter-pill-active" : ""}`}
                      onClick={() => setFilterType(t.value)}
                    >
                      {t.icon} {t.value} ({count})
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {loading ? (
            <div className="loading-card">
              <span className="spinner" /> Loading your holdings inventory…
            </div>
          ) : filteredHoldings.length > 0 ? (
            <div className="holdings-table-wrap">
              <table className="holdings-table">
                <thead>
                  <tr>
                    <th>Asset &amp; Instrument</th>
                    <th>Category</th>
                    <th>Units</th>
                    <th>Total Invested</th>
                    <th>Avg Price/Unit</th>
                    <th style={{ textAlign: "right" }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredHoldings.map((h) => {
                    const typeMeta = ASSET_TYPES.find((t) => t.value === h.asset_type) || {
                      label: h.asset_type,
                      icon: "•",
                      badgeClass: "badge-default",
                    };
                    const unitsVal = Number(h.units || 0);
                    const investedVal = Number(h.invested_value || 0);
                    const avgPrice = unitsVal > 0 ? investedVal / unitsVal : 0;

                    return (
                      <tr key={h.id} className={editingId === h.id ? "row-highlight" : ""}>
                        <td>
                          <div className="holding-name-cell">
                            <span className="holding-icon">{typeMeta.icon}</span>
                            <div>
                              <strong>{h.name}</strong>
                              <small className="tiny-muted">ID #{h.id}</small>
                            </div>
                          </div>
                        </td>
                        <td>
                          <span className={`asset-badge ${typeMeta.badgeClass}`}>
                            {h.asset_type}
                          </span>
                        </td>
                        <td>
                          <strong>{unitsVal.toLocaleString("en-IN", { maximumFractionDigits: 4 })}</strong>
                        </td>
                        <td>
                          <strong className="holding-val-text">{formatMoney(investedVal)}</strong>
                        </td>
                        <td>
                          <span className="tiny-muted">{formatMoney(avgPrice)}</span>
                        </td>
                        <td style={{ textAlign: "right" }}>
                          <div className="action-btns-cell">
                            <button
                              type="button"
                              className="edit-goal-btn"
                              title="Edit this holding"
                              onClick={() => startEdit(h)}
                            >
                              ✎
                            </button>
                            <button
                              type="button"
                              className="delete-goal-btn"
                              title="Delete this holding"
                              onClick={() => handleDelete(h.id, h.name)}
                            >
                              ✕
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">📊</span>
              <h3>No holdings found</h3>
              <p>
                {searchQuery || filterType !== "all"
                  ? "No holdings matched your filter criteria."
                  : "You haven't recorded any holdings yet. Add your current mutual funds, stocks, gold, or bonds using the form."}
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
