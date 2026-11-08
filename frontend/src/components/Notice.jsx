export default function Notice({ type = "info", children, onClose }) {
  if (!children) return null;
  return <div className={`notice notice-${type}`} role={type === "error" ? "alert" : "status"}>
    <span>{children}</span>
    {onClose && <button className="notice-close" onClick={onClose} aria-label="Dismiss message">×</button>}
  </div>;
}