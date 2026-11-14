const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

export async function apiRequest(path, { method = "GET", body, token, headers = {} } = {}) {
  const requestHeaders = { Accept: "application/json", ...headers };
  if (body !== undefined) requestHeaders["Content-Type"] = "application/json";
  if (token) requestHeaders.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: requestHeaders,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    throw new ApiError(
      "Unable to connect to the service. Please check your internet connection or try again in a moment.",
      0,
      null
    );
  }

  if (response.status === 204) return null;

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    let message = data && typeof data === "object" ? data.detail : null;
    if (Array.isArray(message)) {
      message = message
        .map((item) => {
          if (typeof item === "string") return item;
          const field = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
          return field ? `${field}: ${item.msg}` : item.msg || JSON.stringify(item);
        })
        .join("; ");
    }
    throw new ApiError(message || `Request could not be completed (${response.status})`, response.status, data);
  }

  return data;
}

export const authApi = {
  register: (email, password) =>
    apiRequest("/api/v1/auth/register", { method: "POST", body: { email, password } }),
  login: (email, password) =>
    apiRequest("/api/v1/auth/login", { method: "POST", body: { email, password } }),
  me: (token) => apiRequest("/api/v1/auth/me", { token }),
};

export const goalsApi = {
  list: (token) => apiRequest("/api/v1/goals/", { token }),
  get: (token, id) => apiRequest(`/api/v1/goals/${id}`, { token }),
  create: (token, goal) => apiRequest("/api/v1/goals/", { method: "POST", token, body: goal }),
  update: (token, id, changes) =>
    apiRequest(`/api/v1/goals/${id}`, { method: "PATCH", token, body: changes }),
  remove: (token, id) => apiRequest(`/api/v1/goals/${id}`, { method: "DELETE", token }),
};

export const riskApi = {
  getMine: (token) => apiRequest("/api/v1/risk-profiles/me", { token }),
  create: (token, profile) =>
    apiRequest("/api/v1/risk-profiles/", { method: "POST", token, body: profile }),
};

export const holdingsApi = {
  list: (token) => apiRequest("/api/v1/holdings/", { token }),
  create: (token, holding) =>
    apiRequest("/api/v1/holdings/", { method: "POST", token, body: holding }),
  update: (token, id, changes) =>
    apiRequest(`/api/v1/holdings/${id}`, { method: "PUT", token, body: changes }),
  remove: (token, id) =>
    apiRequest(`/api/v1/holdings/${id}`, { method: "DELETE", token }),
};

export const portfoliosApi = {
  optimize: (token, { investment_amount, optimization_mode = "max_sharpe" }) =>
    apiRequest("/api/v1/portfolios/optimize", {
      method: "POST",
      token,
      body: {
        investment_amount: String(investment_amount),
        optimization_mode,
      },
    }),
  list: (token) => apiRequest("/api/v1/portfolios/", { token }),
  get: (token, id) => apiRequest(`/api/v1/portfolios/${id}`, { token }),
  getAllocations: (token, id) =>
    apiRequest(`/api/v1/portfolios/${id}/allocations`, { token }),
};
