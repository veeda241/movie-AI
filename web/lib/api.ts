export type Mode = "image" | "video" | "movie";

export type User = {
  id: string;
  email: string;
  name: string;
  plan: string;
  credit_balance: number;
};

export type Project = {
  id: string;
  name: string;
  description: string;
  owner_id: string;
  org_id: string | null;
  created_at: string;
  updated_at: string;
};

export type Asset = {
  id: string;
  project_id: string;
  kind: "image" | "video" | "packet";
  prompt: string;
  mime_type: string;
  created_at: string;
  file_url: string | null;
  meta?: Record<string, unknown>;
};

export type Job = {
  id: string;
  project_id: string;
  kind: Mode | "assemble";
  status: "queued" | "running" | "succeeded" | "failed";
  prompt: string;
  model: string;
  credits_charged: number;
  result_asset_ids: string[];
  error: string;
  events: string[];
  created_at: string;
  updated_at: string;
};

export type Plan = {
  id: string;
  name: string;
  price_monthly: number;
  credits: number;
  seats: number;
  features: string[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("mf_token");
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) localStorage.setItem("mf_token", token);
  else localStorage.removeItem("mf_token");
}

export function assetFileUrl(assetId: string): string {
  const token = getToken();
  return `${API_URL}/assets/${assetId}/file?token=${encodeURIComponent(token || "")}`;
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {});
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const client = {
  register: (email: string, password: string, name: string) =>
    api<{ access_token: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, name }),
    }),
  login: (email: string, password: string) =>
    api<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => api<User>("/auth/me"),
  projects: {
    list: () => api<Project[]>("/projects"),
    create: (name: string, description = "") =>
      api<Project>("/projects", { method: "POST", body: JSON.stringify({ name, description }) }),
    get: (id: string) => api<Project>(`/projects/${id}`),
  },
  assets: {
    list: (projectId?: string, kind?: string) => {
      const q = new URLSearchParams();
      if (projectId) q.set("project_id", projectId);
      if (kind) q.set("kind", kind);
      const qs = q.toString();
      return api<Asset[]>(`/assets${qs ? `?${qs}` : ""}`);
    },
  },
  generate: {
    image: (project_id: string, prompt: string, model: string) =>
      api<Job>("/generate/image", {
        method: "POST",
        body: JSON.stringify({ project_id, prompt, model }),
      }),
    video: (project_id: string, prompt: string, model: string) =>
      api<Job>("/generate/video", {
        method: "POST",
        body: JSON.stringify({ project_id, prompt, model }),
      }),
    movie: (project_id: string, prompt: string, model: string) =>
      api<Job>("/generate/movie", {
        method: "POST",
        body: JSON.stringify({ project_id, prompt, model }),
      }),
    assemble: (project_id: string, asset_ids: string[], title: string) =>
      api<Job>("/generate/assemble", {
        method: "POST",
        body: JSON.stringify({ project_id, asset_ids, title }),
      }),
  },
  jobs: {
    get: (id: string) => api<Job>(`/jobs/${id}`),
  },
  billing: {
    plans: () => api<Plan[]>("/billing/plans"),
    checkout: (payload: { plan?: string; credit_pack?: number }) =>
      api<{ mode: string; url?: string; status?: string; credits?: number }>("/billing/checkout", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    portal: () => api<{ mode: string; url: string }>("/billing/portal", { method: "POST" }),
  },
  orgs: {
    list: () => api<Array<{ id: string; name: string; plan: string; seat_limit: number; member_count: number }>>("/orgs"),
    create: (name: string) =>
      api<{ id: string; name: string }>("/orgs", { method: "POST", body: JSON.stringify({ name }) }),
    invite: (orgId: string, email: string) =>
      api<{ token: string; email: string }>(`/orgs/${orgId}/invite`, {
        method: "POST",
        body: JSON.stringify({ email }),
      }),
    members: (orgId: string) =>
      api<Array<{ user_id: string; email: string; name: string; role: string }>>(`/orgs/${orgId}/members`),
  },
};

export async function pollJob(
  jobId: string,
  onEvent: (job: Job) => void,
  intervalMs = 900
): Promise<Job> {
  for (;;) {
    const job = await client.jobs.get(jobId);
    onEvent(job);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}

export function isClipAsset(asset: Asset): boolean {
  if (asset.kind !== "video") return false;
  const role = asset.meta?.role;
  return role !== "film";
}

export function isFilmAsset(asset: Asset): boolean {
  return asset.kind === "video" && asset.meta?.role === "film";
}

export { API_URL };
