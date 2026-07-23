"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { API_URL, Asset, Project, assetFileUrl, client } from "@/lib/api";

export default function ProjectDetailPage() {
  const params = useParams();
  const id = String(params.id);
  const [project, setProject] = useState<Project | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [filter, setFilter] = useState<"all" | "image" | "video">("all");
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      setProject(await client.projects.get(id));
      setAssets(await client.assets.list(id));
    })().catch((err) => setError(err instanceof Error ? err.message : "Failed"));
  }, [id]);

  const visible = assets.filter((a) => {
    if (a.kind === "packet") return false;
    if (filter === "all") return true;
    return a.kind === filter;
  });

  function exportZip() {
    const token = localStorage.getItem("mf_token");
    window.open(`${API_URL}/assets/export/project/${id}?token=${encodeURIComponent(token || "")}`, "_blank");
  }

  return (
    <div className="p-8">
      <Link href="/app/projects" className="text-sm text-tungsten-400">
        ← Projects
      </Link>
      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-4xl text-mist-100">{project?.name || "…"}</h1>
          <p className="mt-2 text-mist-400">{project?.description}</p>
        </div>
        <button onClick={exportZip} className="rounded-full border border-white/20 px-4 py-2 text-sm">
          Export ZIP
        </button>
      </div>
      {error && <p className="mt-4 text-red-300">{error}</p>}
      <div className="mt-6 flex gap-2 text-sm">
        {(["all", "image", "video"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-full px-4 py-1 capitalize ${filter === f ? "bg-tungsten-500 text-ink-950" : "bg-ink-800 text-mist-200"}`}
          >
            {f}
          </button>
        ))}
      </div>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((asset) => (
          <div key={asset.id} className="overflow-hidden border border-white/10 bg-ink-900/50">
            <div className="aspect-video bg-ink-950">
              {asset.kind === "image" ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={assetFileUrl(asset.id)} alt="" className="h-full w-full object-cover transition duration-300 hover:scale-[1.02]" />
              ) : (
                <video src={assetFileUrl(asset.id)} className="h-full w-full object-cover" controls />
              )}
            </div>
            <p className="line-clamp-2 p-3 text-xs text-mist-400">{asset.prompt}</p>
          </div>
        ))}
        {visible.length === 0 && <p className="text-mist-400">No assets yet — generate from Create.</p>}
      </div>
    </div>
  );
}
