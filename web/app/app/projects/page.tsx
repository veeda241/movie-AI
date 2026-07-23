"use client";

import Link from "next/link";
import { FormEvent, useEffect, useState } from "react";
import { Project, client } from "@/lib/api";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  async function load() {
    setProjects(await client.projects.list());
  }

  useEffect(() => {
    load().catch((err) => setError(err instanceof Error ? err.message : "Failed"));
  }, []);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    await client.projects.create(name.trim());
    setName("");
    await load();
  }

  return (
    <div className="p-8">
      <h1 className="font-display text-4xl text-mist-100">Projects</h1>
      <p className="mt-2 text-mist-400">Collections of images, clips, and scene packets.</p>
      {error && <p className="mt-4 text-red-300">{error}</p>}
      <form onSubmit={onCreate} className="mt-8 flex gap-3">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New project name"
          className="rounded-xl border border-white/10 bg-ink-950 px-4 py-2 outline-none focus:border-tungsten-500"
        />
        <button className="rounded-full bg-tungsten-500 px-5 py-2 text-sm font-medium text-ink-950">Create</button>
      </form>
      <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {projects.map((p) => (
          <Link
            key={p.id}
            href={`/app/projects/${p.id}`}
            className="border border-white/10 bg-ink-900/60 p-5 transition hover:border-tungsten-500/50"
          >
            <h2 className="font-display text-2xl text-mist-100">{p.name}</h2>
            <p className="mt-2 line-clamp-2 text-sm text-mist-400">{p.description || "No description"}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
