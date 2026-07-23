"use client";

import { motion } from "framer-motion";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  Asset,
  Mode,
  Project,
  assetFileUrl,
  client,
  isClipAsset,
  isFilmAsset,
  pollJob,
} from "@/lib/api";

const MODELS: Record<Mode, string[]> = {
  image: ["sdxl-local", "imagen-style"],
  video: ["wan-2.2", "motif-local"],
  movie: ["scene-clip", "multi-agent"],
};

export default function StudioPage() {
  const [mode, setMode] = useState<Mode>("video");
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState(MODELS.video[0]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [events, setEvents] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<Asset | null>(null);
  const [credits, setCredits] = useState<number | null>(null);
  const [clips, setClips] = useState<Asset[]>([]);
  const [films, setFilms] = useState<Asset[]>([]);
  const [filmTitle, setFilmTitle] = useState("My Film");

  const refreshTimeline = useCallback(async (pid: string) => {
    const assets = await client.assets.list(pid, "video");
    const clipList = assets.filter(isClipAsset).sort((a, b) => {
      const ai = Number(a.meta?.clip_index ?? a.meta?.scene ?? 0);
      const bi = Number(b.meta?.clip_index ?? b.meta?.scene ?? 0);
      if (ai && bi) return ai - bi;
      return a.created_at.localeCompare(b.created_at);
    });
    setClips(clipList);
    setFilms(assets.filter(isFilmAsset));
  }, []);

  useEffect(() => {
    (async () => {
      const me = await client.me();
      setCredits(me.credit_balance);
      let list = await client.projects.list();
      if (list.length === 0) {
        const created = await client.projects.create("Untitled Project");
        list = [created];
      }
      setProjects(list);
      setProjectId(list[0].id);
      await refreshTimeline(list[0].id);
    })().catch((err) => setError(err instanceof Error ? err.message : "Failed to load"));
  }, [refreshTimeline]);

  useEffect(() => {
    setModel(MODELS[mode][0]);
  }, [mode]);

  useEffect(() => {
    if (!projectId) return;
    refreshTimeline(projectId).catch(() => undefined);
  }, [projectId, refreshTimeline]);

  const showTimeline = mode === "movie";

  const costHint = useMemo(() => {
    if (mode === "image") return "1 credit";
    if (mode === "video") return "5 credits · single video";
    if (model === "multi-agent") return "~24 credits (agent scenes + assemble)";
    return `5 credits · scene ${String(clips.length + 1).padStart(2, "0")}`;
  }, [mode, clips.length, model]);

  function moveClip(index: number, direction: -1 | 1) {
    const next = index + direction;
    if (next < 0 || next >= clips.length) return;
    const copy = [...clips];
    const tmp = copy[index];
    copy[index] = copy[next];
    copy[next] = tmp;
    setClips(copy);
  }

  function removeClipFromTimeline(id: string) {
    setClips((prev) => prev.filter((c) => c.id !== id));
  }

  async function onGenerate(e: FormEvent) {
    e.preventDefault();
    if (!projectId || !prompt.trim()) return;
    setBusy(true);
    setError("");
    setEvents(["Submitting..."]);
    try {
      let job;
      if (mode === "image") {
        job = await client.generate.image(projectId, prompt.trim(), model);
      } else if (mode === "video") {
        job = await client.generate.video(projectId, prompt.trim(), model);
      } else if (model === "multi-agent") {
        job = await client.generate.movie(projectId, prompt.trim(), model);
      } else {
        // Movie page: add one scene clip at a time via video generation
        job = await client.generate.video(projectId, prompt.trim(), model === "scene-clip" ? "wan-2.2" : model);
      }

      const final = await pollJob(job.id, (j) => setEvents(j.events));
      if (final.status === "failed") {
        setError(final.error || "Generation failed");
      } else if (final.result_asset_ids.length) {
        await refreshTimeline(projectId);
        const assets = await client.assets.list(projectId);
        if (mode === "video") {
          const video = assets.find((a) => final.result_asset_ids.includes(a.id) && a.kind === "video");
          setPreview(video || null);
        } else {
          const film = assets.find((a) => final.result_asset_ids.includes(a.id) && isFilmAsset(a));
          const clip = assets.find((a) => final.result_asset_ids.includes(a.id) && a.kind === "video");
          const image = assets.find((a) => final.result_asset_ids.includes(a.id) && a.kind === "image");
          setPreview(film || clip || image || null);
          if (mode === "movie" && model !== "multi-agent") {
            setPrompt("");
          }
        }
      }
      const me = await client.me();
      setCredits(me.credit_balance);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setBusy(false);
    }
  }

  async function onAssemble() {
    if (!projectId || clips.length < 2) return;
    setBusy(true);
    setError("");
    setEvents(["Assembling timeline..."]);
    try {
      const job = await client.generate.assemble(
        projectId,
        clips.map((c) => c.id),
        filmTitle.trim() || "My Film"
      );
      const final = await pollJob(job.id, (j) => setEvents(j.events));
      if (final.status === "failed") {
        setError(final.error || "Assemble failed");
      } else if (final.result_asset_ids.length) {
        await refreshTimeline(projectId);
        const assets = await client.assets.list(projectId, "video");
        const film = assets.find((a) => final.result_asset_ids.includes(a.id));
        setPreview(film || null);
      }
      const me = await client.me();
      setCredits(me.credit_balance);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Assemble failed");
    } finally {
      setBusy(false);
    }
  }

  const nextSceneLabel = String(clips.length + 1).padStart(2, "0");

  return (
    <div className="flex h-full min-h-screen flex-col">
      <header className="flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div>
          <h1 className="font-display text-3xl text-mist-100">
            {mode === "movie" ? "Movie" : mode === "video" ? "Video" : "Create"}
          </h1>
          <p className="text-sm text-mist-400">
            {mode === "movie"
              ? "Add scenes one by one, reorder the timeline, then assemble the film"
              : mode === "video"
                ? "Generate a single video from your prompt"
                : "Generate images for your storyboard"}
          </p>
        </div>
        <div className="text-sm text-mist-200">
          {credits !== null && <span className="text-tungsten-400">{credits} credits</span>}
        </div>
      </header>

      <div className="grid flex-1 gap-0 lg:grid-cols-[200px_1fr]">
        <aside className="border-r border-white/10 p-4">
          <p className="mb-3 text-xs uppercase tracking-wider text-mist-400">Mode</p>
          {(["image", "video", "movie"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`mb-2 block w-full rounded-lg px-3 py-2 text-left capitalize transition ${
                mode === m ? "bg-ink-800 text-tungsten-400" : "text-mist-200 hover:bg-ink-800/50"
              }`}
            >
              {m}
            </button>
          ))}
          <p className="mb-2 mt-8 text-xs uppercase tracking-wider text-mist-400">Project</p>
          <select
            className="w-full rounded-lg border border-white/10 bg-ink-950 px-2 py-2 text-sm"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </aside>

        <section className="relative flex flex-col">
          <div className="flex flex-1 items-center justify-center p-6">
            <motion.div
              key={preview?.id || "empty"}
              initial={{ opacity: 0.4, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="relative aspect-video w-full max-w-4xl overflow-hidden rounded-xl border border-white/10 bg-ink-900"
            >
              {!preview && (
                <div className="flex h-full flex-col items-center justify-center text-mist-400">
                  <p className="font-display text-4xl text-mist-200">
                    {mode === "movie" ? "Movie canvas" : "Canvas"}
                  </p>
                  <p className="mt-2 max-w-md text-center text-sm">
                    {mode === "movie"
                      ? `Describe scene ${nextSceneLabel}, generate it, then assemble all scenes into one movie.`
                      : mode === "video"
                        ? "Write a prompt and generate a video clip."
                        : "Write a prompt and generate an image."}
                  </p>
                </div>
              )}
              {preview?.kind === "image" && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={assetFileUrl(preview.id)} alt={preview.prompt} className="h-full w-full object-contain" />
              )}
              {preview?.kind === "video" && (
                <video src={assetFileUrl(preview.id)} controls autoPlay className="h-full w-full object-contain" />
              )}
              {busy && (
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                  <div className="h-1 overflow-hidden rounded bg-white/10">
                    <motion.div
                      className="h-full bg-tungsten-500"
                      initial={{ width: "8%" }}
                      animate={{ width: ["8%", "70%", "90%"] }}
                      transition={{ duration: 8, repeat: Infinity }}
                    />
                  </div>
                  <p className="mt-2 text-xs text-mist-200">{events[events.length - 1]}</p>
                </div>
              )}
            </motion.div>
          </div>

          {showTimeline && (
            <div className="mx-6 mb-3 border border-white/10 bg-ink-950/70 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-wider text-mist-400">
                    {mode === "movie" ? "Scene timeline" : "Timeline"}
                  </p>
                  <p className="text-sm text-mist-200">
                    {clips.length} scene{clips.length === 1 ? "" : "s"}
                    {films.length > 0
                      ? ` · ${films.length} assembled film${films.length === 1 ? "" : "s"}`
                      : ""}
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <input
                    value={filmTitle}
                    onChange={(e) => setFilmTitle(e.target.value)}
                    className="rounded-lg border border-white/10 bg-ink-900 px-3 py-1.5 text-sm outline-none focus:border-tungsten-500"
                    placeholder="Film title"
                  />
                  <button
                    type="button"
                    disabled={busy || clips.length < 2}
                    onClick={onAssemble}
                    className="rounded-full bg-tungsten-500 px-4 py-1.5 text-sm font-medium text-ink-950 disabled:opacity-40"
                  >
                    Assemble movie ({clips.length})
                  </button>
                </div>
              </div>
              {clips.length === 0 ? (
                <p className="text-sm text-mist-400">
                  No scenes yet — describe scene 01 below and generate it.
                </p>
              ) : (
                <div className="flex gap-3 overflow-x-auto pb-1">
                  {clips.map((clip, index) => (
                    <div
                      key={clip.id}
                      className={`w-44 shrink-0 border ${
                        preview?.id === clip.id ? "border-tungsten-500" : "border-white/10"
                      } bg-ink-900`}
                    >
                      <button type="button" className="block w-full" onClick={() => setPreview(clip)}>
                        <video
                          src={assetFileUrl(clip.id)}
                          muted
                          className="aspect-video w-full object-cover"
                          onMouseEnter={(e) => e.currentTarget.play().catch(() => undefined)}
                          onMouseLeave={(e) => {
                            e.currentTarget.pause();
                            e.currentTarget.currentTime = 0;
                          }}
                        />
                      </button>
                      <div className="space-y-1 p-2">
                        <p className="text-xs font-medium text-tungsten-400">
                          Scene {String(index + 1).padStart(2, "0")}
                        </p>
                        <p className="line-clamp-2 text-[11px] text-mist-400">{clip.prompt}</p>
                        <div className="flex gap-1 pt-1">
                          <button
                            type="button"
                            className="rounded bg-ink-800 px-2 py-0.5 text-[10px] text-mist-200"
                            onClick={() => moveClip(index, -1)}
                          >
                            ←
                          </button>
                          <button
                            type="button"
                            className="rounded bg-ink-800 px-2 py-0.5 text-[10px] text-mist-200"
                            onClick={() => moveClip(index, 1)}
                          >
                            →
                          </button>
                          <button
                            type="button"
                            className="ml-auto rounded bg-ink-800 px-2 py-0.5 text-[10px] text-red-300"
                            onClick={() => removeClipFromTimeline(clip.id)}
                          >
                            hide
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {films.length > 0 && (
                <div className="mt-4 border-t border-white/10 pt-3">
                  <p className="mb-2 text-xs uppercase tracking-wider text-mist-400">Assembled movies</p>
                  <div className="flex flex-wrap gap-2">
                    {films.map((film) => (
                      <button
                        key={film.id}
                        type="button"
                        onClick={() => setPreview(film)}
                        className="rounded-full border border-tungsten-500/40 px-3 py-1 text-xs text-tungsten-400"
                      >
                        {film.prompt || "Film"}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {(events.length > 0 || error) && (
            <div className="mx-6 mb-2 max-h-28 overflow-auto rounded-lg border border-white/5 bg-ink-950/60 p-3 font-mono text-xs text-mist-400">
              {error && <p className="text-red-300">{error}</p>}
              {events.map((ev, i) => (
                <p key={`${ev}-${i}`}>{ev}</p>
              ))}
            </div>
          )}

          <form
            onSubmit={onGenerate}
            className="prompt-glow m-6 rounded-2xl border border-white/10 bg-ink-900/90 p-3 transition"
          >
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder={
                mode === "movie"
                  ? model === "multi-agent"
                    ? "Paste a full movie idea — agents plan every scene, then we assemble the film…"
                    : `Scene ${nextSceneLabel}: what happens, where, camera move, mood…`
                  : mode === "video"
                    ? "Cinematic drone shot over a rainy neon city at night, slow push-in…"
                    : "Describe the frame, lighting, and mood…"
              }
              className="w-full resize-none bg-transparent px-2 py-2 text-mist-100 outline-none placeholder:text-mist-400"
            />
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="rounded-lg border border-white/10 bg-ink-950 px-3 py-2 text-sm"
              >
                {MODELS[mode].map((m) => (
                  <option key={m} value={m}>
                    {mode === "movie" && m === "scene-clip"
                      ? "scene-clip (one at a time)"
                      : mode === "movie" && m === "multi-agent"
                        ? "multi-agent (auto all scenes)"
                        : m}
                  </option>
                ))}
              </select>
              <span className="text-xs text-mist-400">{costHint}</span>
              <button
                type="submit"
                disabled={busy || !prompt.trim()}
                className="ml-auto rounded-full bg-tungsten-500 px-6 py-2 text-sm font-medium text-ink-950 hover:bg-tungsten-400 disabled:opacity-50"
              >
                {busy
                  ? "Working…"
                  : mode === "video"
                    ? "Generate video"
                    : mode === "movie" && model !== "multi-agent"
                      ? `Add scene ${nextSceneLabel}`
                      : mode === "movie"
                        ? "Generate full movie"
                        : "Generate"}
              </button>
            </div>
          </form>
        </section>
      </div>
    </div>
  );
}
