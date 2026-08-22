"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { client, setToken } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await client.register(email, password, name);
      setToken(res.access_token);
      router.push("/app");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="film-grain flex min-h-screen items-center justify-center px-6">
      <form onSubmit={onSubmit} className="w-full max-w-md space-y-5 rounded-2xl border border-white/10 bg-ink-900/80 p-8 backdrop-blur">
        <Link href="/" className="font-display text-3xl text-mist-100">
          Movie Flow
        </Link>
        <h1 className="text-lg text-mist-200">Create your studio account</h1>
        <p className="text-sm text-mist-400">Includes {2000} free credits to start generating.</p>
        {error && <p className="text-sm text-red-300">{error}</p>}
        <input
          className="w-full rounded-xl border border-white/10 bg-ink-950 px-4 py-3 outline-none focus:border-tungsten-500"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          className="w-full rounded-xl border border-white/10 bg-ink-950 px-4 py-3 outline-none focus:border-tungsten-500"
          placeholder="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          className="w-full rounded-xl border border-white/10 bg-ink-950 px-4 py-3 outline-none focus:border-tungsten-500"
          placeholder="Password (min 6)"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={6}
        />
        <button
          disabled={loading}
          className="w-full rounded-full bg-tungsten-500 py-3 font-medium text-ink-950 hover:bg-tungsten-400 disabled:opacity-60"
        >
          {loading ? "Creating…" : "Create account"}
        </button>
        <p className="text-sm text-mist-400">
          Already have an account?{" "}
          <Link href="/login" className="text-tungsten-400">
            Sign in
          </Link>
        </p>
      </form>
    </main>
  );
}
