"use client";

import Link from "next/link";
import { motion } from "framer-motion";

export default function LandingPage() {
  return (
    <main className="film-grain relative min-h-screen overflow-hidden">
      <div
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            "linear-gradient(to bottom, transparent 40%, #0a0b0d 100%), url(https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=2000&q=80)",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      />
      <nav className="relative z-10 flex items-center justify-between px-6 py-5 md:px-12">
        <span className="font-display text-2xl tracking-wide text-mist-100">Movie Flow</span>
        <div className="flex items-center gap-4 text-sm text-mist-200">
          <Link href="/pricing" className="hover:text-tungsten-400">
            Pricing
          </Link>
          <Link href="/login" className="hover:text-tungsten-400">
            Sign in
          </Link>
          <Link
            href="/register"
            className="rounded-full bg-tungsten-500 px-4 py-2 font-medium text-ink-950 transition hover:bg-tungsten-400"
          >
            Start creating
          </Link>
        </div>
      </nav>

      <section className="relative z-10 flex min-h-[85vh] flex-col items-start justify-center px-6 md:px-12">
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="font-display text-6xl leading-none text-mist-100 md:text-8xl"
        >
          Movie Flow
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1 }}
          className="mt-6 max-w-xl text-xl text-mist-200 md:text-2xl"
        >
          From prompt to picture, clip, and cut — one studio for generative preproduction.
        </motion.h1>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mt-10 flex flex-wrap gap-4"
        >
          <Link
            href="/register"
            className="rounded-full bg-tungsten-500 px-7 py-3 font-medium text-ink-950 transition hover:bg-tungsten-400"
          >
            Start creating
          </Link>
          <Link
            href="/pricing"
            className="rounded-full border border-mist-400/40 px-7 py-3 text-mist-100 transition hover:border-tungsten-400 hover:text-tungsten-400"
          >
            View pricing
          </Link>
        </motion.div>
      </section>
    </main>
  );
}
