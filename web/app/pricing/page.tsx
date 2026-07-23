"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { client, Plan } from "@/lib/api";

export default function PricingPage() {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    client.billing.plans().then(setPlans).catch(() => setPlans([]));
  }, []);

  async function checkout(planId: string) {
    try {
      const res = await client.billing.checkout({ plan: planId });
      if (res.url && res.mode === "stripe") {
        window.location.href = res.url;
      } else {
        setMessage(`Demo upgrade applied: ${res.status}. Credits: ${res.credits}`);
      }
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Checkout failed — sign in first.");
    }
  }

  return (
    <main className="film-grain min-h-screen px-6 py-12 md:px-12">
      <div className="mb-10 flex items-center justify-between">
        <Link href="/" className="font-display text-3xl text-mist-100">
          Movie Flow
        </Link>
        <Link href="/app" className="text-sm text-tungsten-400">
          Open studio
        </Link>
      </div>
      <h1 className="font-display text-5xl text-mist-100">Plans built for production teams</h1>
      <p className="mt-3 max-w-xl text-mist-400">
        Seat-based access for planning, plus consumption credits for cloud renders.
      </p>
      {message && <p className="mt-4 text-tungsten-400">{message}</p>}
      <div className="mt-12 grid gap-6 md:grid-cols-3">
        {plans.map((plan) => (
          <div key={plan.id} className="border border-white/10 bg-ink-900/70 p-6">
            <h2 className="font-display text-3xl text-mist-100">{plan.name}</h2>
            <p className="mt-2 text-2xl text-tungsten-400">
              {plan.price_monthly > 0 ? `$${plan.price_monthly}/seat/mo` : "Custom"}
            </p>
            <ul className="mt-6 space-y-2 text-sm text-mist-200">
              {plan.features.map((f) => (
                <li key={f}>— {f}</li>
              ))}
            </ul>
            {plan.id === "enterprise" ? (
              <a
                href="mailto:sales@movieflow.studio"
                className="mt-8 inline-block rounded-full border border-mist-400/40 px-5 py-2 text-sm"
              >
                Contact sales
              </a>
            ) : (
              <button
                onClick={() => checkout(plan.id)}
                className="mt-8 rounded-full bg-tungsten-500 px-5 py-2 text-sm font-medium text-ink-950"
              >
                Choose {plan.name}
              </button>
            )}
          </div>
        ))}
      </div>
    </main>
  );
}
