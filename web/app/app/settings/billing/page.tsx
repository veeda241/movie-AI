"use client";

import { useEffect, useState } from "react";
import { Plan, User, client } from "@/lib/api";

export default function BillingSettingsPage() {
  const [user, setUser] = useState<User | null>(null);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    client.me().then(setUser);
    client.billing.plans().then(setPlans);
  }, []);

  async function upgrade(plan: string) {
    const res = await client.billing.checkout({ plan });
    if (res.url && res.mode === "stripe") window.location.href = res.url;
    else {
      setMessage(`Upgraded to ${res.status}. Balance: ${res.credits}`);
      setUser(await client.me());
    }
  }

  async function topup(pack: number) {
    const res = await client.billing.checkout({ credit_pack: pack });
    if (res.url && res.mode === "stripe") window.location.href = res.url;
    else {
      setMessage(`Top-up applied. Balance: ${res.credits}`);
      setUser(await client.me());
    }
  }

  async function portal() {
    const res = await client.billing.portal();
    if (res.url) window.location.href = res.url;
  }

  return (
    <div className="p-8">
      <h1 className="font-display text-4xl text-mist-100">Billing</h1>
      <p className="mt-2 text-mist-400">
        Plan: <span className="text-tungsten-400">{user?.plan}</span> · Credits:{" "}
        <span className="text-tungsten-400">{user?.credit_balance}</span>
      </p>
      {message && <p className="mt-4 text-tungsten-400">{message}</p>}
      <div className="mt-8 flex flex-wrap gap-3">
        {plans
          .filter((p) => p.id !== "enterprise")
          .map((p) => (
            <button
              key={p.id}
              onClick={() => upgrade(p.id)}
              className="rounded-full bg-tungsten-500 px-5 py-2 text-sm font-medium text-ink-950"
            >
              Upgrade {p.name} (${p.price_monthly}/mo)
            </button>
          ))}
        <button onClick={() => topup(100)} className="rounded-full border border-white/20 px-5 py-2 text-sm">
          +100 credits
        </button>
        <button onClick={() => topup(500)} className="rounded-full border border-white/20 px-5 py-2 text-sm">
          +500 credits
        </button>
        <button onClick={portal} className="rounded-full border border-white/20 px-5 py-2 text-sm">
          Customer portal
        </button>
      </div>
      <p className="mt-6 max-w-xl text-sm text-mist-400">
        Without Stripe keys, upgrades run in demo mode and grant credits locally. With Stripe configured, Checkout and
        webhooks handle live billing.
      </p>
    </div>
  );
}
