"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { client, setToken, User } from "@/lib/api";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    client
      .me()
      .then(setUser)
      .catch(() => {
        setToken(null);
        router.replace("/login");
      });
  }, [router, pathname]);

  function logout() {
    setToken(null);
    router.push("/");
  }

  const nav = [
    { href: "/app", label: "Create" },
    { href: "/app/projects", label: "Projects" },
    { href: "/app/settings/billing", label: "Billing" },
    { href: "/app/settings/team", label: "Team" },
  ];

  return (
    <div className="film-grain flex min-h-screen">
      <aside className="flex w-56 flex-col border-r border-white/10 bg-ink-950/80 p-5">
        <Link href="/" className="font-display text-2xl text-mist-100">
          Movie Flow
        </Link>
        <nav className="mt-10 flex flex-col gap-2 text-sm">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`rounded-lg px-3 py-2 transition ${
                pathname === item.href ? "bg-ink-800 text-tungsten-400" : "text-mist-200 hover:bg-ink-800/60"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto space-y-2 text-sm text-mist-400">
          {user && (
            <>
              <p className="text-mist-100">{user.name || user.email}</p>
              <p>
                {user.credit_balance} credits · {user.plan}
              </p>
            </>
          )}
          <button onClick={logout} className="text-left text-tungsten-400 hover:underline">
            Sign out
          </button>
        </div>
      </aside>
      <div className="flex-1 overflow-auto">{children}</div>
    </div>
  );
}
