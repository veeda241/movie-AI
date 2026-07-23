import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Movie Flow",
  description: "AI creative studio for images, video, and multi-agent preproduction.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
