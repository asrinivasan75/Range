import { Hero } from "@/components/landing/hero";
import { Leaderboard } from "@/components/landing/leaderboard";
import { GameTheory } from "@/components/landing/game-theory";
import { HowItWorks } from "@/components/landing/how-it-works";
import { Architecture } from "@/components/landing/architecture";
import { Features } from "@/components/landing/features";
import { CTA } from "@/components/landing/cta";

export default function LandingPage() {
  return (
    <main className="relative">
      <Hero />
      <Leaderboard />
      <GameTheory />
      <HowItWorks />
      <Architecture />
      <Features />
      <CTA />
    </main>
  );
}
