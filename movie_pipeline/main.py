from __future__ import annotations

try:
    from movie_pipeline.pipeline.orchestrator import Orchestrator
except ImportError:  # pragma: no cover - direct execution fallback
    from pipeline.orchestrator import Orchestrator


def main() -> None:
    idea = input("Enter your movie idea: ")
    orchestrator = Orchestrator()
    try:
        result = orchestrator.run(idea)
    except (RuntimeError, ValueError) as exc:
        print(f"\nPipeline failed: {exc}")
        raise SystemExit(1) from exc

    if result:
        runtime = result[-1].edit_plan["total_duration_sec"]
        print(f"\nMovie assembled: {len(result)} scenes, runtime: {runtime}s")
        for packet in result:
            print(f"  Scene {packet.scene_number}: {packet.title} -> {packet.video_path or 'no video'}")
    else:
        print("\nMovie assembled: 0 scenes, runtime: 0s")


if __name__ == "__main__":
    main()
