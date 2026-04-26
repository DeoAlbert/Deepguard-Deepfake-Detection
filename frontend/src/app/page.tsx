import DeepfakeDetector from "./DeepfakeDetector";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center px-4 py-16 sm:py-24">
      <header className="mb-12 max-w-xl text-center">
        <h1 className="text-balance text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
          DeepGuard
        </h1>
        <p className="mt-3 text-pretty text-zinc-600 dark:text-zinc-400">
          Upload an image, a video, or audio. With a finetuned checkpoint, the
          API fuses DeepGuard visual features with log-mel audio and can run an
          audio-only branch; otherwise it falls back to the Hugging Face
          DeepGuard image model.
        </p>
      </header>
      <DeepfakeDetector />
    </div>
  );
}
