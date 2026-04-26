"use client";

import { useCallback, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8001";

type Result = {
  label: string;
  confidence: number;
  probability_fake: number;
  backend?: string;
  branch?: string;
};

function isMediaFile(f: File) {
  const t = f.type;
  return (
    t.startsWith("image/") ||
    t.startsWith("video/") ||
    t.startsWith("audio/")
  );
}

export default function DeepfakeDetector() {
  const [preview, setPreview] = useState<string | null>(null);
  const [previewKind, setPreviewKind] = useState<"image" | "video" | "audio" | null>(
    null,
  );
  const [fileName, setFileName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  const resetForNewFile = useCallback(() => {
    setError(null);
    setResult(null);
  }, []);

  const onFile = useCallback(
    (file: File | null) => {
      if (!file || !isMediaFile(file)) return;
      resetForNewFile();
      setFileName(file.name);
      const url = URL.createObjectURL(file);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      if (file.type.startsWith("video/")) setPreviewKind("video");
      else if (file.type.startsWith("audio/")) setPreviewKind("audio");
      else setPreviewKind("image");
    },
    [resetForNewFile],
  );

  const analyze = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body,
      });
      const data = (await res.json().catch(() => null)) as
        | Result
        | { detail?: string }
        | null;
      if (!res.ok) {
        const msg =
          typeof data === "object" && data && "detail" in data
            ? String(data.detail)
            : res.statusText;
        throw new Error(msg || `Request failed (${res.status})`);
      }
      if (
        !data ||
        typeof (data as Result).label !== "string" ||
        typeof (data as Result).confidence !== "number"
      ) {
        throw new Error("Unexpected response from server");
      }
      setResult(data as Result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="mx-auto w-full max-w-xl space-y-8">
      <label
        className="group relative flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/80 px-6 py-14 transition hover:border-cyan-500/50 hover:bg-cyan-50/30 dark:border-zinc-700 dark:bg-zinc-900/40 dark:hover:border-cyan-400/40 dark:hover:bg-cyan-950/20"
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
        onDrop={(e) => {
          e.preventDefault();
          const f = e.dataTransfer.files?.[0];
          if (f && isMediaFile(f)) {
            onFile(f);
            void analyze(f);
          }
        }}
      >
        <input
          type="file"
          accept="image/*,video/*,audio/*"
          className="absolute inset-0 cursor-pointer opacity-0"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f && isMediaFile(f)) {
              onFile(f);
              void analyze(f);
            }
            e.target.value = "";
          }}
        />
        <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
          Drop image, video, or audio — multimodal API when finetuned
        </span>
        <span className="mt-1 text-xs text-zinc-500">
          Images use visual (+ learned null-audio); video fuses frame + mel;
          audio uses the auxiliary mel head
        </span>
      </label>

      {preview && previewKind && (
        <div className="overflow-hidden rounded-xl border border-zinc-200 bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900">
          {previewKind === "image" && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={preview}
              alt="Preview"
              className="mx-auto max-h-72 w-auto object-contain"
            />
          )}
          {previewKind === "video" && (
            <video
              src={preview}
              className="mx-auto max-h-72 w-full object-contain"
              controls
              muted
            />
          )}
          {previewKind === "audio" && (
            <audio src={preview} className="w-full px-4 py-6" controls />
          )}
          {fileName && (
            <p className="truncate border-t border-zinc-200 px-3 py-2 text-xs text-zinc-500 dark:border-zinc-800">
              {fileName}
            </p>
          )}
        </div>
      )}

      {loading && (
        <p className="text-center text-sm text-zinc-500">Running detector…</p>
      )}

      {error && (
        <div
          className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200"
          role="alert"
        >
          {error}
        </div>
      )}

      {result && !loading && (
        <div
          className={`rounded-xl border px-5 py-6 ${
            result.label === "FAKE"
              ? "border-amber-200 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-950/40"
              : "border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-950/40"
          }`}
        >
          <p className="text-xs font-medium uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
            Verdict
          </p>
          <p className="mt-1 text-3xl font-semibold tracking-tight">
            {result.label === "FAKE" ? (
              <span className="text-amber-800 dark:text-amber-200">
                Likely fake
              </span>
            ) : (
              <span className="text-emerald-800 dark:text-emerald-200">
                Likely real
              </span>
            )}
          </p>
          {(result.backend || result.branch) && (
            <p className="mt-2 font-mono text-xs text-zinc-500 dark:text-zinc-400">
              {[result.backend, result.branch].filter(Boolean).join(" · ")}
            </p>
          )}
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">Confidence</dt>
              <dd className="font-mono font-medium">
                {(result.confidence * 100).toFixed(1)}%
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500 dark:text-zinc-400">P(fake)</dt>
              <dd className="font-mono font-medium">
                {(result.probability_fake * 100).toFixed(1)}%
              </dd>
            </div>
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-zinc-500 dark:text-zinc-500">
            Probabilistic only — not legal proof. Backbone: DeepGuard (HF);
            optional FF++ multimodal finetune uses frozen visual features + mel
            fusion and a separate audio-only head for clips without video.
          </p>
        </div>
      )}
    </div>
  );
}
