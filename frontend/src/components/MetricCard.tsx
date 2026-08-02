export function MetricCard({ name, value }: { name: string; value: number }) {
  return (
    <div className="stamp-shadow rounded border-2 border-rule bg-surface p-4 text-center">
      <p className="font-wire text-[10px] uppercase tracking-wide text-ink-muted">{name}</p>
      <p className="font-display text-3xl text-accent">{value.toFixed(2)}</p>
    </div>
  );
}
