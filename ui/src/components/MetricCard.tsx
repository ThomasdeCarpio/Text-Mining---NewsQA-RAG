export function MetricCard({ name, value }: { name: string; value: number }) {
  return (
    <div className="rounded-lg border border-gray-200 p-4 text-center">
      <p className="text-sm text-gray-500">{name}</p>
      <p className="text-2xl font-semibold text-gray-900">{value.toFixed(2)}</p>
    </div>
  );
}
