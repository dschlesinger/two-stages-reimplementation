import csv from '#lib/assets/charge_shapes/shapes.csv?raw';

// Output of scripts/charge_shapes.py --preset all, copied from outputs/charge/shapes/.
const pdbs = import.meta.glob<string>('./assets/charge_shapes/*.pdb', { query: '?raw', import: 'default' });

export function loadPdb(label: string): Promise<string> {
	return pdbs[`./assets/charge_shapes/${label}.pdb`]();
}

export type Segment = { start: number; end: number; sign: '+' | '-' };
export type ShapeRow = {
	label: string;
	strength: number;
	plddt: number;
	radiusOfGyration: number;
	endToEnd: number;
	distances: Record<string, number>;
	dssp: string;
};
type Preset = { name: string; segments: Segment[]; runs: ShapeRow[] };

/** "4-14+_26-36-" -> 0-based, end-excluded segments, as the script writes them. */
function parseSegments(spec: string): Segment[] {
	return spec.split('_').map((s) => {
		const [, start, end, sign] = s.match(/^(\d+)-(\d+)([+-])$/)!;
		return { start: +start, end: +end, sign: sign as '+' | '-' };
	});
}

function parse() {
	const [header, ...lines] = csv.trim().split('\n');
	const columns = header.split(',');
	const records = lines.map((line) => Object.fromEntries(line.split(',').map((v, i) => [columns[i], v])));

	const toRow = (r: Record<string, string>): ShapeRow => ({
		label: r.label,
		strength: +r.strength,
		plddt: +r.plddt,
		radiusOfGyration: +r.radius_of_gyration,
		endToEnd: +r.end_to_end,
		distances: Object.fromEntries(columns.filter((c) => c.startsWith('dist_') && r[c] !== '').map((c) => [c, +r[c]])),
		dssp: r.dssp
	});

	const baseline = toRow(records.find((r) => r.label === 'baseline')!);
	const presets = new Map<string, Preset>();
	for (const r of records) {
		if (!r.preset) continue;
		if (!presets.has(r.preset)) presets.set(r.preset, { name: r.preset, segments: parseSegments(r.segments), runs: [] });
		presets.get(r.preset)!.runs.push(toRow(r));
	}
	for (const p of presets.values()) p.runs.sort((a, b) => a.strength - b.strength);
	return { baseline, presets };
}

export const { baseline, presets } = parse();
