import csv from '#lib/assets/pairwise_scaling/scaling.csv?raw';

// Output of scripts/scale_pairwise.py --target both, copied from outputs/scaling/. Only the z folds are bundled.
const pdbs = import.meta.glob<string>('./assets/pairwise_scaling/*.pdb', { query: '?raw', import: 'default' });

export function loadPdb(protein: string, scale: number): Promise<string> {
	return pdbs[`./assets/pairwise_scaling/${protein}_z${scale}.pdb`]();
}

export type ScaleRow = {
	scale: number;
	meanCaDistance: number;
	radiusOfGyration: number;
	plddt: number;
	deviation: number;
	dssp: string;
};
export type Protein = { name: string; length: number; runs: ScaleRow[] };

function parse(): Protein[] {
	const [header, ...lines] = csv.trim().split('\n');
	const columns = header.split(',');
	const records = lines.map((line) => Object.fromEntries(line.split(',').map((v, i) => [columns[i], v])));

	const proteins = new Map<string, Protein>();
	for (const r of records) {
		if (r.scaled !== 'z') continue;
		if (!proteins.has(r.protein)) proteins.set(r.protein, { name: r.protein, length: +r.length, runs: [] });
		proteins.get(r.protein)!.runs.push({
			scale: +r.scale,
			meanCaDistance: +r.mean_ca_distance,
			radiusOfGyration: +r.radius_of_gyration,
			plddt: +r.plddt,
			deviation: +r.distance_deviation,
			dssp: r.dssp
		});
	}
	for (const p of proteins.values()) p.runs.sort((a, b) => a.scale - b.scale);
	return [...proteins.values()];
}

export const proteins = parse();
export const BASELINE = 1;
