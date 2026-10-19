<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import type { GLViewer } from '3dmol';
	import { centroid, superpose, type Vec3 } from '#lib/superpose.ts';
	import type { Segment, ShapeRow } from '#lib/chargeShapes.ts';

	let {
		title,
		summary,
		segments,
		baseline,
		runs,
		loadPdb
	}: {
		title: string;
		summary: string;
		segments: Segment[];
		baseline: ShapeRow;
		runs: ShapeRow[];
		loadPdb: (label: string) => Promise<string>;
	} = $props();

	const COLORS = { chain: '#9f98c9', '+': '#d7263d', '-': '#1b998b', background: '#1b182e' };

	let index = $state(untrack(() => Math.max(0, runs.findIndex((r) => r.strength === 0.5))));
	const run = $derived(runs[index]);
	let rmsd = $state<number | null>(null);

	let baselineEl = $state<HTMLDivElement>();
	let steeredEl = $state<HTMLDivElement>();
	let viewers = $state.raw<{ baseline: GLViewer; steered: GLViewer; baselineCa: Vec3[] } | null>(null);
	let failed = $state(false);

	const pairKey = (a: Segment, b: Segment) => `dist_${a.start}_${a.end}__${b.start}_${b.end}`;
	const pairs = $derived(segments.flatMap((a, i) => segments.slice(i + 1).map((b) => [a, b] as const)));
	const span = (s: Segment) => `${s.start + 1}–${s.end}`;

	/** Load a PDB into a viewer, centred on its CA centroid and, if given, rotated onto the baseline. */
	function show(viewer: GLViewer, pdb: string, reference?: Vec3[]) {
		viewer.removeAllModels();
		const model = viewer.addModel(pdb, 'pdb');
		const atoms = model.selectedAtoms({});
		const ca = atoms.filter((a) => a.atom === 'CA').map((a) => [a.x!, a.y!, a.z!] as Vec3);
		const c = centroid(ca);
		let rotation = [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
		];
		if (reference) {
			const fit = superpose(ca, reference);
			rotation = fit.rotation;
			rmsd = fit.rmsd;
		}
		for (const a of atoms) {
			const p = [a.x! - c[0], a.y! - c[1], a.z! - c[2]];
			[a.x, a.y, a.z] = rotation.map((row) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2]);
		}
		viewer.setStyle({}, { cartoon: { color: COLORS.chain } });
		for (const s of segments) viewer.setStyle({ resi: `${s.start + 1}-${s.end}` }, { cartoon: { color: COLORS[s.sign] } });
		return ca.map((p) => p.map((v, i) => v - c[i]) as Vec3);
	}

	onMount(() => {
		let disposed = false;
		(async () => {
			try {
				const mol = await import('3dmol');
				const config = { backgroundColor: COLORS.background, antialias: true };
				const baselineViewer = mol.createViewer(baselineEl, config);
				const steeredViewer = mol.createViewer(steeredEl, config);
				const baselineCa = show(baselineViewer, await loadPdb(baseline.label));
				baselineViewer.linkViewer(steeredViewer);
				steeredViewer.linkViewer(baselineViewer);
				if (!disposed) viewers = { baseline: baselineViewer, steered: steeredViewer, baselineCa };
			} catch (e) {
				console.error(e);
				failed = true;
			}
		})();
		return () => {
			disposed = true;
			viewers?.baseline.clear();
			viewers?.steered.clear();
		};
	});

	$effect(() => {
		const label = run.label;
		const v = viewers;
		if (!v) return;
		loadPdb(label).then((pdb) => {
			if (label !== run.label) return;
			show(v.steered, pdb, v.baselineCa);
			// Frame whichever fold is larger so both fit, then give the other the same camera.
			const [big, small] = run.radiusOfGyration > baseline.radiusOfGyration ? [v.steered, v.baseline] : [v.baseline, v.steered];
			big.zoomTo();
			small.setView(big.getView());
			v.baseline.render();
			v.steered.render();
		});
	});

	const fmt = (x: number, digits = 1) => x.toFixed(digits);
</script>

<section class="mt-6 rounded-lg border border-space-indigo-500 bg-space-indigo-300 p-4">
	<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
		<h3 class="font-semibold text-white">{title}</h3>
		<div class="flex flex-wrap gap-3 font-mono text-xs">
			{#each segments as s (s.start)}
				<span class="flex items-center gap-1.5" style="color: {COLORS[s.sign]}">
					<span class="size-2 rounded-full" style="background: {COLORS[s.sign]}"></span>
					{s.sign === '+' ? 'positive' : 'negative'} {span(s)}
				</span>
			{/each}
		</div>
	</div>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">{summary}</p>

	<div class="mt-4 grid gap-3 sm:grid-cols-2">
		{#each [{ el: 'baseline', name: 'Unsteered', row: baseline }, { el: 'steered', name: `Steered, ${run.strength}σ`, row: run }] as panel (panel.el)}
			<div>
				<div class="mb-1 flex justify-between text-xs text-space-indigo-800">
					<span class="font-medium text-white">{panel.name}</span>
					<span>pLDDT {fmt(panel.row.plddt * 100, 0)}</span>
				</div>
				{#if panel.el === 'baseline'}
					<div bind:this={baselineEl} class="relative h-72 w-full overflow-hidden rounded-md border border-space-indigo-500"></div>
				{:else}
					<div bind:this={steeredEl} class="relative h-72 w-full overflow-hidden rounded-md border border-space-indigo-500"></div>
				{/if}
			</div>
		{/each}
	</div>
	{#if failed}
		<p class="mt-2 text-xs text-classic-crimson-700">The 3D viewer could not start; your browser may not support WebGL.</p>
	{:else}
		<p class="mt-2 text-xs text-space-indigo-700">
			Drag to rotate, scroll to zoom; the two views move together. The steered fold is superposed on the unsteered one by CA atoms.
		</p>
	{/if}

	<label class="mt-4 block">
		<span class="flex justify-between text-xs text-space-indigo-800">
			<span>Steering strength</span>
			<span class="font-mono text-white">{run.strength}σ</span>
		</span>
		<input type="range" min="0" max={runs.length - 1} step="1" bind:value={index} class="mt-1 w-full accent-tiger-flame-500" />
		<!-- Each tick sits under its slider stop; 8px is half the thumb, which never travels past the track ends. -->
		<span class="relative block h-4 font-mono text-xs text-space-indigo-700">
			{#each runs as r, i (r.label)}
				<span
					class="absolute -translate-x-1/2 {i === index ? 'text-white' : ''}"
					style="left: calc(8px + (100% - 16px) * {i / (runs.length - 1)})">{r.strength}</span
				>
			{/each}
		</span>
	</label>

	<table class="mt-4 w-full text-sm">
		<thead class="text-xs text-space-indigo-700">
			<tr>
				<th class="py-1 text-left font-normal"></th>
				<th class="py-1 text-right font-normal">Unsteered</th>
				<th class="py-1 text-right font-normal">{run.strength}σ</th>
			</tr>
		</thead>
		<tbody class="font-mono text-space-indigo-900">
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">Radius of gyration</td>
				<td class="py-1 text-right">{fmt(baseline.radiusOfGyration)} Å</td>
				<td class="py-1 text-right">{fmt(run.radiusOfGyration)} Å</td>
			</tr>
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">End-to-end distance</td>
				<td class="py-1 text-right">{fmt(baseline.endToEnd)} Å</td>
				<td class="py-1 text-right">{fmt(run.endToEnd)} Å</td>
			</tr>
			{#each pairs as [a, b] (pairKey(a, b))}
				<tr class="border-t border-space-indigo-500">
					<td class="py-1 font-sans text-space-indigo-800">Mean distance {span(a)} ↔ {span(b)}</td>
					<td class="py-1 text-right">{fmt(baseline.distances[pairKey(a, b)])} Å</td>
					<td class="py-1 text-right">{fmt(run.distances[pairKey(a, b)])} Å</td>
				</tr>
			{/each}
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">CA RMSD to unsteered</td>
				<td class="py-1 text-right">–</td>
				<td class="py-1 text-right">{rmsd !== null ? `${fmt(rmsd)} Å` : failed ? '–' : '…'}</td>
			</tr>
		</tbody>
	</table>

	<div class="mt-3 overflow-x-auto font-mono text-xs leading-relaxed">
		<div class="whitespace-pre text-space-indigo-700">unsteered {baseline.dssp}</div>
		<div class="whitespace-pre text-space-indigo-900">{run.strength.toString().padStart(4)}σ     {run.dssp}</div>
	</div>
</section>
