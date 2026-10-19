<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import type { GLViewer } from '3dmol';
	import { centroid, superpose, type Vec3 } from '#lib/superpose.ts';
	import { BASELINE, loadPdb, type Protein } from '#lib/pairwiseScaling.ts';

	let { proteins }: { proteins: Protein[] } = $props();

	const COLORS = { background: '#1b182e' };
	// Consecutive CA atoms sit 3.8 Å apart in a real chain; past this the viewer no longer sees them as bonded.
	const MAX_BOND = 4.5;

	let proteinIndex = $state(0);
	const protein = $derived(proteins[proteinIndex]);
	const baseline = $derived(protein.runs.find((r) => r.scale === BASELINE)!);
	let index = $state(untrack(() => proteins[0].runs.length - 1));
	const run = $derived(protein.runs[index]);
	let rmsd = $state<number | null>(null);

	let baselineEl = $state<HTMLDivElement>();
	let scaledEl = $state<HTMLDivElement>();
	let viewers = $state.raw<{ baseline: GLViewer; scaled: GLViewer } | null>(null);
	let baselineCa = $state.raw<Vec3[] | null>(null);
	let failed = $state(false);
	let broken = $state(false);

	/** Position along the chain as a hex colour, blue at the N terminus to red at the C terminus. */
	function spectrum(i: number, n: number) {
		const hue = 240 * (1 - i / (n - 1));
		const [s, l] = [0.85, 0.55];
		const channel = (k: number) => {
			const a = (k + hue / 30) % 12;
			return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(a - 3, 9 - a, 1));
		};
		return [0, 8, 4].reduce((hex, k) => hex * 256 + Math.round(channel(k) * 255), 0);
	}

	/** Load a PDB into a viewer, centred on its CA centroid and, if given, rotated onto the reference. */
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
		viewer.removeAllShapes();
		// Colour by position along the chain, so a fold can be followed from one scale to the next.
		const placed = atoms.filter((a) => a.atom === 'CA').map((a) => [a.x!, a.y!, a.z!] as Vec3);
		const stretched = placed.some((p, i) => i > 0 && Math.hypot(...p.map((v, k) => v - placed[i - 1][k])) > MAX_BOND);
		if (reference) broken = stretched;
		if (reference) {
			// The scaled fold is always the CA trace as balls and sticks, so it reads the same whether or not the backbone is stretched.
			viewer.setStyle({}, {});
			placed.forEach((p, i) => {
				const color = spectrum(i, placed.length);
				viewer.addSphere({ center: { x: p[0], y: p[1], z: p[2] }, radius: 0.7, color });
				if (i > 0) {
					const q = placed[i - 1];
					viewer.addCylinder({ start: { x: q[0], y: q[1], z: q[2] }, end: { x: p[0], y: p[1], z: p[2] }, radius: 0.2, color });
				}
			});
		} else {
			viewer.setStyle({}, { cartoon: { colorfunc: (a: { resi: number }) => spectrum(a.resi - 1, placed.length) } });
		}
		return ca.map((p) => p.map((v, i) => v - c[i]) as Vec3);
	}

	onMount(() => {
		let disposed = false;
		(async () => {
			try {
				const mol = await import('3dmol');
				const config = { backgroundColor: COLORS.background, antialias: true };
				const baselineViewer = mol.createViewer(baselineEl, config);
				const scaledViewer = mol.createViewer(scaledEl, config);
				baselineViewer.linkViewer(scaledViewer);
				scaledViewer.linkViewer(baselineViewer);
				if (!disposed) viewers = { baseline: baselineViewer, scaled: scaledViewer };
			} catch (e) {
				console.error(e);
				failed = true;
			}
		})();
		return () => {
			disposed = true;
			viewers?.baseline.clear();
			viewers?.scaled.clear();
		};
	});

	// The unscaled fold changes only with the protein.
	$effect(() => {
		const name = protein.name;
		const v = viewers;
		if (!v) return;
		loadPdb(name, BASELINE).then((pdb) => {
			if (name !== protein.name) return;
			baselineCa = show(v.baseline, pdb);
			v.baseline.render();
		});
	});

	$effect(() => {
		const name = protein.name;
		const scale = run.scale;
		const v = viewers;
		const reference = baselineCa;
		if (!v || !reference) return;
		loadPdb(name, scale).then((pdb) => {
			if (name !== protein.name || scale !== run.scale) return;
			show(v.scaled, pdb, reference);
			// Frame the unscaled fold, backed off to leave room for the largest scale, so the camera is fixed for the protein
			// and the unscaled fold keeps its size as the slider moves. The scaled view takes the same camera.
			const widest = Math.max(...protein.runs.map((r) => r.radiusOfGyration));
			v.baseline.zoomTo();
			v.baseline.zoom(baseline.radiusOfGyration / widest);
			v.scaled.setView(v.baseline.getView());
			v.baseline.render();
			v.scaled.render();
		});
	});

	const fmt = (x: number, digits = 1) => x.toFixed(digits);
	const signed = (x: number) => `${x > 0 ? '+' : ''}${fmt(x)}`;
</script>

<section class="mt-6 rounded-lg border border-space-indigo-500 bg-space-indigo-300 p-4">
	<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
		<h3 class="font-semibold text-white">Scale z before the structure module</h3>
		<div class="flex flex-wrap gap-1 font-mono text-xs" role="group" aria-label="Protein">
			{#each proteins as p, i (p.name)}
				<button
					type="button"
					aria-pressed={i === proteinIndex}
					class="rounded px-2 py-0.5 {i === proteinIndex
						? 'bg-tiger-flame-500 text-white'
						: 'bg-space-indigo-500 text-space-indigo-900 hover:text-white'}"
					onclick={() => (proteinIndex = i)}>{p.name}</button
				>
			{/each}
		</div>
	</div>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		The trunk runs once and its pairwise output is multiplied by the scale factor just before the structure module reads it. The
		sequence input is left alone. Scale 1 is the ordinary fold, shown on the left.
	</p>

	<div class="mt-4 grid gap-3 sm:grid-cols-2">
		{#each [{ el: 'baseline', name: 'Unscaled', row: baseline }, { el: 'scaled', name: `z × ${run.scale}`, row: run }] as panel (panel.el)}
			<div>
				<div class="mb-1 flex justify-between text-xs text-space-indigo-800">
					<span class="font-medium text-white">{panel.name}</span>
					<span>pLDDT {fmt(panel.row.plddt * 100, 0)}</span>
				</div>
				{#if panel.el === 'baseline'}
					<div bind:this={baselineEl} class="relative h-72 w-full overflow-hidden rounded-md border border-space-indigo-500"></div>
				{:else}
					<div bind:this={scaledEl} class="relative h-72 w-full overflow-hidden rounded-md border border-space-indigo-500"></div>
				{/if}
			</div>
		{/each}
	</div>
	{#if failed}
		<p class="mt-2 text-xs text-classic-crimson-700">The 3D viewer could not start; your browser may not support WebGL.</p>
	{:else}
		<p class="mt-2 text-xs text-space-indigo-700">
			Drag to rotate, scroll to zoom; the two views move together. Colour runs from the N terminus (blue) to the C terminus
			(red). The unscaled fold is drawn as a ribbon and the scaled fold as its CA trace in balls and sticks; the scaled fold is
			superposed on the unscaled one by CA atoms.
		</p>
		{#if broken}
			<p class="mt-1 text-xs text-space-indigo-700">
				At this scale the consecutive CA atoms are no longer 3.8 Å apart, so the backbone is stretched and could not be drawn as
				a cartoon. DSSP finds no secondary structure in it for the same reason, even though pLDDT stays high at scales above 1.
			</p>
		{/if}
	{/if}

	<label class="mt-4 block">
		<span class="flex justify-between text-xs text-space-indigo-800">
			<span>Scale of z</span>
			<span class="font-mono text-white">× {run.scale}</span>
		</span>
		<input type="range" min="0" max={protein.runs.length - 1} step="1" bind:value={index} class="mt-1 w-full accent-tiger-flame-500" />
		<!-- Each tick sits under its slider stop; 8px is half the thumb, which never travels past the track ends. -->
		<span class="relative block h-4 font-mono text-xs text-space-indigo-700">
			{#each protein.runs as r, i (r.scale)}
				<span
					class="absolute -translate-x-1/2 {i === index ? 'text-white' : ''}"
					style="left: calc(8px + (100% - 16px) * {i / (protein.runs.length - 1)})">{r.scale}</span
				>
			{/each}
		</span>
	</label>

	<table class="mt-4 w-full text-sm">
		<thead class="text-xs text-space-indigo-700">
			<tr>
				<th class="py-1 text-left font-normal"></th>
				<th class="py-1 text-right font-normal">Unscaled</th>
				<th class="py-1 text-right font-normal">z × {run.scale}</th>
			</tr>
		</thead>
		<tbody class="font-mono text-space-indigo-900">
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">Mean pairwise CA distance</td>
				<td class="py-1 text-right">{fmt(baseline.meanCaDistance)} Å</td>
				<td class="py-1 text-right">{fmt(run.meanCaDistance)} Å ({signed(run.deviation)})</td>
			</tr>
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">Radius of gyration</td>
				<td class="py-1 text-right">{fmt(baseline.radiusOfGyration)} Å</td>
				<td class="py-1 text-right">{fmt(run.radiusOfGyration)} Å</td>
			</tr>
			<tr class="border-t border-space-indigo-500">
				<td class="py-1 font-sans text-space-indigo-800">CA RMSD to unscaled</td>
				<td class="py-1 text-right">–</td>
				<td class="py-1 text-right">{rmsd !== null ? `${fmt(rmsd)} Å` : failed ? '–' : '…'}</td>
			</tr>
		</tbody>
	</table>

	<div class="mt-3 overflow-x-auto font-mono text-xs leading-relaxed">
		<div class="whitespace-pre text-space-indigo-700">unscaled {baseline.dssp}</div>
		<div class="whitespace-pre text-space-indigo-900">{`×${run.scale}`.padStart(8)} {run.dssp}</div>
	</div>
</section>
