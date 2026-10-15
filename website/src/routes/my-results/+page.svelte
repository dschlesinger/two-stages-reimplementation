<script lang="ts">
	import modulePatching from '#lib/assets/module_patching_success.png';
	import singleBlock from '#lib/assets/single_block_success.png';
	import ShapeComparison from '#lib/components/ShapeComparison.svelte';
	import { baseline, loadPdb, presets } from '#lib/chargeShapes.ts';

	const shapeTests = [
		{
			preset: 'clamp',
			title: 'Clamp: opposite charges at the two ends',
			summary: 'If the charge direction acts like charge, the two ends should attract and fold the chain back on itself.'
		},
		{
			preset: 'splay',
			title: 'Splay: the same charge at both ends',
			summary: 'Like charges should repel, so the two segments should move apart and the chain should straighten.'
		},
		{
			preset: 'zigzag',
			title: 'Zigzag: alternating charges on three segments',
			summary: 'Neighbouring segments carry opposite charges, so each should pull toward the next, giving a meander.'
		}
	];
</script>

<div class="max-w-4xl">
	<h1 class="text-2xl font-semibold text-white">My results</h1>

	<h2 class="mt-6 text-lg font-semibold text-white">Full patching (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		Patching the donor hairpin's sequence and pairwise representations into a helical target at all 48 trunk blocks produces
		a hairpin in 1,865 of 4,870 cases (38.3%), matching the 38.3% the paper reports for the standard intra mask. None of the
		unpatched targets contain a hairpin.
	</p>

	<figure class="mt-4">
		<img
			src={modulePatching}
			alt="Bar chart of hairpin transfer success rate by patched module. Paper: trunk with touch mask 62.5%, trunk with standard intra mask 38.3%, structure module 4.6%, input intervention 2.6%, ESM encoder 0.6%. Our reproduction: trunk with standard intra mask 38.3%, unpatched target 0.0%."
			width="1600"
			height="919"
			class="h-auto w-full rounded-md border border-space-indigo-500"
		/>
		<figcaption class="mt-2 text-xs leading-relaxed text-space-indigo-700">
			Hairpin transfer success by patched module, after Fig. 12 of the paper. The paper's values are read off its figure; the
			touch mask, structure module, input intervention and ESM encoder experiments have not been reproduced yet.
		</figcaption>
	</figure>

	<h2 class="mt-8 text-lg font-semibold text-white">Single-block patching (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		For the 1,865 cases where full patching worked, the sequence or pairwise representation alone is patched at one trunk block
		at a time. Patching the sequence works only early: 61.3% at block 0, falling to almost nothing by block 8. Patching the
		pairwise representation does nothing early and peaks at 30.7% at block 34. The hairpin is carried by the sequence state in
		the first blocks and by the pairwise state later on, the two stages of the paper's title.
	</p>

	<figure class="mt-4">
		<img
			src={singleBlock}
			alt="Line chart of hairpin success rate against patched trunk block. Sequence patching starts at 61% at block 0 and drops to near zero by block 8. Pairwise patching is near zero until about block 20, rises to a peak of 31% at block 34, then falls off by block 47."
			width="1600"
			height="700"
			class="h-auto w-full rounded-md border border-space-indigo-500"
		/>
		<figcaption class="mt-2 text-xs leading-relaxed text-space-indigo-700">
			Hairpin success by patched trunk block, after Fig. 1 of the paper. Rates are out of the 1,865 full-patching successes, not
			all 4,870 cases; over all cases, sequence patching at block 0 succeeds 23.5% of the time.
		</figcaption>
	</figure>

	<h2 class="mt-8 text-lg font-semibold text-white">Charge steering on poly-alanine (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		A 40-residue poly-alanine chain has no charged residues. During trunk blocks 0–14, the positive or negative charge direction
		found in the sequence state is added to chosen segments, scaled to a number of standard deviations of its projection. Each
		steered fold is shown next to the same chain folded without steering, from the same batch. This goes beyond what the paper
		tests: its steering acts on helical motifs in real proteins, while a uniform chain gives the model no other signal, so read
		the geometry more than the pLDDT or DSSP.
	</p>

	{#each shapeTests as test (test.preset)}
		{@const p = presets.get(test.preset)}
		{#if p}
			<ShapeComparison title={test.title} summary={test.summary} segments={p.segments} {baseline} runs={p.runs} {loadPdb} />
		{/if}
	{/each}
</div>
