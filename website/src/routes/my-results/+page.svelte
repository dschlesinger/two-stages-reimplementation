<script lang="ts">
	import modulePatching from '#lib/assets/module_patching_success.png';
	import singleBlock from '#lib/assets/single_block_success.png';
	import scalingDistance from '#lib/assets/scaling_distance.png';
	import pathwayContributions from '#lib/assets/pathway_contributions.png';
	import ShapeComparison from '#lib/components/ShapeComparison.svelte';
	import ScalingComparison from '#lib/components/ScalingComparison.svelte';
	import { baseline, loadPdb, presets } from '#lib/chargeShapes.ts';
	import { proteins } from '#lib/pairwiseScaling.ts';

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

	<h2 class="mt-8 text-lg font-semibold text-white">Scaling the pairwise representation (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		Each of five proteins is folded once per scale factor from 0 to 2 in steps of 0.25, with the trunk's output multiplied by
		that factor just before the structure module reads it. From 0.25 upward, scaling z moves the output distances
		monotonically: the mean pairwise CA distance is 10.2 Å below the unscaled fold at 0.25, 3.6 Å below at 0.75, 2.7 Å above at
		1.25 and 6.8 Å above at 2. Scaling s changes nothing, within 0.03 Å at every scale. The paper reports the same two results,
		that the structure module takes its distances from z and not from s. The curve is steeper below 1 than above it. At 0 and
		0.25 the input carries almost no pair information and the folds collapse (pLDDT 0.32 and 0.46), and the distance at 0 is
		slightly larger than at 0.25, so the trend does not hold all the way down.
		Part of the change is the chain itself changing length: consecutive CA atoms are 3.8 Å apart at scale 1, about 3.1 Å at
		0.75, and about 4.5, 5.1, 5.7 and 6.4 Å at 1.25, 1.5, 1.75 and 2. The folds at scales other than 1 are not physical
		backbones, and pLDDT stays high from 0.75 up regardless.
	</p>

	<figure class="mt-4">
		<img
			src={scalingDistance}
			alt="Line chart of the change in mean pairwise CA distance against the scale factor. Scaling z falls to about minus 10 angstroms at scales 0 and 0.25, rises through minus 7.5 at 0.5 and 0 at scale 1 to plus 7 angstroms at scale 2. Scaling s stays flat at 0."
			width="1600"
			height="920"
			class="h-auto w-full rounded-md border border-space-indigo-500"
		/>
		<figcaption class="mt-2 text-xs leading-relaxed text-space-indigo-700">
			Change in mean pairwise CA distance from the unscaled fold, after Fig. 6c of the paper. Thin lines are single proteins, thick
			lines the mean. The gradient at the unscaled point (Fig. 6d) is not computed.
		</figcaption>
	</figure>

	<ScalingComparison {proteins} />

	<h2 class="mt-8 text-lg font-semibold text-white">Pathway contributions across the trunk (ESMFold)</h2>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">
		At every trunk block, the norm of the update each pathway writes: the sequence-to-pair update added to z, and the
		pair-to-sequence update added to s, taken as the sequence attention output, which is where z reaches s. Each pathway's 48
		values are divided by their maximum and averaged over 517 proteins. This does not yet reproduce Fig. 4b. Pair-to-sequence
		is small in the middle and rises late, as in the paper, but it also spikes at block 0. Sequence-to-pair stays between 0.3
		and 0.5 through the middle and peaks at block 44, where the paper has it peaking in the first ten blocks and declining. The
		curves are consistent across proteins, so the jaggedness comes from the blocks themselves. These are raw updates; ESMFold's
		blocks are pre-norm, so dividing each by the size of the state it is added to may be the comparison the paper makes, and
		that has not been run.
	</p>

	<figure class="mt-4">
		<img
			src={pathwayContributions}
			alt="Line chart of normalized update norm against trunk block. Sequence-to-pair rises from 0.2 at block 0 to about 0.5 by block 6, stays between 0.3 and 0.5 through the middle, and peaks at 1.0 at block 44. Pair-to-sequence starts at 1.0 at block 0, drops to near 0 and stays low through the middle, then rises to 0.8 at block 43. A dotted line for the attention bias alone is noisy, between 0.15 and 1.0."
			width="1600"
			height="840"
			class="h-auto w-full rounded-md border border-space-indigo-500"
		/>
		<figcaption class="mt-2 text-xs leading-relaxed text-space-indigo-700">
			Update norm per trunk block, each pathway divided by its own maximum, after Fig. 4b of the paper. The shaded band is one
			standard deviation over proteins. The dotted line is the pair-to-sequence attention bias alone, a weaker proxy. Not a
			reproduction of the paper's curve yet.
		</figcaption>
	</figure>
</div>
