<script lang="ts">
	import ModelFlow from '#lib/components/ModelFlow.svelte';
	import { models } from '#lib/models/index.ts';
	import { kindLabel, kindStyle } from '#lib/models/kinds.ts';
	import type { Kind } from '#lib/models/types.ts';

	let model = $state(models[0]);

	// Only list the colours this model's chart actually uses.
	const order: Kind[] = ['input', 'lm', 'msa', 'seq', 'pair', 'cross', 'struct', 'head'];
	const legend = $derived(order.filter((k) => [...model.modules, ...model.tensors].some((n) => n.kind === k)));
</script>

<div role="tablist" aria-label="Model" class="mb-5 flex gap-1 border-b border-space-indigo-500">
	{#each models as m (m.name)}
		<button
			role="tab"
			aria-selected={model === m}
			onclick={() => (model = m)}
			class="-mb-px rounded-t-md border border-b-0 px-4 py-2 text-sm font-medium transition-colors {model === m
				? 'border-space-indigo-500 bg-space-indigo-300 text-white'
				: 'border-transparent text-space-indigo-800 hover:text-white'}"
		>
			{m.name}
		</button>
	{/each}
</div>

<div class="mb-4 max-w-4xl">
	<h1 class="text-2xl font-semibold text-white">{model.name}</h1>
	<p class="mt-1 text-sm leading-relaxed text-space-indigo-800">{model.description}</p>
	{#if model.notes}
		<p class="mt-1 text-xs leading-relaxed text-space-indigo-700">{model.notes}</p>
	{/if}
</div>

<ul class="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-space-indigo-800">
	{#each legend as kind (kind)}
		<li class="flex items-center gap-1.5">
			<span class="size-2.5 rounded-full {kindStyle[kind].dot}"></span>
			{kindLabel[kind]}
		</li>
	{/each}
</ul>

<p class="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-space-indigo-700">
	<span class="rounded-md border-2 border-space-indigo-700 px-2 py-0.5 text-space-indigo-900">Box</span> is a module: click it for
	its components, math and PyTorch code.
	<span class="rounded-full border border-dashed border-space-indigo-700 px-2 py-0.5 font-mono text-space-indigo-900">Pill</span>
	is a tensor with its shape: click it for what it holds. L is the sequence length, S the number of aligned sequences. Scroll
	to pan; the corner button in the chart goes full screen.
</p>

{#key model.name}
	<ModelFlow {model} />
{/key}
