<script lang="ts">
	import type { Visual } from '#lib/models/types.ts';
	import { kindStyle } from '#lib/models/kinds.ts';

	let { visual }: { visual: Visual } = $props();
	const cols = $derived(visual.lanes.length);
</script>

<!-- Each lane is a column with a vertical rail; a step sits on the rail of the state it transforms. -->
<div class="grid gap-x-4" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">
	{#each visual.lanes as lane (lane.id)}
		<div class="flex items-center justify-center gap-2 pb-2 font-mono text-xs {kindStyle[lane.kind].text}">
			<span class="size-2 rounded-full {kindStyle[lane.kind].dot}"></span>
			{lane.label}
		</div>
	{/each}

	{#each visual.steps as step, i (i)}
		{#if step.lane === 'all'}
			<div class="py-1.5" style="grid-column: 1 / -1;">
				<div class="rounded-lg border-2 border-classic-crimson-500 bg-classic-crimson-100 px-3 py-2 text-center">
					{@render body(step)}
				</div>
			</div>
		{:else}
			{#each visual.lanes as lane (lane.id)}
				<div class="relative flex justify-center py-1.5">
					<span class="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 {kindStyle[lane.kind].dot} opacity-50"></span>
					{#if lane.id === step.lane}
						<div class="relative w-full rounded-lg border-2 px-3 py-2 text-center {kindStyle[lane.kind].border} {kindStyle[lane.kind].bg}">
							{@render body(step)}
						</div>
					{/if}
				</div>
			{/each}
		{/if}
	{/each}
</div>

{#if visual.repeat}
	<p class="mt-3 text-center text-xs text-space-indigo-800">Repeated: {visual.repeat}</p>
{/if}
<p class="mt-2 text-center text-xs text-space-indigo-700">
	<span class="rounded bg-space-indigo-500 px-1.5 py-0.5 font-mono text-white">+</span> marks a step whose output is added
	back onto the state (residual connection).
</p>

{#snippet body(step: Visual['steps'][number])}
	<div class="text-sm font-medium text-white">
		{#if step.residual}
			<span class="mr-1 rounded bg-space-indigo-500 px-1.5 py-0.5 font-mono text-xs">+</span>
		{/if}
		{step.label}
	</div>
	{#if step.note}
		<div class="text-xs text-space-indigo-900/80">{step.note}</div>
	{/if}
	{#if step.shape}
		<div class="mt-0.5 font-mono text-[11px] text-space-indigo-800">{step.shape}</div>
	{/if}
{/snippet}
