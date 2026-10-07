<script lang="ts">
	import type { ModelSpec, TensorSpec } from '#lib/models/types.ts';
	import { kindLabel, kindStyle } from '#lib/models/kinds.ts';

	let {
		tensor = $bindable(),
		model,
		onnavigate
	}: { tensor: TensorSpec | null; model: ModelSpec; onnavigate: (id: string) => void } = $props();

	let dialog: HTMLDialogElement;

	$effect(() => {
		if (tensor && !dialog.open) dialog.showModal();
		else if (!tensor && dialog.open) dialog.close();
	});

	function name(id: string) {
		const m = model.modules.find((m) => m.id === id);
		if (m) return m.subtitle && m.title.startsWith('Triangle') ? `${m.title} (${m.subtitle.toLowerCase()})` : m.title;
		return model.tensors.find((t) => t.id === id)?.label ?? id;
	}
	const producers = $derived(tensor ? model.edges.filter((e) => e.to === tensor!.id).map((e) => e.from) : []);
	const consumers = $derived(tensor ? model.edges.filter((e) => e.from === tensor!.id).map((e) => e.to) : []);
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<dialog
	bind:this={dialog}
	onclose={() => (tensor = null)}
	onclick={(e) => e.target === dialog && dialog.close()}
	class="m-auto w-[min(36rem,calc(100vw-2rem))] rounded-xl border border-space-indigo-500 bg-space-indigo-300 p-0 text-space-indigo-900 shadow-2xl backdrop:bg-space-indigo-100/80"
>
	{#if tensor}
		<div class="px-6 py-5">
			<div class="flex items-start justify-between gap-4">
				<div>
					<div class="flex items-center gap-2 text-xs {kindStyle[tensor.kind].text}">
						<span class="size-2 rounded-full {kindStyle[tensor.kind].dot}"></span>
						Tensor &middot; {kindLabel[tensor.kind]}
					</div>
					<h2 class="mt-1 text-xl font-semibold text-white">{tensor.label}</h2>
					<p class="font-mono text-sm text-white">{tensor.shape}</p>
				</div>
				<button
					onclick={() => dialog.close()}
					aria-label="Close"
					class="rounded-md px-2 py-1 text-lg leading-none text-space-indigo-800 hover:bg-space-indigo-500 hover:text-white"
				>
					&times;
				</button>
			</div>

			<p class="mt-4 text-sm leading-relaxed">{tensor.description}</p>

			<h3 class="mt-5 text-xs font-semibold tracking-wide text-space-indigo-700 uppercase">Dimensions</h3>
			<dl class="mt-1 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
				{#each tensor.axes as axis (axis.dim)}
					<dt class="font-mono text-white">{axis.dim}</dt>
					<dd>{axis.meaning}</dd>
				{/each}
			</dl>

			<div class="mt-5 grid grid-cols-2 gap-4 text-sm">
				{@render links('Produced by', producers, 'Model input')}
				{@render links('Used by', consumers, 'Model output')}
			</div>
		</div>
	{/if}
</dialog>

{#snippet links(title: string, ids: string[], empty: string)}
	<div>
		<h3 class="text-xs font-semibold tracking-wide text-space-indigo-700 uppercase">{title}</h3>
		{#if ids.length}
			<ul class="mt-1 space-y-1">
				{#each ids as id (id)}
					<li>
						<button
							class="text-left text-classic-crimson-700 underline-offset-2 hover:underline"
							onclick={() => {
								dialog.close();
								// Let this dialog's close handler run before the next one opens.
								setTimeout(() => onnavigate(id));
							}}
						>
							{name(id)}
						</button>
					</li>
				{/each}
			</ul>
		{:else}
			<p class="mt-1 text-space-indigo-800">{empty}</p>
		{/if}
	</div>
{/snippet}
