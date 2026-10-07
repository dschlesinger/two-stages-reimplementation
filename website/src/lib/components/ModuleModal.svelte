<script lang="ts">
	import type { ModuleSpec } from '#lib/models/types.ts';
	import { kindLabel, kindStyle } from '#lib/models/kinds.ts';
	import ModuleVisual from './ModuleVisual.svelte';
	import Equation from './Equation.svelte';
	import Pseudocode from './Pseudocode.svelte';

	let { module = $bindable() }: { module: ModuleSpec | null } = $props();

	const tabs = ['Visual', 'Math', 'PyTorch'] as const;
	let tab = $state<(typeof tabs)[number]>('Visual');
	let dialog: HTMLDialogElement;

	$effect(() => {
		if (module && !dialog.open) {
			tab = 'Visual';
			dialog.showModal();
		} else if (!module && dialog.open) {
			dialog.close();
		}
	});
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<dialog
	bind:this={dialog}
	onclose={() => (module = null)}
	onclick={(e) => e.target === dialog && dialog.close()}
	class="m-auto w-[min(56rem,calc(100vw-2rem))] rounded-xl border border-space-indigo-500 bg-space-indigo-300 p-0 text-space-indigo-900 shadow-2xl backdrop:bg-space-indigo-100/80"
>
	{#if module}
		<div class="flex max-h-[85vh] flex-col">
			<header class="border-b border-space-indigo-500 px-6 pt-5">
				<div class="flex items-start justify-between gap-4">
					<div>
						<div class="flex items-center gap-2 text-xs {kindStyle[module.kind].text}">
							<span class="size-2 rounded-full {kindStyle[module.kind].dot}"></span>
							{kindLabel[module.kind]}
						</div>
						<h2 class="mt-1 text-xl font-semibold text-white">{module.title}</h2>
						<p class="text-sm text-space-indigo-800">{module.subtitle}</p>
					</div>
					<button
						onclick={() => dialog.close()}
						aria-label="Close"
						class="rounded-md px-2 py-1 text-lg leading-none text-space-indigo-800 hover:bg-space-indigo-500 hover:text-white"
					>
						&times;
					</button>
				</div>

				<dl class="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-xs">
					<dt class="text-space-indigo-700">in</dt>
					<dd>{module.in}</dd>
					<dt class="text-space-indigo-700">out</dt>
					<dd>{module.out}</dd>
				</dl>

				<div role="tablist" class="mt-4 flex gap-1">
					{#each tabs as name (name)}
						<button
							role="tab"
							aria-selected={tab === name}
							onclick={() => (tab = name)}
							class="-mb-px rounded-t-md border border-b-0 px-4 py-2 text-sm transition-colors {tab === name
								? 'border-space-indigo-500 bg-space-indigo-200 text-white'
								: 'border-transparent text-space-indigo-800 hover:text-white'}"
						>
							{name}
						</button>
					{/each}
				</div>
			</header>

			<div role="tabpanel" class="overflow-y-auto bg-space-indigo-200 px-6 py-5">
				{#if tab === 'Visual'}
					<p class="mb-4 text-sm leading-relaxed">{module.summary}</p>
					{#if module.paper}
						<p class="mb-5 rounded-md border-l-4 border-classic-crimson-500 bg-space-indigo-300 px-3 py-2 text-sm leading-relaxed">
							<span class="font-semibold text-classic-crimson-700">In the paper.</span>
							{module.paper}
						</p>
					{/if}
					<ModuleVisual visual={module.visual} />
				{:else if tab === 'Math'}
					<div class="space-y-5">
						{#each module.math as eq, i (i)}
							<div>
								{#if eq.label}
									<div class="text-xs font-semibold tracking-wide text-space-indigo-700 uppercase">{eq.label}</div>
								{/if}
								<Equation tex={eq.tex} />
								{#if eq.note}
									<p class="text-xs text-space-indigo-800">{eq.note}</p>
								{/if}
							</div>
						{/each}
					</div>
				{:else}
					<p class="mb-3 text-xs leading-relaxed text-space-indigo-800">
						PyTorch, rewritten for clarity rather than copied from the reference implementation. Assumes
						<code class="text-white">import torch</code>, <code class="text-white">from torch import nn</code> and
						<code class="text-white">import torch.nn.functional as F</code>. Padding masks, dropout and memory chunking are left
						out.
					</p>
					<Pseudocode code={module.code} />
					<div class="mt-4 text-xs">
						<div class="font-semibold tracking-wide text-space-indigo-700 uppercase">Source</div>
						<ul class="mt-1 space-y-0.5">
							{#each module.sources as source (source.url)}
								<li>
									<a
										href={source.url}
										target="_blank"
										rel="noreferrer"
										class="font-mono text-classic-crimson-700 underline-offset-2 hover:underline">{source.symbol}</a
									>
									<span class="text-space-indigo-700">in {source.repo}, {source.file}</span>
								</li>
							{/each}
						</ul>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</dialog>
