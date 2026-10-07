<script lang="ts">
	import { untrack } from 'svelte';
	import { SvelteFlow, Background, Controls, ControlButton, type Edge, type Node } from '@xyflow/svelte';
	import '@xyflow/svelte/dist/style.css';
	import type { ModelSpec, ModuleSpec, TensorSpec } from '#lib/models/types.ts';
	import { kindStyle } from '#lib/models/kinds.ts';
	import ModuleNode from './ModuleNode.svelte';
	import GroupNode from './GroupNode.svelte';
	import TensorNode from './TensorNode.svelte';
	import ModuleModal from './ModuleModal.svelte';
	import TensorModal from './TensorModal.svelte';

	let { model }: { model: ModelSpec } = $props();

	const nodeTypes = { module: ModuleNode, block: GroupNode, tensor: TensorNode };

	// The spec is static, so the chart is built once. Parents must come before their children.
	const spec = untrack(() => model);
	let nodes = $state.raw<Node[]>(
		[...spec.modules]
			.sort((a, b) => Number(!!a.parent) - Number(!!b.parent))
			.map((m): Node => ({
				id: m.id,
				type: m.size ? 'block' : 'module',
				position: m.position,
				parentId: m.parent,
				data: { module: m },
				...(m.size ? { width: m.size.width, height: m.size.height } : {})
			}))
			.concat(
				spec.tensors.map((t) => ({
					id: t.id,
					type: 'tensor',
					position: t.position,
					parentId: t.parent,
					data: { tensor: t }
				}))
			)
	);
	let edges = $state.raw<Edge[]>(
		spec.edges.map((e) => ({
			id: `${e.from}-${e.to}`,
			source: e.from,
			target: e.to,
			label: e.label,
			type: 'smoothstep',
			style: `stroke: ${kindStyle[e.kind].stroke}; stroke-width: 2;`
		}))
	);

	let selected = $state<ModuleSpec | null>(null);
	let selectedTensor = $state<TensorSpec | null>(null);

	function open(id: string) {
		selected = spec.modules.find((m) => m.id === id) ?? null;
		selectedTensor = spec.tensors.find((t) => t.id === id) ?? null;
	}

	// The modals live inside the shell so they stay visible while it is full screen.
	let shell: HTMLDivElement;
	let fullscreen = $state(false);

	function toggleFullscreen() {
		if (document.fullscreenElement) document.exitFullscreen();
		else shell.requestFullscreen();
	}
</script>

<svelte:document onfullscreenchange={() => (fullscreen = document.fullscreenElement === shell)} />

<div bind:this={shell} class="bg-space-indigo-200">
	<div class="flow overflow-hidden {fullscreen ? 'h-screen' : 'h-[78vh] rounded-xl border border-space-indigo-500'}">
		<SvelteFlow
			bind:nodes
			bind:edges
			{nodeTypes}
			colorMode="dark"
			fitView
			fitViewOptions={{ nodes: model.initialView.map((id) => ({ id })), padding: 0.1 }}
			minZoom={0.1}
			panOnScroll
			nodesDraggable={false}
			nodesConnectable={false}
			elementsSelectable={false}
			proOptions={{ hideAttribution: false }}
			onnodeclick={({ node }) => open(node.id)}
		>
			<Background />
			<Controls showLock={false}>
				<ControlButton
					onclick={toggleFullscreen}
					title={fullscreen ? 'Exit full screen (Esc)' : 'Full screen'}
					aria-label={fullscreen ? 'Exit full screen' : 'Full screen'}
				>
					<svg class="fs-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
						{#if fullscreen}
							<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />
						{:else}
							<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
						{/if}
					</svg>
				</ControlButton>
			</Controls>
		</SvelteFlow>
	</div>

	<ModuleModal bind:module={selected} />
	<TensorModal bind:tensor={selectedTensor} {model} onnavigate={open} />
</div>

<style>
	.flow :global(.svelte-flow) {
		--xy-background-color: var(--color-space-indigo-200);
		--xy-edge-label-background-color: var(--color-space-indigo-200);
		--xy-edge-label-color: var(--color-space-indigo-900);
	}
	.flow :global(.svelte-flow__controls-button svg.fs-icon) {
		fill: none;
	}
	.flow :global(.svelte-flow__node) {
		cursor: pointer;
	}
</style>
